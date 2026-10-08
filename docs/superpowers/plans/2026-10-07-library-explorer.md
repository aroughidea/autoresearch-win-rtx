# Library Report and Training Decisions Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the library (six baselines plus agent sessions) with the data a browser cannot compute, and build the `/training/` explorer in llm-explorables that shows Growth, Compare, Tokens and Evolution, and loads a learner's own runs.

**Architecture:** In the worked example, `library/report.py` writes `index.json` (the collections and their files), `tokens.json` (each tokenizer's split of the four prompts, characters per token, unused share) and `copies.json` (spans copied verbatim from Folktales' training text). In llm-explorables, `scripts/sync-training-library.mjs` copies the library into `training/library/` at a pinned commit, and `/training/` is a static `noCanvas` page: `data.js` (pure, tested), `views.js` (DOM), `ownRuns.js` (GitHub and folder loading), `Main.js` (boot, state), `initUI.js` (header), D3 for the Evolution chart.

**Tech Stack:** Python 3 + pytest (worked example, `uv run`); plain classic-script JavaScript, p5 1.9.4 (`noCanvas`), D3 v7 (vendored), `node --test` with `test/helpers/sketch.js` (llm-explorables).

**Spec:** `docs/superpowers/specs/2026-10-06-training-decisions-journey-design.md`, sections "Component 3 design: library" and "Component 4 design: explorer" (TJ approved 2026-10-07).

## Global Constraints

- Scores (`val_bpb`, bits per byte, lower is better) compare only within one dataset; any view that shows two scores from different datasets says so instead of comparing them.
- The explorer is static: no sign-in, no `/api`, `<base href="/training/">`, data from its own folder.
- Every run is keyed by its file path (library-relative, or `yours:<owner>/<repo>/runs/<file>`), never by `run_id`: the six baselines share one `run_id` (they predate capture's rerun fix).
- Weights never enter git. The library publishes generated samples and scores only, never dataset text.
- Copy follows llm-explorables: plain sentences that say what you see; British spelling (colour, catalogue); Title Case page title "Training Decisions".
- llm-explorables delivery (AGENTS.md): feature branch, `npm test`, `git diff --check`, PR, READY Vercel preview checked in a browser.
- Measured noise: 0.003 `val_bpb` run to run (constant `NOISE_BPB`).

## Review Focus

1. **A learner's repo with a `results.tsv` but no run files** (an agent session from before capture): Evolution draws it; Growth and Compare say there are no run files instead of breaking. Test in Task 8 (`loadRepo` with an empty `runs/`).
2. **GitHub refuses** (private repo 404, anonymous rate limit 403): a readable message, the library stays usable. Tests in Task 8.
3. **A run with fewer snapshots** (capture stopped early, `capture_error`) or no final snapshot: the scrubber shows what exists. Test in Task 3 (`snapIndexAt`, `defaultCompareIndex` with unequal runs).
4. **`results.tsv` with CRLF line endings, blank or `missing` scores, crash rows:** parsed, crashes left out of the staircase. Tests in Task 3.
5. **A dropped folder that also holds `library/` or nested repos:** only the top folder's `runs/*.json` and `results.tsv` are taken. Test in Task 3 (`pickRepoFiles`).

---

## Part A — the library (autoresearch-win-rtx, worktree `../ar-library`, branch `library/baselines`)

### Task 1: `library/report.py`

**Files:**
- Create: `library/report.py`
- Test: `tests/test_library_report.py`

**Interfaces:**
- Consumes: `capture.PROMPTS`; `prepare.Tokenizer.from_directory(dataset=, tokenizer=)` (`.encode(str|list)`, `.decode(ids)`, `.get_vocab_size()`, `.source`); `prepare.text_iterator(dataset)` (training documents).
- Produces: `token_pieces(tokenizer, text) -> list[str]`; `vocab_stats(tokenizer, docs) -> {"chars_per_token", "unused_share"}`; `shingles(texts, n=8) -> set[int]`; `copied_spans(text, index, n=8) -> list[[start, end]]`; `collections(library) -> list[dict]`; `load_runs(library) -> dict[path, run]`; files `library/index.json`, `library/tokens.json`, `library/copies.json` with these shapes:
  - `index.json`: `{"schema": 1, "collections": [{"id", "title", "kind": "baselines"|"session", "note", "runs": [library-relative paths], "results": path|null}], "tokens": "tokens.json", "copies": "copies.json"}`
  - `tokens.json`: `{"prompts": [4], "sample_docs": int, "datasets": {dataset: {tokenizer: {"vocab_size", "source", "splits": [[pieces] x4], "chars_per_token", "unused_share"}}}}`
  - `copies.json`: `{library-relative run path: [[[start, end], ...] per prompt] per snapshot}` (Folktales runs only)

- [ ] **Step 1: Write the failing tests**

```python
"""library/report.py: token pieces, vocabulary statistics, copied spans, the collection index."""
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("library_report", ROOT / "library" / "report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class FakeTok:
    """Words with their leading space as tokens. Like Phi-3, decoding a lone token drops its leading space."""
    source = "fake"

    def __init__(self):
        self.vocab = {}

    def encode(self, text):
        if isinstance(text, list):
            return [self.encode(t) for t in text]
        return [self.vocab.setdefault(p, len(self.vocab)) for p in re.findall(r" ?\S+", text)]

    def decode(self, ids):
        inverse = {v: k for k, v in self.vocab.items()}
        text = "".join(inverse[i] for i in ids)
        return text.lstrip(" ") if len(ids) == 1 else text

    def get_vocab_size(self):
        return 10


def test_token_pieces_keep_each_tokens_leading_space():
    assert report.token_pieces(FakeTok(), "Once upon a time") == ["Once", " upon", " a", " time"]


def test_vocab_stats():
    stats = report.vocab_stats(FakeTok(), ["a b", "a c"])
    assert stats == {"chars_per_token": 1.5, "unused_share": 0.7}


def test_copied_spans_mark_eight_word_repeats_and_merge_them():
    index = report.shingles(["the king had three sons and the youngest was a fool"], n=4)
    text = "Then the king had three sons, all tall."
    spans = report.copied_spans(text, index, n=4)
    assert [text[a:b] for a, b in spans] == ["the king had three sons"]
    assert report.copied_spans("nothing here repeats at all today", index, n=4) == []


def test_collections_list_runs_and_scoreboards(tmp_path):
    lib = tmp_path / "library"
    (lib / "baselines" / "runs").mkdir(parents=True)
    (lib / "baselines" / "collection.json").write_text(json.dumps({"title": "Baselines", "kind": "baselines"}), encoding="utf-8")
    (lib / "baselines" / "runs" / "tinystories-own.json").write_text("{}", encoding="utf-8")
    (lib / "sessions" / "may").mkdir(parents=True)
    (lib / "sessions" / "may" / "collection.json").write_text(json.dumps({"title": "May", "kind": "session", "note": "n"}), encoding="utf-8")
    (lib / "sessions" / "may" / "results.tsv").write_text("timestamp\tcommit\n", encoding="utf-8")
    assert report.collections(lib) == [
        {"id": "baselines", "title": "Baselines", "kind": "baselines", "note": "",
         "runs": ["baselines/runs/tinystories-own.json"], "results": None},
        {"id": "sessions/may", "title": "May", "kind": "session", "note": "n",
         "runs": [], "results": "sessions/may/results.tsv"},
    ]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -q tests/test_library_report.py`
Expected: FAIL (`library/report.py` does not exist).

- [ ] **Step 3: Write `library/report.py`**

```python
"""What the explorer cannot compute in a browser, written next to the library.

Run from the repo root once the library's runs exist:  uv run library/report.py
Writes library/index.json (collections and their files), library/tokens.json (each tokenizer's
split of the four prompts, characters per token, unused share) and library/copies.json (spans
of Folktales samples repeated word for word from the training text). Reads the local cache; never
changes the active dataset or tokenizer, so it is safe while a training session runs.
"""
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capture import PROMPTS  # noqa: E402
from prepare import Tokenizer, text_iterator  # noqa: E402

LIBRARY = ROOT / "library"
DATASETS = ("tinystories", "folktales")
TOKENIZERS = ("own", "phi3", "gpt2")
SAMPLE_DOCS = 100_000
COPY_WORDS = 8
# TinyStories' training text is too large to index here; the check found copying in Folktales.
COPY_DATASETS = ("folktales",)
WORD = re.compile(r"[A-Za-z0-9']+")


def token_pieces(tokenizer, text):
    """The text cut into its tokens, each piece as it reads in the text. Decodes growing
    prefixes, because some tokenizers (Phi-3) drop a lone token's leading space."""
    ids = tokenizer.encode(text)
    pieces, previous = [], ""
    for i in range(1, len(ids) + 1):
        current = tokenizer.decode(ids[:i])
        pieces.append(current[len(previous):])
        previous = current
    return pieces


def vocab_stats(tokenizer, docs):
    """Characters per token, and the share of the vocabulary these documents never use."""
    seen, chars, tokens = set(), 0, 0
    for ids, doc in zip(tokenizer.encode(list(docs)), docs):
        seen.update(ids)
        chars += len(doc)
        tokens += len(ids)
    return {"chars_per_token": round(chars / max(tokens, 1), 2),
            "unused_share": round(1 - len(seen) / tokenizer.get_vocab_size(), 3)}


def _words(text):
    return [(m.start(), m.end(), m.group().lower()) for m in WORD.finditer(text)]


def shingles(texts, n=COPY_WORDS):
    """Hashes of every run of n consecutive words in the texts."""
    out = set()
    for text in texts:
        words = [w for _, _, w in _words(text)]
        for i in range(len(words) - n + 1):
            out.add(hash(" ".join(words[i:i + n])))
    return out


def copied_spans(text, index, n=COPY_WORDS):
    """[start, end) spans of text that repeat n or more consecutive indexed words, merged."""
    words = _words(text)
    spans = []
    for i in range(len(words) - n + 1):
        if hash(" ".join(w for _, _, w in words[i:i + n])) in index:
            start, end = words[i][0], words[i + n - 1][1]
            if spans and start <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], end)
            else:
                spans.append([start, end])
    return spans


def collections(library=LIBRARY):
    """Every folder under the library with a collection.json: title, kind, run files, scoreboard."""
    out = []
    for meta_path in sorted(Path(library).rglob("collection.json")):
        folder = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        results = folder / "results.tsv"
        out.append({
            "id": folder.relative_to(library).as_posix(),
            "title": meta["title"],
            "kind": meta["kind"],
            "note": meta.get("note", ""),
            "runs": sorted(p.relative_to(library).as_posix() for p in (folder / "runs").glob("*.json")),
            "results": results.relative_to(library).as_posix() if results.exists() else None,
        })
    return out


def load_runs(library=LIBRARY):
    runs = {}
    for c in collections(library):
        for rel in c["runs"]:
            runs[rel] = json.loads((Path(library) / rel).read_text(encoding="utf-8"))
    return runs


def tokens_report():
    out = {"prompts": list(PROMPTS), "sample_docs": SAMPLE_DOCS, "datasets": {}}
    for dataset in DATASETS:
        docs = list(itertools.islice(text_iterator(dataset), SAMPLE_DOCS))
        out["datasets"][dataset] = {}
        for name in TOKENIZERS:
            tok = Tokenizer.from_directory(dataset=dataset, tokenizer=name)
            out["datasets"][dataset][name] = {
                "vocab_size": tok.get_vocab_size(),
                "source": tok.source,
                "splits": [token_pieces(tok, p) for p in PROMPTS],
                **vocab_stats(tok, docs),
            }
            print(f"tokens: {dataset} / {name}: {out['datasets'][dataset][name]['chars_per_token']} chars per token")
    return out


def copies_report(runs):
    out = {}
    for dataset in COPY_DATASETS:
        mine = {rel: r for rel, r in runs.items() if r.get("dataset") == dataset}
        if not mine:
            continue
        index = shingles(text_iterator(dataset))
        for rel, run in mine.items():
            out[rel] = [[copied_spans(s, index) for s in snap.get("samples", [])] for snap in run.get("snapshots", [])]
            print(f"copies: {rel}: {sum(len(p) for snap in out[rel] for p in snap)} spans")
    return out


def _write(name, obj):
    (LIBRARY / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    index = {"schema": 1, "collections": collections(), "tokens": "tokens.json", "copies": "copies.json"}
    _write("index.json", index)
    _write("tokens.json", tokens_report())
    _write("copies.json", copies_report(load_runs()))
    print(f"index: {len(index['collections'])} collections, {sum(len(c['runs']) for c in index['collections'])} runs")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q tests/test_library_report.py` then `uv run pytest -q`
Expected: 4 passed; whole suite green.

- [ ] **Step 5: Commit**

```bash
git add library/report.py tests/test_library_report.py
git commit -m "library: report.py writes index, token splits and copied spans"
```

### Task 2: Publish the library

**Files:**
- Create: `library/README.md`, `library/baselines/collection.json`, `library/sessions/tinystories-may2026/collection.json`, `library/sessions/tinystories-may2026/results.tsv` (copy of master's 16 rows)
- Generated: `library/index.json`, `library/tokens.json`, `library/copies.json`
- Modify: `library/make_baselines.sh` (name results by pair, as the running fix did)

**Interfaces:**
- Consumes: Task 1's `report.py`; the six `library/baselines/runs/<dataset>-<tokenizer>.json`.
- Produces: the library at a merged commit on master, which Task 4 syncs from.

- [ ] **Step 1: Collections and README**

`library/baselines/collection.json`:
```json
{"title": "Six baselines: two datasets × three tokenizers", "kind": "baselines", "note": "The starter kit's recipe, five minutes each, no agent. Only the dataset and the tokenizer change."}
```
`library/sessions/tinystories-may2026/collection.json`:
```json
{"title": "Agent session, TinyStories, May 2026", "kind": "session", "note": "16 experiments by the agent on 23 May 2026, before run files existed: scores only, no writing."}
```
Copy `results.tsv` from master into `library/sessions/tinystories-may2026/results.tsv`.

`library/README.md` says what the library is, how each collection was made (commands), the measured noise, and the licences (TinyStories CDLA-Sharing-1.0; Phi-3 tokenizer MIT; GPT-2 MIT; Folktales' card says CC0 1.0, its source D. L. Ashliman's Folktexts carries its own copyright notice; only generated samples and scores are published). Weights stay in `library/checkpoints/` (ignored).

`library/make_baselines.sh`: replace the two `mv`/`cp` lines with
```bash
    mv "$run" "library/baselines/runs/$dataset-$tokenizer.json"
    cp checkpoint_pre_eval.pt "library/checkpoints/$dataset-$tokenizer.pt"
```

- [ ] **Step 2: Generate**

Run: `uv run library/report.py`
Expected: lines for six token reports and three Folktales copies reports, then `index: 2 collections, 6 runs`.

- [ ] **Step 3: Check**

```bash
python -c "import json; i=json.load(open('library/index.json')); print([(c['id'], len(c['runs']), c['results']) for c in i['collections']])"
python -c "import json; t=json.load(open('library/tokens.json',encoding='utf-8')); print({d:{k:(v['chars_per_token'],v['unused_share'],len(v['splits'][0])) for k,v in t['datasets'][d].items()} for d in t['datasets']})"
```
Expected: `baselines` 6 runs, `sessions/tinystories-may2026` 0 runs with its `results.tsv`; tokens close to the check's figures (TinyStories own 4.18 chars per token, about 5% unused; Phi-3 and GPT-2 about 70% unused).

- [ ] **Step 4: Commit, PR, merge** (TJ approved the branch → PR → merge flow for this repo)

```bash
git add library/ && git commit -m "library: six baselines, the May session, and the explorer's report"
git push -u origin library/baselines
gh pr create --repo aroughidea/autoresearch-win-rtx --base master --head library/baselines --title "Library: six baselines and the explorer's data" --body-file <body>
gh pr merge <n> --repo aroughidea/autoresearch-win-rtx --merge --delete-branch
```
The branch's `train.py` carries the starter's recipe. Revert that file to master's before the PR (`git checkout master -- train.py`), so the merge changes only `library/` and `.gitignore`.

---

## Part B — the explorer (llm-explorables, branch `feat/training-explorer` from `main`)

### Task 3: `training/src/data.js`, the pure helpers

**Files:**
- Create: `training/src/data.js`
- Test: `test/training/Data.test.js`

**Interfaces:**
- Produces (globals): `VIEWS`, `DATASET_NAMES`, `TOKENIZER_NAMES`, `NOISE_BPB`, `parseResultsTsv(text)`, `staircase(rows)`, `recipeDiff(a, b)`, `scoresComparable(a, b)`, `timeLabel(t_s)`, `snapIndexAt(run, t_s)`, `defaultCompareIndex(a, b)`, `markSpans(text, spans)`, `validateIndex(index)`, `isRunFile(run)`, `tokenizerLabel(name)`, `runLabel(run)`, `runFacts(run)`, `growthCaption(run, i)`, `evolutionCaption(stairs)`, `pieceLabel(piece)`, `parseRepoUrl(url)`, `pickRepoFiles(paths)`, `viewFromUrl(search)`.

- [ ] **Step 1: Write the failing tests**

```js
/** Training Decisions: the pure helpers in training/src/data.js. */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { loadSketch, sketch } = require('../helpers/sketch');

const s = loadSketch(sketch('training', ['data.js']));
const plain = (v) => JSON.parse(JSON.stringify(v));
const f = (name) => s.get(name);

const run = (over = {}) => ({
  schema: 1, run_id: 'r', commit: 'abc1234', dataset: 'tinystories',
  tokenizer: { name: 'own', vocab_size: 8192 }, recipe: { DEPTH: 6, MATRIX_LR: 0.05, vocab_size: 8192 },
  prompts: ['Once upon a time'], snapshots: [0, 10, 30, 60, 120].map((t) => ({ t_s: t, step: t * 2, samples: ['x'] }))
    .concat([{ t_s: 300.2, step: 640, samples: ['y'], final: true }]),
  final: { val_bpb: 0.521935, params_m: 18.9 }, ...over,
});

test('parseResultsTsv: CRLF, blank and missing scores, column order from the header', () => {
  const rows = f('parseResultsTsv')('commit\ttimestamp\tval_bpb\tstatus\tdescription\r\n' +
    'aaa\tt1\t0.52\tkeep\tbaseline\r\n\r\nbbb\tt2\tmissing\tcrash\toom\r\nccc\tt3\t\tdiscard\tx\r\n');
  assert.deepEqual(plain(rows.map((r) => [r.n, r.commit, r.val_bpb, r.status])), [[1, 'aaa', 0.52, 'keep'], [2, 'bbb', null, 'crash'], [3, 'ccc', null, 'discard']]);
  assert.deepEqual(plain(f('parseResultsTsv')('')), []);
});

test('staircase: best so far steps down at kept improvements; crashes are left out; the gain', () => {
  const rows = [
    { n: 1, commit: 'a', val_bpb: 0.52, status: 'keep', description: 'baseline' },
    { n: 2, commit: 'b', val_bpb: 0.53, status: 'discard', description: 'deeper' },
    { n: 3, commit: 'c', val_bpb: null, status: 'crash', description: 'oom' },
    { n: 4, commit: 'd', val_bpb: 0.518, status: 'keep', description: 'lr' },
  ];
  const st = f('staircase')(rows);
  assert.deepEqual(plain(st.points.map((p) => [p.n, p.kept, p.improved, p.bestBefore, p.bestCommitBefore])),
    [[1, true, true, null, null], [2, false, false, 0.52, 'a'], [4, true, true, 0.52, 'a']]);
  assert.deepEqual(plain(st.steps), [{ n: 1, best: 0.52 }, { n: 2, best: 0.52 }, { n: 4, best: 0.518 }]);
  assert.equal(Number(st.gain.bpb.toFixed(4)), 0.002);
  assert.equal(f('staircase')([]).gain, null);
});

test('recipeDiff names the decisions that differ; vocabulary size follows the tokenizer', () => {
  const d = f('recipeDiff')(run(), run({ tokenizer: { name: 'phi3' }, recipe: { DEPTH: 6, MATRIX_LR: 0.045, vocab_size: 32015 } }));
  assert.deepEqual(plain(d), [{ key: 'tokenizer', a: 'built from the data', b: 'Phi-3 / Llama 2' }, { key: 'MATRIX_LR', a: 0.05, b: 0.045 }]);
  assert.deepEqual(plain(f('recipeDiff')(run(), run())), []);
  assert.equal(f('scoresComparable')(run(), run({ dataset: 'folktales' })), false);
});

test('times: labels, nearest snapshot, and where a comparison opens', () => {
  assert.deepEqual([0, 10, 30, 60, 120, 300.2].map(f('timeLabel')), ['0 s', '10 s', '30 s', '1 min', '2 min', '5 min']);
  const short = run({ snapshots: run().snapshots.slice(0, 3) });
  assert.equal(f('snapIndexAt')(short, 120), 2, 'a run that stopped early shows its last moment');
  assert.equal(f('defaultCompareIndex')(run(), run({ tokenizer: { name: 'gpt2' } })), 2, 'tokenizers differ: open at 30 s');
  assert.equal(f('defaultCompareIndex')(run(), run({ dataset: 'folktales' })), 5, 'otherwise at the end');
  assert.equal(f('defaultCompareIndex')(short, run({ tokenizer: { name: 'gpt2' } })), 2);
});

test('markSpans: copied stretches, clipped, sorted and overlapping spans merged', () => {
  assert.deepEqual(plain(f('markSpans')('abcdefgh', [[5, 7], [1, 3], [2, 4], [6, 99]])),
    [{ text: 'a', copied: false }, { text: 'bcd', copied: true }, { text: 'e', copied: false }, { text: 'fgh', copied: true }]);
  assert.deepEqual(plain(f('markSpans')('abc', undefined)), [{ text: 'abc', copied: false }]);
});

test('validateIndex and isRunFile', () => {
  const ok = { schema: 1, collections: [{ id: 'baselines', title: 'B', kind: 'baselines', runs: ['baselines/runs/a.json'], results: null }], tokens: 'tokens.json', copies: 'copies.json' };
  assert.deepEqual(plain(f('validateIndex')(ok)), []);
  assert.ok(f('validateIndex')({ schema: 2 }).length > 0);
  assert.ok(f('validateIndex')({ ...ok, collections: [{ ...ok.collections[0], kind: 'x' }] }).length > 0);
  assert.equal(f('isRunFile')(run()), true);
  assert.equal(f('isRunFile')({ schema: 1, run_id: 'r', dataset: 'x', prompts: [], snapshots: [{ samples: [] }] }), false);
  assert.equal(f('isRunFile')(null), false);
});

test('labels and captions', () => {
  assert.equal(f('runLabel')(run()), 'TinyStories · built from the data · 0.522');
  assert.match(f('runFacts')(run()), /8,192 tokens.*18\.9 M parameters.*0\.521935 \(lower is better\)/);
  assert.match(f('growthCaption')(run(), 0), /portrait of the vocabulary/);
  assert.match(f('growthCaption')(run(), 5), /5 min of training \(640 steps\)/);
  assert.equal(f('pieceLabel')(' upon'), '␣upon');
  const stairs = f('staircase')([{ n: 1, commit: 'a', val_bpb: 0.52, status: 'keep', description: '' }, { n: 2, commit: 'b', val_bpb: 0.5186, status: 'keep', description: '' }]);
  assert.match(f('evolutionCaption')(stairs), /0\.0014 \(0\.27%\), less than the noise/);
});

test('parseRepoUrl, pickRepoFiles, viewFromUrl', () => {
  assert.deepEqual(plain(f('parseRepoUrl')('https://github.com/you/autoresearch-starter')), { owner: 'you', repo: 'autoresearch-starter', ref: null });
  assert.deepEqual(plain(f('parseRepoUrl')('github.com/you/r.git/')), { owner: 'you', repo: 'r', ref: null });
  assert.deepEqual(plain(f('parseRepoUrl')('https://github.com/a/b/tree/session/folktales-2026-10-07')), { owner: 'a', repo: 'b', ref: 'session/folktales-2026-10-07' });
  assert.equal(f('parseRepoUrl')('https://gitlab.com/a/b'), null);
  assert.deepEqual(plain(f('pickRepoFiles')(['me/runs/a.json', 'me/results.tsv', 'me/library/baselines/runs/x.json', 'me/runs/sub/b.json', 'me/runs/notes.txt'])),
    { runs: ['me/runs/a.json'], results: 'me/results.tsv' });
  assert.deepEqual(plain(f('pickRepoFiles')(['runs/a.json'])), { runs: ['runs/a.json'], results: null });
  assert.equal(f('viewFromUrl')('?view=evolution&mode=presentation'), 'evolution');
  assert.equal(f('viewFromUrl')('?view=nope'), 'growth');
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `node --test test/training/Data.test.js`
Expected: FAIL (`training/src/data.js` missing).

- [ ] **Step 3: Write `training/src/data.js`**

```js
/**
 * Training Decisions explorer: the pure parts. No DOM, no fetch; views.js, ownRuns.js and
 * Main.js call these. Tested headless in test/training/Data.test.js.
 *
 * A run file is one five-minute training run written by autoresearch's capture.py (schema 1):
 * dataset, tokenizer, recipe, the four fixed prompts, snapshots of the writing at 0, 10, 30,
 * 60 and 120 s and at the end, and the final score. Runs are keyed by file path everywhere,
 * never by run_id: the library's six baselines share one run_id.
 */
const VIEWS = [
  { id: 'growth', title: 'Growth' },
  { id: 'compare', title: 'Compare' },
  { id: 'tokens', title: 'Tokens' },
  { id: 'evolution', title: 'Evolution' },
];
const DATASET_NAMES = { tinystories: 'TinyStories', folktales: 'Folktales' };
const TOKENIZER_NAMES = { own: 'built from the data', phi3: 'Phi-3 / Llama 2', gpt2: 'GPT-2' };
/** Two runs of the same code differ by about this much (measured on a laptop GPU). */
const NOISE_BPB = 0.003;

function parseResultsTsv(text) {
  const lines = String(text || '').replace(/\r/g, '').split('\n').filter((l) => l.trim());
  if (!lines.length) return [];
  const head = lines[0].split('\t').map((h) => h.trim());
  const col = (row, name) => (row[head.indexOf(name)] || '').trim();
  return lines.slice(1).map((line, i) => {
    const row = line.split('\t');
    const score = Number(col(row, 'val_bpb'));
    return {
      n: i + 1,
      timestamp: col(row, 'timestamp'),
      commit: col(row, 'commit'),
      val_bpb: col(row, 'val_bpb') !== '' && Number.isFinite(score) && score > 0 ? score : null,
      status: col(row, 'status'),
      description: col(row, 'description'),
    };
  });
}

/** Points (one per scored experiment), the best-so-far steps, and the session's whole gain. */
function staircase(rows) {
  let best = null;
  let bestCommit = null;
  const points = [];
  const steps = [];
  for (const r of rows || []) {
    if (r.val_bpb === null || r.val_bpb === undefined) continue;
    const kept = r.status === 'keep';
    const improved = kept && (best === null || r.val_bpb < best);
    points.push({ ...r, kept, improved, bestBefore: best, bestCommitBefore: bestCommit });
    if (improved) { best = r.val_bpb; bestCommit = r.commit; }
    if (best !== null) steps.push({ n: r.n, best });
  }
  const first = points.find((p) => p.kept);
  const gain = first && best !== null
    ? { from: first.val_bpb, to: best, bpb: first.val_bpb - best, share: (first.val_bpb - best) / first.val_bpb }
    : null;
  return { points, steps, gain };
}

function tokenizerLabel(name) { return TOKENIZER_NAMES[name] || name || 'unknown'; }
function datasetLabel(name) { return DATASET_NAMES[name] || name || 'unknown'; }

/** The decisions two runs differ on: dataset, tokenizer, then any recipe value. */
function recipeDiff(a, b) {
  const out = [];
  if (a.dataset !== b.dataset) out.push({ key: 'dataset', a: datasetLabel(a.dataset), b: datasetLabel(b.dataset) });
  const ta = (a.tokenizer || {}).name;
  const tb = (b.tokenizer || {}).name;
  if (ta !== tb) out.push({ key: 'tokenizer', a: tokenizerLabel(ta), b: tokenizerLabel(tb) });
  const ra = a.recipe || {};
  const rb = b.recipe || {};
  for (const k of [...new Set([...Object.keys(ra), ...Object.keys(rb)])]) {
    if (k === 'vocab_size') continue; // follows from the tokenizer
    if (JSON.stringify(ra[k]) !== JSON.stringify(rb[k])) out.push({ key: k, a: ra[k] === undefined ? '—' : ra[k], b: rb[k] === undefined ? '—' : rb[k] });
  }
  return out;
}

function scoresComparable(a, b) { return a.dataset === b.dataset; }

function timeLabel(t) {
  return t < 59.5 ? `${Math.round(t)} s` : `${Number((t / 60).toFixed(1))} min`;
}

/** The index of the run's snapshot nearest t_s. */
function snapIndexAt(run, t) {
  let best = 0;
  run.snapshots.forEach((s, i) => { if (Math.abs(s.t_s - t) < Math.abs(run.snapshots[best].t_s - t)) best = i; });
  return best;
}

/** Index into the shorter run's snapshots: 30 s when the tokenizers differ (where the gap reads best), else the end. */
function defaultCompareIndex(a, b) {
  const shorter = a.snapshots.length <= b.snapshots.length ? a : b;
  const differ = (a.tokenizer || {}).name !== (b.tokenizer || {}).name;
  return differ ? snapIndexAt(shorter, 30) : shorter.snapshots.length - 1;
}

/** The text split into plain and copied stretches. */
function markSpans(text, spans) {
  const s = String(text || '');
  const clean = (spans || [])
    .map(([x, y]) => [Math.max(0, x), Math.min(s.length, y)])
    .filter(([x, y]) => y > x)
    .sort((p, q) => p[0] - q[0]);
  const out = [];
  let at = 0;
  for (const [x, y] of clean) {
    const start = Math.max(x, at);
    if (y <= start) continue;
    if (start > at) out.push({ text: s.slice(at, start), copied: false });
    if (out.length && out[out.length - 1].copied && start === at) out[out.length - 1].text += s.slice(start, y);
    else out.push({ text: s.slice(start, y), copied: true });
    at = y;
  }
  if (at < s.length) out.push({ text: s.slice(at), copied: false });
  return out;
}

function validateIndex(index) {
  const problems = [];
  if (!index || typeof index !== 'object') return ['index.json is not an object'];
  if (index.schema !== 1) problems.push(`unknown schema ${index.schema}`);
  if (!Array.isArray(index.collections)) return problems.concat('collections is not a list');
  for (const c of index.collections) {
    if (typeof c.id !== 'string' || typeof c.title !== 'string') problems.push('a collection has no id or title');
    if (!['baselines', 'session'].includes(c.kind)) problems.push(`${c.id}: unknown kind ${c.kind}`);
    if (!Array.isArray(c.runs) || c.runs.some((r) => typeof r !== 'string' || !r.endsWith('.json'))) problems.push(`${c.id}: runs must be .json paths`);
    if (c.results !== null && !(typeof c.results === 'string' && c.results.endsWith('results.tsv'))) problems.push(`${c.id}: results must be a results.tsv path or null`);
  }
  return problems;
}

function isRunFile(r) {
  return Boolean(r && r.schema === 1 && typeof r.run_id === 'string' && typeof r.dataset === 'string'
    && Array.isArray(r.prompts) && Array.isArray(r.snapshots) && r.snapshots.length > 0
    && r.snapshots.every((s) => s && typeof s.t_s === 'number' && Array.isArray(s.samples)));
}

function scoreOf(run) { return ((run && run.final) || {}).val_bpb; }

function runLabel(run) {
  const score = scoreOf(run);
  return [datasetLabel(run.dataset), tokenizerLabel((run.tokenizer || {}).name), score ? score.toFixed(3) : null].filter(Boolean).join(' · ');
}

function runFacts(run) {
  const t = run.tokenizer || {};
  const parts = [datasetLabel(run.dataset), `${tokenizerLabel(t.name)} tokenizer${t.vocab_size ? ` (${t.vocab_size.toLocaleString('en-GB')} tokens)` : ''}`];
  const f = run.final || {};
  if (f.params_m) parts.push(`${f.params_m} M parameters`);
  if (f.val_bpb) parts.push(`score ${f.val_bpb.toFixed(6)} (lower is better)`);
  return parts.join(' · ');
}

function growthCaption(run, i) {
  const s = run.snapshots[i];
  if (!s) return '';
  if (s.t_s === 0) return 'Before any training. Random weights pick tokens almost at random, so the words you see are a portrait of the vocabulary itself.';
  const steps = s.step ? ` (${s.step} steps)` : '';
  return s.final ? `After the full ${timeLabel(s.t_s)} of training${steps}.` : `After ${timeLabel(s.t_s)} of training${steps}.`;
}

function evolutionCaption(stairs) {
  const base = 'Each dot is one five-minute experiment by the agent: filled dots it kept, hollow ones it threw away. The line is the best score so far, and the shaded band above it is run-to-run noise (0.003): a step down inside the band is not evidence.';
  const g = stairs.gain;
  if (!g) return base;
  const small = g.bpb < NOISE_BPB ? ', less than the noise' : '';
  return `${base} In all, this session improved the score by ${g.bpb.toFixed(4)} (${(g.share * 100).toFixed(2)}%)${small}. Gains this small cannot be read in the writing, which is why the agent needs a score.`;
}

function pieceLabel(piece) { return String(piece).replace(/ /g, '␣'); }

function parseRepoUrl(input) {
  const m = String(input || '').trim()
    .match(/^(?:https?:\/\/)?(?:www\.)?github\.com\/([\w.-]+)\/([\w.-]+?)(?:\.git)?(?:\/tree\/([^?#]+?))?\/?(?:[?#].*)?$/);
  return m ? { owner: m[1], repo: m[2], ref: m[3] || null } : null;
}

/** From a dropped folder's file paths: the top folder's runs/*.json and results.tsv only. */
function pickRepoFiles(paths) {
  const runs = paths.filter((p) => /^(?:[^/]+\/)?runs\/[^/]+\.json$/.test(p)).sort();
  const results = paths.filter((p) => /^(?:[^/]+\/)?results\.tsv$/.test(p)).sort((a, b) => a.length - b.length)[0] || null;
  return { runs, results };
}

function viewFromUrl(search) {
  const v = new URLSearchParams(search || '').get('view');
  return VIEWS.some((x) => x.id === v) ? v : 'growth';
}
```

- [ ] **Step 4: Run the tests**

Run: `node --test test/training/Data.test.js`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add training/src/data.js test/training/Data.test.js
git commit -m "training: pure helpers for the Training Decisions explorer"
```

### Task 4: Sync the library into `training/library/`

**Files:**
- Create: `scripts/sync-training-library.mjs`, `training/library/**` (generated), `test/training/Library.test.js`

**Interfaces:**
- Consumes: the worked example's merged `library/` (Task 2); `validateIndex`, `isRunFile` (Task 3).
- Produces: `training/library/index.json`, `tokens.json`, `copies.json`, the listed run files and scoreboards, `SOURCE.json` (`{repo, path, commit, synced, files}`).

- [ ] **Step 1: Write the failing test**

```js
/** The committed library copy is whole: every listed file exists and parses; the source is pinned. */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { loadSketch, sketch } = require('../helpers/sketch');

const LIB = path.join(__dirname, '..', '..', 'training', 'library');
const read = (rel) => JSON.parse(fs.readFileSync(path.join(LIB, rel), 'utf8'));
const s = loadSketch(sketch('training', ['data.js']));

test('index.json is valid and every file it lists is here', () => {
  const index = read('index.json');
  assert.deepEqual(JSON.parse(JSON.stringify(s.get('validateIndex')(index))), []);
  for (const c of index.collections) {
    for (const rel of c.runs) assert.ok(s.get('isRunFile')(read(rel)), rel);
    if (c.results) assert.ok(fs.existsSync(path.join(LIB, c.results)), c.results);
  }
  assert.ok(index.collections.some((c) => c.kind === 'baselines' && c.runs.length === 6), 'the six baselines');
});

test('tokens.json covers both datasets and all three tokenizers', () => {
  const t = read('tokens.json');
  assert.equal(t.prompts.length, 4);
  for (const d of ['tinystories', 'folktales']) {
    for (const k of ['own', 'phi3', 'gpt2']) {
      const e = t.datasets[d][k];
      assert.equal(e.splits.length, 4, `${d}/${k}`);
      assert.equal(e.splits[0].join(''), t.prompts[0], `${d}/${k} pieces rebuild the prompt`);
    }
  }
});

test('the copy is pinned to a commit of the worked example', () => {
  const src = read('SOURCE.json');
  assert.equal(src.repo, 'aroughidea/autoresearch-win-rtx');
  assert.match(src.commit, /^[0-9a-f]{40}$/);
});
```

- [ ] **Step 2: Run it to see it fail**

Run: `node --test test/training/Library.test.js`
Expected: FAIL (no `training/library/index.json`).

- [ ] **Step 3: Write `scripts/sync-training-library.mjs`**

```js
#!/usr/bin/env node
/**
 * Copy the Training Decisions library into training/library/ from the worked example
 * (aroughidea/autoresearch-win-rtx, folder library/). Static hosting cannot list a folder,
 * so library/index.json names every file and this script fetches exactly those.
 *
 *   node scripts/sync-training-library.mjs --ref <commit>          from GitHub at a commit
 *   node scripts/sync-training-library.mjs --from <local checkout>  from a local clone (pins its HEAD)
 */
import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const REPO = 'aroughidea/autoresearch-win-rtx';
const OUT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'training', 'library');
const args = process.argv.slice(2);
const opt = (k) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : null; };
const from = opt('--from');
let ref = opt('--ref');
if (!ref && !from) {
  console.error('usage: node scripts/sync-training-library.mjs --ref <commit> | --from <path>');
  process.exit(2);
}
if (from) ref = execSync('git rev-parse HEAD', { cwd: from }).toString().trim();
if (!/^[0-9a-f]{40}$/.test(ref)) {
  console.error('pin a full 40-character commit, not a branch name');
  process.exit(2);
}

async function read(rel) {
  if (from) return fs.readFileSync(path.join(from, 'library', rel));
  const url = `https://raw.githubusercontent.com/${REPO}/${ref}/library/${rel}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

const index = JSON.parse((await read('index.json')).toString('utf8'));
const files = ['index.json', index.tokens, index.copies,
  ...index.collections.flatMap((c) => [...c.runs, ...(c.results ? [c.results] : [])])];
fs.rmSync(OUT, { recursive: true, force: true });
for (const rel of files) {
  const dest = path.join(OUT, rel);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, await read(rel));
}
const source = { repo: REPO, path: 'library/', commit: ref, synced: new Date().toISOString().slice(0, 10), files: files.length };
fs.writeFileSync(path.join(OUT, 'SOURCE.json'), JSON.stringify(source, null, 1) + '\n');
console.log(`synced ${files.length} files from ${REPO}@${ref.slice(0, 7)}`);
```

- [ ] **Step 4: Sync and test**

Run: `node scripts/sync-training-library.mjs --ref <Task 2 merge commit, 40 chars>` then `node --test test/training/Library.test.js`
Expected: `synced N files`; 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/sync-training-library.mjs training/library test/training/Library.test.js
git commit -m "training: sync the library from the worked example at a pinned commit"
```

### Task 5: The page, its header, and the site index

**Files:**
- Create: `training/index.html`, `training/style.css`, `training/src/Main.js`, `training/src/initUI.js`, `training/src/views.js` (with `el` and stub renderers this task, filled in Tasks 6–7), `training/src/ownRuns.js` (stub `loadRepo`/`loadDropped` returning an error until Task 8), `test/training/Page.test.js`
- Modify: `index.html` (sidebar item and table row 7), `test/site/Index.test.js` (ORDER, numbering, titles)

**Interfaces:**
- Consumes: Task 3's helpers; `Modes` (`shared/Modes.js`: `Modes.init()`, `Modes.togglePresentation()`); `d3`.
- Produces: globals `LIB` (`{collections: [{id, title, kind, note, runIds, rows}], runs: Map<path, run>, tokens, copies}`), `STATE`, `showView(id)`, `showStatus(text)`, `labelFor(runKey)`, `findRun(pred)`, `addCollection(c)`; DOM ids `#training_header`, `#view_tabs`, `#tab_<view>`, `#your_runs`, `#repo_url`, `#repo_folder`, `#load_status`, `#view`, `#home_link`.

- [ ] **Step 1: Write the failing tests**

`test/training/Page.test.js`:
```js
/** The Training Decisions page: base URL, stylesheet, script order. */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '..', '..', 'training', 'index.html'), 'utf8');

test('pins its base URL, title and stylesheet', () => {
  assert.match(html, /<base href="\/training\/">/);
  assert.ok(html.indexOf('<base ') < html.indexOf('<link'));
  assert.match(html, /<title>Training Decisions<\/title>/);
  assert.match(html, /<link rel="stylesheet" href="style\.css">/);
});

test('loads p5, shared files, D3, then its own tabs in order', () => {
  const srcs = [...html.matchAll(/<script src="([^"]+)"><\/script>/g)].map((m) => m[1]);
  assert.deepEqual(srcs, [
    'https://cdn.jsdelivr.net/npm/p5@1.9.4/lib/p5.js',
    '/shared/Modes.js',
    '/shared/d3.v7.min.js',
    'src/data.js',
    'src/views.js',
    'src/ownRuns.js',
    'src/Main.js',
    'src/initUI.js',
  ]);
});
```
In `test/site/Index.test.js`: `ORDER` gains `'/training/'` at the end; the numbering arrays become `['1'..'7']`; the sidebar test title says "seven"; the titles list gains `['training', 'Training Decisions']`; `admin after the explorables` compares against `/training/`.

- [ ] **Step 2: Run them to see them fail**

Run: `node --test test/training/Page.test.js test/site/Index.test.js`
Expected: FAIL (no `training/index.html`; index has six rows).

- [ ] **Step 3: Write the page**

`training/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <base href="/training/">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Training Decisions</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <link rel="stylesheet" href="style.css">
  <!--
    What a decision does to a model: small models trained for five minutes each by autoresearch
    (aroughidea/autoresearch-win-rtx), read from their run files. Growth, Compare, Tokens and
    Evolution views over training/library/ (synced at a pinned commit, see library/SOURCE.json),
    plus a learner's own runs from a public GitHub repo or a dropped folder. Classic scripts in
    one global scope: shared files from /shared/, this page's own from src/.
  -->
  <script src="https://cdn.jsdelivr.net/npm/p5@1.9.4/lib/p5.js"></script>
</head>
<body>
  <script src="/shared/Modes.js"></script>
  <script src="/shared/d3.v7.min.js"></script>
  <script src="src/data.js"></script>
  <script src="src/views.js"></script>
  <script src="src/ownRuns.js"></script>
  <script src="src/Main.js"></script>
  <script src="src/initUI.js"></script>
</body>
</html>
```

`training/src/Main.js`:
```js
/**
 * Training Decisions explorer: boot, the loaded library, view switching, keys.
 * Views draw in views.js; pure helpers live in data.js; a learner's runs load in ownRuns.js.
 */
const LIB = { collections: [], runs: new Map(), tokens: null, copies: {} };
const STATE = {
  view: 'growth',
  growth: { run: null, t: null },
  compare: { a: null, b: null, t: null },
  tokens: { dataset: 'tinystories', prompt: 0 },
  evolution: { collection: null, point: null },
};

function setup() {
  noCanvas();
  Modes.init();
  initUI();
  loadLibrary();
}

function keyPressed(event) {
  const e = event || {};
  if (e.ctrlKey && e.altKey && keyCode === 80) { // P
    if (e.preventDefault) e.preventDefault();
    Modes.togglePresentation();
    return false;
  }
  const typing = document.activeElement && ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName);
  if (!typing && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) stepTime(e.key === 'ArrowRight' ? 1 : -1);
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} answered ${res.status}`);
  return res.json();
}

async function loadLibrary() {
  showStatus('Loading the library…');
  try {
    const index = await fetchJson('library/index.json');
    const problems = validateIndex(index);
    if (problems.length) throw new Error(problems[0]);
    const [tokens, copies] = await Promise.all([
      fetchJson('library/' + index.tokens).catch(() => null),
      fetchJson('library/' + index.copies).catch(() => ({})),
    ]);
    LIB.tokens = tokens;
    LIB.copies = copies;
    for (const c of index.collections) {
      const loaded = await Promise.all(c.runs.map((rel) => fetchJson('library/' + rel).then((r) => [rel, r]).catch(() => [rel, null])));
      const runIds = [];
      for (const [rel, run] of loaded) if (isRunFile(run)) { LIB.runs.set(rel, run); runIds.push(rel); }
      const rows = c.results ? parseResultsTsv(await (await fetch('library/' + c.results)).text()) : null;
      LIB.collections.push({ id: c.id, title: c.title, kind: c.kind, note: c.note || '', runIds, rows });
    }
    showStatus('');
  } catch (err) {
    showStatus('Could not load the library: ' + err.message);
  }
  showView(viewFromUrl(location.search));
}

/** A learner's collection: runs keyed by their own paths, appended after the library. */
function addCollection(c) {
  for (const [key, run] of Object.entries(c.runs)) LIB.runs.set(key, run);
  LIB.collections = LIB.collections.filter((x) => x.id !== c.id);
  LIB.collections.push({ id: c.id, title: c.title, kind: 'yours', note: '', runIds: Object.keys(c.runs), rows: c.rows });
  showView(STATE.view);
}

function findRun(pred) {
  for (const c of LIB.collections) for (const key of c.runIds) if (!pred || pred(LIB.runs.get(key))) return key;
  return null;
}

function findBaseline(dataset, tokenizer) {
  return findRun((r) => r.dataset === dataset && (r.tokenizer || {}).name === tokenizer);
}

/** A run's name: the agent's description when its collection has a scoreboard row for it. */
function labelFor(key) {
  const run = LIB.runs.get(key);
  const c = LIB.collections.find((x) => x.runIds.includes(key));
  const row = c && c.rows && c.rows.find((r) => r.commit === run.commit);
  return row && row.description ? `${row.description} · ${row.val_bpb ? row.val_bpb.toFixed(3) : row.status}` : runLabel(run);
}

function showStatus(text) {
  const s = document.getElementById('load_status');
  if (s) s.textContent = text;
}

function showView(id) {
  STATE.view = VIEWS.some((v) => v.id === id) ? id : 'growth';
  for (const v of VIEWS) {
    const tab = document.getElementById('tab_' + v.id);
    if (tab) tab.setAttribute('aria-selected', String(v.id === STATE.view));
  }
  const root = document.getElementById('view');
  root.replaceChildren();
  ({ growth: renderGrowth, compare: renderCompare, tokens: renderTokens, evolution: renderEvolution })[STATE.view](root);
  const params = new URLSearchParams(location.search);
  params.set('view', STATE.view);
  history.replaceState(null, '', '?' + params.toString());
}

function stepTime(delta) {
  const slot = STATE.view === 'growth' ? STATE.growth : STATE.view === 'compare' ? STATE.compare : null;
  if (!slot || slot.t === null) return;
  const key = STATE.view === 'growth' ? slot.run : slot.a;
  const run = key && LIB.runs.get(key);
  if (!run) return;
  slot.t = Math.max(0, Math.min(run.snapshots.length - 1, slot.t + delta));
  showView(STATE.view);
}
```

`training/src/initUI.js`:
```js
/** The header: title, intro, view tabs, and the "your runs" loader. Layout lives in style.css. */
function initUI() {
  const form = el('form', { id: 'your_runs' },
    el('label', { for: 'repo_url', text: 'Your runs:' }),
    el('input', { id: 'repo_url', type: 'url', placeholder: 'https://github.com/you/your-autoresearch' }),
    el('button', { type: 'submit', text: 'Load' }),
    el('label', { class: 'folder' }, 'or choose its folder ',
      el('input', { id: 'repo_folder', type: 'file', webkitdirectory: true, multiple: true })));
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    showStatus('Loading from GitHub…');
    const result = await loadRepo(document.getElementById('repo_url').value);
    showLoadResult(result);
  });
  form.querySelector('#repo_folder').addEventListener('change', async (ev) => {
    const files = [...ev.target.files].map((f) => ({ path: f.webkitRelativePath || f.name, text: () => f.text() }));
    showLoadResult(await loadDropped(files, 'your folder'));
  });
  document.body.append(
    el('a', { id: 'home_link', href: '/', title: 'All explorables', text: '⌂' }),
    el('header', { id: 'training_header' },
      el('h1', { text: 'Training Decisions' }),
      el('p', { id: 'training_intro', text: 'Small language models, each trained for five minutes on a laptop GPU. Change one decision (the dataset, the tokenizer, or the recipe an AI agent tunes) and read what it does to the writing.' }),
      el('nav', { id: 'view_tabs', 'aria-label': 'Views' },
        VIEWS.map((v) => el('button', { id: 'tab_' + v.id, class: 'tab', type: 'button', 'aria-selected': 'false', onclick: () => showView(v.id), text: v.title }))),
      form,
      el('p', { id: 'load_status', role: 'status' })),
    el('main', { id: 'view' }));
  document.body.addEventListener('dragover', (ev) => ev.preventDefault());
  document.body.addEventListener('drop', async (ev) => {
    ev.preventDefault();
    const entries = [...ev.dataTransfer.items].map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
    showLoadResult(await loadDropped(await filesFromEntries(entries), 'the dropped folder'));
  });
}

function showLoadResult(result) {
  if (result.error) { showStatus(result.error); return; }
  addCollection(result.collection);
  const n = Object.keys(result.collection.runs).length;
  showStatus(`Loaded ${n} run${n === 1 ? '' : 's'}${result.collection.rows ? ' and a scoreboard' : ''} from ${result.collection.title.replace(/^Your runs: /, '')}.`);
}
```

`training/src/views.js` (this task: `el` and placeholder renderers that Tasks 6–7 replace):
```js
/**
 * The four views, drawn into #view as plain DOM from LIB and STATE (Main.js) with data.js's
 * helpers. Each re-renders whole on any change; layout lives in style.css.
 */
function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function renderGrowth(root) { root.append(el('p', { class: 'empty', text: 'Growth' })); }
function renderCompare(root) { root.append(el('p', { class: 'empty', text: 'Compare' })); }
function renderTokens(root) { root.append(el('p', { class: 'empty', text: 'Tokens' })); }
function renderEvolution(root) { root.append(el('p', { class: 'empty', text: 'Evolution' })); }
```

`training/src/ownRuns.js` (stub until Task 8):
```js
/** Loading a learner's own runs (Task 8). */
async function loadRepo() { return { error: 'Loading your runs is not built yet.' }; }
async function loadDropped() { return { error: 'Loading your runs is not built yet.' }; }
async function filesFromEntries() { return []; }
```

`training/style.css`:
```css
/* Layout for the Training Decisions page, keyed by the mode class shared/Modes.js puts on
   <body> (mode-presentation | mode-desktop | mode-mobile). Desktop is the base. */
body { margin: 0; font-family: sans-serif; color: #1f2328; background: #fff; line-height: 1.45; }
#home_link { position: absolute; top: 12px; left: 16px; text-decoration: none; color: #57606a; font-size: 22px; }
#training_header { padding: 16px 48px 8px; border-bottom: 1px solid #d0d7de; }
#training_header h1 { margin: 0 0 4px; font-size: 26px; }
#training_intro { margin: 0 0 12px; max-width: 70ch; color: #424a53; }
#view_tabs { display: flex; gap: 4px; margin-bottom: 10px; }
.tab { font: inherit; padding: 6px 14px; border: 1px solid #d0d7de; border-radius: 6px; background: #f6f8fa; cursor: pointer; }
.tab[aria-selected="true"] { background: #1f2328; color: #fff; border-color: #1f2328; }
#your_runs { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 14px; }
#repo_url { width: 26em; max-width: 100%; font: inherit; padding: 4px 6px; }
#load_status { min-height: 1.2em; margin: 6px 0 0; font-size: 14px; color: #8250df; }
#view { padding: 16px 48px 48px; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 8px; }
.controls select { font: inherit; padding: 4px; max-width: 100%; }
.facts { color: #57606a; font-size: 14px; margin: 4px 0 10px; }
.caption { max-width: 75ch; color: #424a53; }
.times { display: flex; gap: 4px; margin: 8px 0; }
.time { font: inherit; padding: 4px 10px; border: 1px solid #d0d7de; border-radius: 14px; background: #fff; cursor: pointer; }
.time[aria-pressed="true"] { background: #0969da; color: #fff; border-color: #0969da; }
.samples { display: grid; gap: 10px; }
.sample { border-left: 3px solid #d0d7de; padding: 2px 0 2px 12px; white-space: pre-wrap; max-width: 75ch; }
.sample strong { margin-right: 0.2em; }
mark { background: #fff1a8; }
.legend { font-size: 13px; color: #57606a; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.pair h3 { font-size: 15px; margin: 6px 0; }
.diff { margin: 6px 0; }
.chip { display: inline-block; margin: 2px 6px 2px 0; padding: 2px 8px; border-radius: 10px; background: #f6f8fa; border: 1px solid #d0d7de; font-size: 13px; }
.note { background: #fff8c5; border: 1px solid #d4a72c; padding: 6px 10px; border-radius: 6px; max-width: 75ch; }
.tok-row { margin: 12px 0; }
.pieces { display: flex; flex-wrap: wrap; gap: 3px; }
.piece { font-family: monospace; padding: 2px 4px; border-radius: 4px; background: #ddf4ff; border: 1px solid #54aeff; }
.first p { font-family: monospace; font-size: 13px; color: #424a53; max-width: 90ch; }
#evolution_chart svg { width: 100%; height: auto; }
#evolution_chart .best { fill: none; stroke: #1a7f37; stroke-width: 2; }
#evolution_chart .noise { fill: #1a7f37; opacity: 0.12; }
#evolution_chart .dot { stroke: #1a7f37; stroke-width: 2; cursor: pointer; }
#evolution_chart .dot.kept { fill: #1a7f37; }
#evolution_chart .dot.discarded { fill: #fff; stroke: #8c959f; }
#evolution_chart .dot.selected { stroke: #cf222e; stroke-width: 3; }
.empty { color: #57606a; }

body.mode-presentation { font-size: 22px; }
body.mode-presentation #your_runs, body.mode-presentation #training_intro { display: none; }
body.mode-presentation .sample, body.mode-presentation .caption { max-width: none; }
body.mode-mobile #training_header, body.mode-mobile #view { padding-left: 16px; padding-right: 16px; }
body.mode-mobile .pair { grid-template-columns: 1fr; }
```

Site `index.html`: after the embeddings sidebar item add `<li><a href="/training/">7. Training Decisions</a></li>`; after the embeddings table row add
```html
            <tr>
              <td class="num">7</td>
              <td><a href="/training/">Training Decisions</a></td>
              <td>What a decision does to a model. The same small model trained on different datasets and tokenizers, scrubbed from noise to stories and read side by side, and an AI agent's experiments step by step. Load your own overnight run beside them. <span class="also"><a href="/training/?mode=presentation">Presentation mode</a></span></td>
            </tr>
```

- [ ] **Step 4: Run the tests, then look**

Run: `npm test` then `git diff --check`
Expected: all green, no whitespace errors. Then `npx serve .` and open `http://localhost:3000/training/`: the header, four tabs, "Growth" placeholder, no console errors.

- [ ] **Step 5: Commit**

```bash
git add training test/training/Page.test.js index.html test/site/Index.test.js
git commit -m "training: the page, its header and tabs, listed seventh on the index"
```

### Task 6: Growth and Compare

**Files:**
- Modify: `training/src/views.js` (replace `renderGrowth`, `renderCompare`; add `runPicker`, `timeButtons`, `samplesBlock`, `scoreLine`)

**Interfaces:**
- Consumes: `LIB`, `STATE`, `showView`, `labelFor`, `findRun`, `findBaseline` (Task 5); `snapIndexAt`, `defaultCompareIndex`, `recipeDiff`, `scoresComparable`, `markSpans`, `runFacts`, `growthCaption`, `timeLabel` (Task 3).

- [ ] **Step 1: Write Growth and Compare**

```js
function runPicker(id, selected, label) {
  const sel = el('select', { id, 'aria-label': label });
  for (const c of LIB.collections) {
    if (!c.runIds.length) continue;
    sel.append(el('optgroup', { label: c.title }, c.runIds.map((key) => el('option', { value: key, selected: key === selected, text: labelFor(key) }))));
  }
  return sel;
}

function timeButtons(run, active, onPick) {
  return el('div', { class: 'times', role: 'group', 'aria-label': 'Moment in training' },
    run.snapshots.map((s, i) => el('button', { class: 'time', type: 'button', 'aria-pressed': String(i === active), onclick: () => onPick(i), text: timeLabel(s.t_s) })));
}

function samplesBlock(key, run, i) {
  const snap = run.snapshots[i] || { samples: [] };
  const marks = (LIB.copies[key] || [])[i] || [];
  const anyCopied = marks.some((m) => m && m.length);
  return el('div', { class: 'samples' },
    run.prompts.map((p, j) => el('div', { class: 'sample' }, el('strong', { text: p }),
      markSpans(snap.samples[j] || '', marks[j]).map((seg) => (seg.copied ? el('mark', { text: seg.text }) : seg.text)))),
    anyCopied ? el('p', { class: 'legend', text: 'Highlighted: eight or more words copied word for word from the training text.' }) : null);
}

function noRuns(root) {
  root.append(el('p', { class: 'empty', text: 'No run files are loaded. A scoreboard alone (results.tsv) shows in Evolution.' }));
}

function renderGrowth(root) {
  const g = STATE.growth;
  if (!g.run || !LIB.runs.has(g.run)) { g.run = findBaseline('tinystories', 'own') || findRun(); g.t = null; }
  if (!g.run) return noRuns(root);
  const run = LIB.runs.get(g.run);
  if (g.t === null || g.t >= run.snapshots.length) g.t = 0;
  const picker = runPicker('growth_run', g.run, 'Run');
  picker.addEventListener('change', () => { g.run = picker.value; g.t = 0; showView('growth'); });
  root.append(
    el('div', { class: 'controls' }, picker),
    el('p', { class: 'facts', text: runFacts(run) }),
    timeButtons(run, g.t, (i) => { g.t = i; showView('growth'); }),
    el('p', { class: 'caption', text: growthCaption(run, g.t) }),
    samplesBlock(g.run, run, g.t));
}

function scoreLine(a, b) {
  const sa = (a.final || {}).val_bpb;
  const sb = (b.final || {}).val_bpb;
  if (!sa || !sb) return null;
  if (!scoresComparable(a, b)) {
    return el('p', { class: 'note', text: 'These two learned from different datasets, so their scores don’t compare: each score measures how well a model predicts its own dataset. Compare them by what they write.' });
  }
  return el('p', { class: 'facts', text: `Scores: ${sa.toFixed(4)} and ${sb.toFixed(4)} (lower is better; two runs of the same code differ by about ${NOISE_BPB}).` });
}

function renderCompare(root) {
  const c = STATE.compare;
  if (!c.a || !LIB.runs.has(c.a)) { c.a = findBaseline('tinystories', 'own') || findRun(); c.t = null; }
  if (!c.b || !LIB.runs.has(c.b) || c.b === c.a) { c.b = findBaseline('tinystories', 'phi3') || findRun((r) => r !== LIB.runs.get(c.a)); c.t = null; }
  if (!c.a || !c.b) return root.append(el('p', { class: 'empty', text: 'Load at least two runs to compare.' }));
  const a = LIB.runs.get(c.a);
  const b = LIB.runs.get(c.b);
  const shorter = a.snapshots.length <= b.snapshots.length ? a : b;
  if (c.t === null || c.t >= shorter.snapshots.length) c.t = defaultCompareIndex(a, b);
  const t = shorter.snapshots[c.t].t_s;
  const pa = runPicker('cmp_a', c.a, 'First run');
  const pb = runPicker('cmp_b', c.b, 'Second run');
  pa.addEventListener('change', () => { c.a = pa.value; c.t = null; showView('compare'); });
  pb.addEventListener('change', () => { c.b = pb.value; c.t = null; showView('compare'); });
  const diff = recipeDiff(a, b);
  root.append(
    el('div', { class: 'controls' }, pa, el('span', { text: 'and' }), pb),
    el('div', { class: 'diff' }, diff.length
      ? [el('strong', { text: 'What differs: ' }), diff.map((d) => el('span', { class: 'chip', text: `${d.key}: ${d.a} → ${d.b}` }))]
      : el('span', { text: 'The same decisions: any difference you read is run-to-run noise.' })),
    scoreLine(a, b),
    timeButtons(shorter, c.t, (i) => { c.t = i; showView('compare'); }),
    el('div', { class: 'pair' },
      el('div', {}, el('h3', { text: labelFor(c.a) }), samplesBlock(c.a, a, snapIndexAt(a, t))),
      el('div', {}, el('h3', { text: labelFor(c.b) }), samplesBlock(c.b, b, snapIndexAt(b, t)))));
}
```

- [ ] **Step 2: Test and look**

Run: `npm test`
Expected: green.
Browser, `/training/?view=growth`: TinyStories built-from-data run, 0 s caption about the vocabulary, stories by 5 min; switch to a Folktales run: copied phrases highlighted with the legend. `/training/?view=compare`: own vs Phi-3, opens at 30 s, diff chip "tokenizer: built from the data → Phi-3 / Llama 2", both scores; switch one to Folktales: the note replaces the scores. Arrow keys step the time.

- [ ] **Step 3: Commit**

```bash
git add training/src/views.js
git commit -m "training: Growth and Compare views"
```

### Task 7: Tokens and Evolution

**Files:**
- Modify: `training/src/views.js` (replace `renderTokens`, `renderEvolution`; add `drawStaircase`, `evolutionDetail`)

**Interfaces:**
- Consumes: `LIB.tokens` (Task 2's `tokens.json` shape), collections' `rows`; `staircase`, `evolutionCaption`, `pieceLabel`, `NOISE_BPB` (Task 3); `d3`.

- [ ] **Step 1: Write Tokens and Evolution**

```js
function renderTokens(root) {
  const s = STATE.tokens;
  if (!LIB.tokens) return root.append(el('p', { class: 'empty', text: 'The token data (library/tokens.json) did not load.' }));
  const datasets = Object.keys(LIB.tokens.datasets);
  if (!datasets.includes(s.dataset)) s.dataset = datasets[0];
  const dsSel = el('select', { id: 'tok_dataset', 'aria-label': 'Dataset' },
    datasets.map((d) => el('option', { value: d, selected: d === s.dataset, text: DATASET_NAMES[d] || d })));
  const prSel = el('select', { id: 'tok_prompt', 'aria-label': 'Sentence' },
    LIB.tokens.prompts.map((p, i) => el('option', { value: String(i), selected: i === s.prompt, text: p })));
  dsSel.addEventListener('change', () => { s.dataset = dsSel.value; showView('tokens'); });
  prSel.addEventListener('change', () => { s.prompt = Number(prSel.value); showView('tokens'); });
  const entries = Object.entries(LIB.tokens.datasets[s.dataset]);
  root.append(
    el('div', { class: 'controls' }, dsSel, prSel),
    el('p', { class: 'caption', text: 'The same sentence, cut into tokens by each tokenizer. A space travels with the word after it (␣).' }),
    entries.map(([name, t]) => el('div', { class: 'tok-row' },
      el('h3', { text: `${tokenizerLabel(name)} · ${t.vocab_size.toLocaleString('en-GB')} tokens` }),
      el('div', { class: 'pieces' }, t.splits[s.prompt].map((p) => el('span', { class: 'piece', text: pieceLabel(p) }))),
      el('p', { class: 'facts', text: `${t.splits[s.prompt].length} tokens for this sentence · ${t.chars_per_token} characters per token across ${DATASET_NAMES[s.dataset]} · ${Math.round(t.unused_share * 100)}% of the vocabulary never appears in it` }))),
    el('h2', { text: 'First words' }),
    el('p', { class: 'caption', text: 'Each model before any training. It picks tokens almost at random, so what it writes is a portrait of its vocabulary.' }),
    entries.map(([name]) => {
      const key = findBaseline(s.dataset, name);
      const snap = key && LIB.runs.get(key).snapshots[0];
      return snap ? el('div', { class: 'first' }, el('h3', { text: tokenizerLabel(name) }), el('p', { text: (snap.samples[s.prompt] || '').slice(0, 280) })) : null;
    }));
}

function drawStaircase(node, stairs, selected, onPick) {
  const pts = stairs.points;
  if (!pts.length) return;
  const W = Math.max(480, node.clientWidth || 900);
  const H = 340;
  const m = { t: 16, r: 20, b: 44, l: 72 };
  const scores = pts.map((p) => p.val_bpb).sort(d3.ascending);
  const lo = scores[0];
  const hi = Math.min(scores[scores.length - 1], lo + Math.max(0.02, (d3.quantile(scores, 0.9) - lo) * 1.3));
  const x = d3.scaleLinear().domain([0.5, d3.max(pts, (p) => p.n) + 0.5]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([lo - 0.002, hi + 0.002]).range([H - m.b, m.t]).clamp(true);
  const svg = d3.select(node).append('svg').attr('viewBox', `0 0 ${W} ${H}`)
    .attr('role', 'img').attr('aria-label', 'Score of each experiment, and the best so far');
  svg.append('path').datum(stairs.steps).attr('class', 'noise')
    .attr('d', d3.area().curve(d3.curveStepAfter).x((s) => x(s.n)).y0((s) => y(s.best)).y1((s) => y(s.best + NOISE_BPB)));
  svg.append('path').datum(stairs.steps).attr('class', 'best')
    .attr('d', d3.line().curve(d3.curveStepAfter).x((s) => x(s.n)).y((s) => y(s.best)));
  svg.append('g').selectAll('circle').data(pts).join('circle')
    .attr('class', (p) => `dot ${p.kept ? 'kept' : 'discarded'}${p.n === selected ? ' selected' : ''}`)
    .attr('cx', (p) => x(p.n)).attr('cy', (p) => y(p.val_bpb)).attr('r', 7).attr('tabindex', 0)
    .on('click', (ev, p) => onPick(p))
    .on('keydown', (ev, p) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); onPick(p); } })
    .append('title').text((p) => `#${p.n} ${p.description}: ${p.val_bpb.toFixed(6)} (${p.status})`);
  svg.append('g').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(Math.min(pts.length, 12)).tickFormat(d3.format('d')));
  svg.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.3f')));
  svg.append('text').attr('x', (m.l + W - m.r) / 2).attr('y', H - 6).attr('text-anchor', 'middle').text('Experiment');
  svg.append('text').attr('transform', `translate(16,${(H - m.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('Score (lower is better)');
}

function evolutionDetail(c, p) {
  const byCommit = new Map(c.runIds.map((key) => [LIB.runs.get(key).commit, key]));
  const delta = p.bestBefore === null ? null : p.val_bpb - p.bestBefore;
  const verdict = p.bestBefore === null ? 'The baseline: where the agent started.'
    : `${delta < 0 ? 'Better' : 'Worse'} than the best before it by ${Math.abs(delta).toFixed(4)}${Math.abs(delta) < NOISE_BPB ? ', inside the noise' : ''}. ${p.kept ? 'Kept.' : 'Thrown away.'}`;
  const mine = byCommit.get(p.commit);
  const before = p.bestCommitBefore && byCommit.get(p.bestCommitBefore);
  const last = (key) => LIB.runs.get(key).snapshots.length - 1;
  return el('div', { class: 'detail' },
    el('h3', { text: `#${p.n} ${p.description}` }),
    el('p', { class: 'facts', text: `Score ${p.val_bpb.toFixed(6)}. ${verdict}` }),
    mine && before
      ? el('div', { class: 'pair' },
        el('div', {}, el('h3', { text: 'This experiment' }), samplesBlock(mine, LIB.runs.get(mine), last(mine))),
        el('div', {}, el('h3', { text: 'The best before it' }), samplesBlock(before, LIB.runs.get(before), last(before))))
      : el('p', { class: 'empty', text: 'No run files for this pair of experiments, so there is no writing to show.' }));
}

function renderEvolution(root) {
  const e = STATE.evolution;
  const sessions = LIB.collections.filter((c) => c.rows && c.rows.length);
  if (!sessions.length) return root.append(el('p', { class: 'empty', text: 'No agent session (results.tsv) is loaded.' }));
  if (!sessions.some((c) => c.id === e.collection)) { e.collection = sessions[0].id; e.point = null; }
  const c = sessions.find((x) => x.id === e.collection);
  const sel = el('select', { id: 'evo_session', 'aria-label': 'Agent session' },
    sessions.map((x) => el('option', { value: x.id, selected: x.id === c.id, text: x.title })));
  sel.addEventListener('change', () => { e.collection = sel.value; e.point = null; showView('evolution'); });
  const stairs = staircase(c.rows);
  const chart = el('div', { id: 'evolution_chart' });
  root.append(el('div', { class: 'controls' }, sel), c.note ? el('p', { class: 'facts', text: c.note }) : null, chart,
    el('p', { class: 'caption', text: evolutionCaption(stairs) }));
  drawStaircase(chart, stairs, e.point, (p) => { e.point = p.n; showView('evolution'); });
  const p = stairs.points.find((x) => x.n === e.point);
  if (p) root.append(evolutionDetail(c, p));
}
```

- [ ] **Step 2: Test and look**

Run: `npm test`
Expected: green.
Browser, `/training/?view=tokens`: three rows of chips per sentence, built-from-data shortest; switching dataset changes the unused share; "First words" shows three portraits. `/training/?view=evolution`: the May session's staircase (5 filled steps, 11 hollow dots, the 0.5376 outlier clamped at the top), the caption says 0.0014 (0.27%), less than the noise; clicking a dot shows its description and score and "No run files" for May.

- [ ] **Step 3: Commit**

```bash
git add training/src/views.js
git commit -m "training: Tokens and Evolution views"
```

### Task 8: Your own runs

**Files:**
- Modify: `training/src/ownRuns.js` (replace the stubs)
- Test: `test/training/OwnRuns.test.js`

**Interfaces:**
- Consumes: `parseRepoUrl`, `pickRepoFiles`, `isRunFile`, `parseResultsTsv` (Task 3).
- Produces: `loadRepo(url) -> Promise<{collection} | {error}>`, `loadDropped(files: [{path, text: () => Promise<string>}], name) -> same`, `filesFromEntries(entries) -> Promise<[{path, text}]>`; `collection = {id, title, runs: {key: run}, rows: rows|null}`.

- [ ] **Step 1: Write the failing tests**

```js
/** Loading a learner's runs: GitHub answers and dropped folders. */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { loadSketch, sketch } = require('../helpers/sketch');

const RUN = { schema: 1, run_id: 'r1', dataset: 'folktales', prompts: ['a'], snapshots: [{ t_s: 0, samples: ['x'] }] };
const json = (body, status = 200) => new Response(JSON.stringify(body), { status });
const text = (body, status = 200) => new Response(body, { status });
const load = (fetch) => loadSketch(sketch('training', ['data.js', 'ownRuns.js']), { fetch });

test('a public repo: its run files and its scoreboard', async () => {
  const s = load(async (url) => {
    url = String(url);
    if (url.includes('/contents/runs')) return json([{ type: 'file', name: 'r1.json', download_url: 'https://raw/r1.json' }, { type: 'file', name: 'x.txt', download_url: 'u' }]);
    if (url === 'https://raw/r1.json') return json(RUN);
    if (url.endsWith('/results.tsv')) return text('timestamp\tcommit\tval_bpb\tstatus\tdescription\nt\tabc\t1.38\tkeep\tbaseline\n');
    return text('', 404);
  });
  const out = await s.get('loadRepo')('https://github.com/you/night');
  assert.equal(out.collection.title, 'Your runs: you/night');
  assert.deepEqual(Object.keys(out.collection.runs), ['yours:you/night/runs/r1.json']);
  assert.equal(out.collection.rows.length, 1);
});

test('a repo whose agent ran before capture: scoreboard only', async () => {
  const s = load(async (url) => (String(url).includes('/contents/runs') ? json({ message: 'Not Found' }, 404)
    : String(url).endsWith('/results.tsv') ? text('timestamp\tcommit\tval_bpb\tstatus\tdescription\nt\tabc\t0.52\tkeep\tbaseline\n') : text('', 404)));
  const out = await s.get('loadRepo')('https://github.com/you/may');
  assert.deepEqual(Object.keys(out.collection.runs), []);
  assert.equal(out.collection.rows.length, 1);
});

test('GitHub refuses: readable errors', async () => {
  const notFound = await load(async () => text('', 404)).get('loadRepo')('https://github.com/you/private');
  assert.match(notFound.error, /public/);
  const limited = await load(async () => text('', 403)).get('loadRepo')('https://github.com/you/r');
  assert.match(limited.error, /60 an hour/);
  const bad = await load(async () => text('')).get('loadRepo')('not a url');
  assert.match(bad.error, /GitHub repo address/);
});

test('a dropped folder: only its own runs/ and results.tsv', async () => {
  const s = load(async () => text('', 404));
  const file = (path, body) => ({ path, text: async () => body });
  const out = await s.get('loadDropped')([
    file('night/runs/r1.json', JSON.stringify(RUN)),
    file('night/runs/broken.json', '{'),
    file('night/library/baselines/runs/x.json', JSON.stringify(RUN)),
    file('night/results.tsv', 'timestamp\tcommit\tval_bpb\tstatus\tdescription\n'),
  ], 'your folder');
  assert.deepEqual(Object.keys(out.collection.runs), ['yours:your folder/night/runs/r1.json']);
  const empty = await s.get('loadDropped')([file('x/notes.txt', 'hi')], 'your folder');
  assert.match(empty.error, /runs\//);
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `node --test test/training/OwnRuns.test.js`
Expected: FAIL (stubs return "not built yet").

- [ ] **Step 3: Write `training/src/ownRuns.js`**

```js
/**
 * A learner's own runs, read in the browser: from a public GitHub repo (the REST contents API
 * lists runs/; anonymous, 60 requests an hour) or from a dropped or chosen folder. Nothing is
 * uploaded anywhere. Keys are `yours:<source>/<path>` so they never collide with the library.
 */
const MAX_RUNS = 200;

async function loadRepo(input) {
  const repo = parseRepoUrl(input);
  if (!repo) return { error: 'That doesn’t look like a GitHub repo address, for example https://github.com/you/autoresearch-starter.' };
  const name = `${repo.owner}/${repo.repo}`;
  const ref = repo.ref ? `?ref=${encodeURIComponent(repo.ref)}` : '';
  let listing = [];
  const res = await fetch(`https://api.github.com/repos/${name}/contents/runs${ref}`);
  if (res.status === 403 || res.status === 429) return { error: 'GitHub’s limit for anonymous requests is reached (60 an hour). Try again later, or choose the folder instead.' };
  if (res.ok) listing = await res.json();
  else if (res.status !== 404) return { error: `GitHub answered ${res.status} for ${name}.` };
  const runs = {};
  for (const f of (Array.isArray(listing) ? listing : []).filter((x) => x.type === 'file' && x.name.endsWith('.json')).slice(0, MAX_RUNS)) {
    try {
      const run = await (await fetch(f.download_url)).json();
      if (isRunFile(run)) runs[`yours:${name}/runs/${f.name}`] = run;
    } catch (err) { /* a file that is not a run file is skipped */ }
  }
  let rows = null;
  const raw = await fetch(`https://raw.githubusercontent.com/${name}/${repo.ref || 'HEAD'}/results.tsv`);
  if (raw.ok) rows = parseResultsTsv(await raw.text());
  if (!Object.keys(runs).length && !rows) {
    return { error: `Nothing to load from ${name}: no runs/ folder and no results.tsv. Is the repo public, and has a night's work been pushed?` };
  }
  return { collection: { id: `yours:${name}`, title: `Your runs: ${name}`, runs, rows } };
}

async function loadDropped(files, label) {
  const picked = pickRepoFiles(files.map((f) => f.path));
  const byPath = new Map(files.map((f) => [f.path, f]));
  const runs = {};
  for (const p of picked.runs.slice(0, MAX_RUNS)) {
    try {
      const run = JSON.parse(await byPath.get(p).text());
      if (isRunFile(run)) runs[`yours:${label}/${p}`] = run;
    } catch (err) { /* skipped */ }
  }
  const rows = picked.results ? parseResultsTsv(await byPath.get(picked.results).text()) : null;
  if (!Object.keys(runs).length && !(rows && rows.length)) {
    return { error: 'No run files found. Choose the repo’s own folder: the one holding runs/ and results.tsv.' };
  }
  return { collection: { id: `yours:${label}`, title: `Your runs: ${label}`, runs, rows } };
}

/** Walk dropped directory entries into [{path, text}]. */
async function filesFromEntries(entries) {
  const out = [];
  const walk = async (entry, prefix) => {
    if (entry.isFile) {
      const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
      out.push({ path: prefix + entry.name, text: () => file.text() });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        for (const child of batch) await walk(child, prefix + entry.name + '/');
      } while (batch.length);
    }
  };
  for (const e of entries) await walk(e, '');
  return out;
}
```

- [ ] **Step 4: Run the tests**

Run: `node --test test/training/OwnRuns.test.js` then `npm test`
Expected: 4 passed; all green.

- [ ] **Step 5: Commit**

```bash
git add training/src/ownRuns.js test/training/OwnRuns.test.js
git commit -m "training: load your own runs from a public repo or a folder"
```

### Task 9: README, presentation snapshot, browser check, PR and preview

**Files:**
- Create: `training/README.md`, `test/snapshots/training/presentation-1280x800.json`

- [ ] **Step 1: README**

`training/README.md`: what the page shows (one paragraph), the four views (one line each), a Controls table (Ctrl+Alt+P presentation; ← → step the moment in Growth and Compare; Enter on a dot in Evolution), a Three modes table (presentation: larger type, loader hidden; desktop; mobile: Compare stacks), "Your runs" (public repo URL or folder; nothing uploaded), and Provenance (`library/SOURCE.json`, the sync command, licences as in the worked example's `library/README.md`).

- [ ] **Step 2: Browser check and the layout snapshot**

`npx serve .`, open `/training/?mode=presentation` at 1280×800, and capture every element with an id:
```js
() => [...document.querySelectorAll('[id]')].map((e) => { const r = e.getBoundingClientRect(); const cs = getComputedStyle(e);
  return { id: e.id, x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height), display: cs.display, dir: cs.flexDirection }; })
```
Save it as `test/snapshots/training/presentation-1280x800.json`. Then walk all four views at desktop width and at 390 px wide (mobile), load `https://github.com/aroughidea/autoresearch-win-rtx` (the May scoreboard arrives as "Your runs"), and check the console for errors.

- [ ] **Step 3: Full checks, commit, PR, preview**

Run: `npm test` and `git diff --check`
Expected: green, clean.
```bash
git add training/README.md test/snapshots/training
git commit -m "training: README and presentation layout baseline"
git push -u origin feat/training-explorer
gh pr create --repo tj60647/llm-explorables --base main --head feat/training-explorer --title "Training Decisions explorer" --body-file <body>
```
Wait for the Vercel preview of this commit to reach READY, open `/training/` on it in a browser, and hand back the PR and preview links. Merging to `main` (production) waits for TJ.

---

## After the overnight sessions (not part of this plan's build)

When a session finishes: copy its `results.tsv` and `runs/` into `library/sessions/<dataset>-<date>/` with a `collection.json`, rerun `uv run library/report.py`, merge, then `node scripts/sync-training-library.mjs --ref <merge commit>` in llm-explorables and open a PR.
