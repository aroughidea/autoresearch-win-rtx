# Phase A1: 10-minute Study, Steps and Epochs, Tokenizer Export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 10-minute training study to the library, define and show steps and epochs in the Training Decisions explorer, and export the home-made vocabularies as Hugging Face `tokenizer.json` files proven identical to our tokenizers (the first piece of the tie-in).

**Architecture:** In the worked example, `prepare.py` reads an optional `AUTORESEARCH_TIME_BUDGET` (set only by a person, for a study); `capture.py` adds a 5-minute snapshot to longer runs and records the budget in each run file. `library/report.py` adds an estimate of each dataset's size in tokens (so the explorer can turn steps into passes through the data) and writes the HF exports after a parity check. A study script trains the six baselines and the Folktales session's best recipe for 10 minutes into `library/study-10min/`. In llm-explorables, the explorer accepts a `study` collection, lists a change of training time as a decision, and defines steps and epochs beside each run's real numbers.

**Tech Stack:** Python 3 + pytest, `tiktoken` 0.12, `tokenizers` 0.23 (worked example, `uv run python -m pytest`; the bare `pytest.exe` is blocked by Smart App Control in the starter); classic-script JavaScript with `node --test` (llm-explorables).

**Spec:** `docs/superpowers/specs/2026-10-06-training-decisions-journey-design.md`, sections "Tie-in" (control tokens, export) and the 8 October decisions; TJ's requests of 8 October: step and epoch defined in the explorables, the 10-minute study folded into Phase A.

## Global Constraints

- The agent never changes the time budget: `program.md` forbids setting `AUTORESEARCH_TIME_BUDGET`, as it forbids the dataset and tokenizer variables. Every agent experiment stays at 5 minutes.
- Scores from runs with different budgets are not compared for keep or discard; the study is its own collection.
- Exported vocabularies keep every id; only the two control tokens' display names change (`<|startoftext|>`, `<|endoftext|>`, CLIP's convention). `prepare.py`'s names stay, so no learner's cache is rebuilt.
- An export is written only if it tokenizes a real sample identically to our tokenizer (parity), token for token.
- Nothing trains while an agent session runs: the active dataset is machine-wide.
- Plain words backed by professional terms (glossary): step = optimizer step; epoch = one full pass over the training data [Goodfellow and others, *Deep Learning*].

## Review Focus

1. **A run file without `time_budget_s`** (every run made before this change): treated as 5 minutes. Test in Task 4.
2. **Comparing a 5-minute and a 10-minute run:** the diff names training time; the shared scrubber uses the shorter run. Test in Task 4.
3. **A dataset the explorer has no size for** (a learner's own dataset): steps shown, passes omitted. Test in Task 4.
4. **Text the split pattern treats specially** (numbers, contractions, newlines, runs of spaces, accented letters, emoji): the export splits it exactly as tiktoken does. Test in Task 2, and the parity check over real documents.
5. **A bad `AUTORESEARCH_TIME_BUDGET`** (text, a decimal, too short, too long): training refuses with a message naming the variable. Test in Task 1.

---

### Task 1: An optional time budget, a 5-minute snapshot for longer runs, the budget in each run file

**Files:** Modify `prepare.py`, `capture.py`, `program.md`; test `tests/test_time_budget.py` (new) and `tests/test_capture.py`. Then port the same diff to the starter (`autoresearch-starter`, branch `main`).

**Interfaces:** Produces `prepare._time_budget_from_env(env) -> int`, `prepare.TIME_BUDGET`, `capture.snapshot_times_for(budget) -> tuple`, `capture.SNAPSHOT_TIMES`, `RunCapture(..., time_budget_s=TIME_BUDGET)`, run file field `time_budget_s`.

- [ ] **Step 1: Failing tests** — `tests/test_time_budget.py`:

```python
import pytest

import capture
import prepare


def test_time_budget_defaults_to_five_minutes():
    assert prepare._time_budget_from_env({}) == 300
    assert prepare._time_budget_from_env({"AUTORESEARCH_TIME_BUDGET": ""}) == 300


def test_a_study_can_set_a_longer_budget():
    assert prepare._time_budget_from_env({"AUTORESEARCH_TIME_BUDGET": "600"}) == 600


@pytest.mark.parametrize("bad", ["ten", "600.5", "30", "4000", "-600"])
def test_a_bad_budget_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="AUTORESEARCH_TIME_BUDGET"):
        prepare._time_budget_from_env({"AUTORESEARCH_TIME_BUDGET": bad})


def test_longer_runs_also_sample_at_five_minutes():
    assert capture.snapshot_times_for(300) == (0, 10, 30, 60, 120)
    assert capture.snapshot_times_for(600) == (0, 10, 30, 60, 120, 300)
```

and in `tests/test_capture.py`:

```python
def test_run_file_records_the_time_budget(tmp_path):
    record = json.loads(_finish(_capture(tmp_path)).read_text(encoding="utf-8"))
    assert record["time_budget_s"] == 300
```

Run `uv run python -m pytest -q tests/test_time_budget.py tests/test_capture.py`. Expected: the new tests fail (no `_time_budget_from_env`, `snapshot_times_for`, `time_budget_s`).

- [ ] **Step 2: Implement.** In `prepare.py`, replace `TIME_BUDGET = 300` with:

```python
def _time_budget_from_env(env=None):
    """Training seconds per run: 300 unless AUTORESEARCH_TIME_BUDGET sets another (60 to 3600).
    Only a person sets it, for a study; the agent never does (program.md)."""
    env = os.environ if env is None else env
    raw = (env.get("AUTORESEARCH_TIME_BUDGET") or "").strip()
    if not raw:
        return 300
    if not raw.isdigit() or not 60 <= int(raw) <= 3600:
        raise ValueError(f"AUTORESEARCH_TIME_BUDGET must be whole seconds from 60 to 3600, not {raw!r}")
    return int(raw)


TIME_BUDGET = _time_budget_from_env()  # training time budget in seconds (5 minutes unless a study sets it)
```

In `capture.py`: `from prepare import TIME_BUDGET`; replace `SNAPSHOT_TIMES = (0, 10, 30, 60, 120)` with

```python
BASE_SNAPSHOT_TIMES = (0, 10, 30, 60, 120)


def snapshot_times_for(budget):
    """Moments to sample, in training seconds. A run longer than 5 minutes also samples at
    5 minutes, so it lines up with the normal runs."""
    return BASE_SNAPSHOT_TIMES + ((300,) if budget > 300 else ())


SNAPSHOT_TIMES = snapshot_times_for(TIME_BUDGET)
```

add `time_budget_s=TIME_BUDGET` to `RunCapture.__init__` (stored as `self.time_budget_s`), and `"time_budget_s": self.time_budget_s,` after `"dataset"` in the record. In `program.md`, after the dataset and tokenizer rule under **What you CANNOT do**, add: `- Change the time budget: do not set AUTORESEARCH_TIME_BUDGET. Every experiment trains for the same 5 minutes, so scores compare.`

- [ ] **Step 3:** Run the whole suite (`uv run python -m pytest -q`). Expected: all pass. Commit `capture: an optional time budget for studies, recorded in each run file`. Port to the starter (same diff, `main`), run its suite, commit.

### Task 2: Token-count estimates and the Hugging Face export, with parity

**Files:** Modify `library/report.py`; test `tests/test_library_report.py`.

**Interfaces:** Produces `report.train_token_estimate(chars, docs, chars_per_token) -> int`; `report.bytes_to_unicode() -> dict`; `report.merges_from_ranks(ranks) -> list[tuple[bytes, bytes]]`; `report.hf_tokenizer_json(ranks, specials, pattern) -> dict`; `report.HF_PATTERN`; `tokens.json` field `train_tokens` per dataset and tokenizer; files `library/tokenizers/<dataset>-own.json` listed in `index.json` as `"tokenizers": {"<dataset>-own": {"file", "sha256", "size"}}`.

- [ ] **Step 1: Failing tests:**

```python
import tiktoken
from tokenizers import Tokenizer as HFTokenizer


def test_train_token_estimate_counts_the_document_markers():
    assert report.train_token_estimate(chars=4000, docs=10, chars_per_token=4.0) == 1020


def _tiny_encoding():
    ranks = {bytes([b]): b for b in range(256)}
    for merged in (b"th", b"the", b" t", b" the", b"in", b"ing", b"10"):
        ranks[merged] = len(ranks)
    specials = {"<|reserved_0|>": len(ranks), "<|reserved_1|>": len(ranks) + 1}
    return tiktoken.Encoding(name="tiny", pat_str=prepare.SPLIT_PATTERN, mergeable_ranks=ranks, special_tokens=specials)


def test_hf_export_tokenizes_exactly_like_tiktoken():
    enc = _tiny_encoding()
    hf = HFTokenizer.from_str(json.dumps(report.hf_tokenizer_json(enc._mergeable_ranks, enc._special_tokens, report.HF_PATTERN)))
    for text in ["the thing", "  two  spaces", "line\n\nbreaks\r\n", "it's 1010 they'll", "été café", "emoji \U0001F642!", "the.the,the"]:
        assert hf.encode(text, add_special_tokens=False).ids == enc.encode_ordinary(text), text


def test_hf_export_names_the_control_tokens_by_clips_convention():
    enc = _tiny_encoding()
    added = report.hf_tokenizer_json(enc._mergeable_ranks, enc._special_tokens, report.HF_PATTERN)["added_tokens"]
    assert [(a["id"], a["content"]) for a in added] == [(263, "<|startoftext|>"), (264, "<|endoftext|>")]
```

Run `uv run python -m pytest -q tests/test_library_report.py`. Expected: the three new tests fail.

- [ ] **Step 2: Implement** in `library/report.py`:

```python
# tiktoken's pattern uses possessive quantifiers (?+ ++), which JavaScript lacks. Here they match
# what the plain forms match: the optional character can never be a letter, and the trailing
# group can always match empty. The parity check below proves it on real text.
HF_PATTERN = SPLIT_PATTERN.replace("?+", "?").replace("++", "+")
CONTROL_NAMES = {"<|reserved_0|>": "<|startoftext|>", "<|reserved_1|>": "<|endoftext|>"}


def train_token_estimate(chars, docs, chars_per_token):
    """Tokens in one pass over the training split: its text, plus a start and an end marker per document."""
    return round(chars / chars_per_token + 2 * docs)


def bytes_to_unicode():
    """GPT-2's byte alphabet: each byte as a printable character."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars, extra = keep[:], 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(keep, chars)}


def _bpe_parts(ranks, token, max_rank):
    parts = [bytes([b]) for b in token]
    while True:
        best = None
        for i in range(len(parts) - 1):
            rank = ranks.get(parts[i] + parts[i + 1])
            if rank is not None and rank < max_rank and (best is None or rank < best[1]):
                best = (i, rank)
        if best is None:
            return parts
        i = best[0]
        parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]


def merges_from_ranks(ranks):
    """The merge list a BPE model needs, recovered from tiktoken's ranks (each multi-byte token is
    the merge of the two parts BPE reaches with only lower-ranked merges)."""
    merges = []
    for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(token) > 1:
            parts = _bpe_parts(ranks, token, rank)
            assert len(parts) == 2, token
            merges.append((parts[0], parts[1]))
    return merges


def hf_tokenizer_json(ranks, specials, pattern):
    alphabet = bytes_to_unicode()
    show = lambda bs: "".join(alphabet[b] for b in bs)
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": i, "content": CONTROL_NAMES.get(name, name), "single_word": False, "lstrip": False,
                          "rstrip": False, "normalized": False, "special": True} for name, i in sorted(specials.items(), key=lambda kv: kv[1])],
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": {show(t): r for t, r in ranks.items()},
                  "merges": [f"{show(a)} {show(b)}" for a, b in merges_from_ranks(ranks)]},
    }
```

(`added_tokens` lists only `reserved_0` and `reserved_1` renamed; `reserved_2`, `reserved_3` keep their names.) In `tokens_report`, count the training split once per dataset (`_iter_tinystories_texts("train", dataset)`, the iterator the data loader uses): `chars`, `docs`, then `train_tokens = train_token_estimate(chars, docs, chars_per_token)` per tokenizer. In `main`, for each dataset's `own` tokenizer: build the export, run `parity(enc, hf, docs[:20000])` (identical ids on every document, else raise and write nothing), write `library/tokenizers/<dataset>-own.json`, and list it with its sha256 and size in `index.json`.

- [ ] **Step 3:** Run the suite. Expected: all pass. Commit `library: token-count estimates, and the home-made vocabularies exported as tokenizer.json with a parity check`.

### Task 3: Run the 10-minute study and publish it

**Files:** Create `library/make_study.sh`, `library/study-10min/collection.json`; modify `library/README.md`, `TRAINING-DECISIONS.md` (glossary: step, epoch, batch size; sources for Hoffmann and others 2022 and Wu and others 2018).

- [ ] **Step 1:** `library/make_study.sh` (GPU idle; no session running): for each dataset × tokenizer, `git show 6ad8ddd:train.py > train.py` (the starter recipe), `uv run prepare.py --dataset D --tokenizer T`, `AUTORESEARCH_TIME_BUDGET=600 uv run train.py`, move the new run file to `library/study-10min/runs/D-T.json` and the checkpoint to `library/checkpoints/study-D-T.pt`. Then the Folktales session's best recipe: `git show f9352a8:train.py > train.py`, Folktales and its own tokenizer, the same 600 seconds, to `folktales-own-agent-best.json`. Finally restore `train.py`, prepare TinyStories and its own tokenizer, restore `checkpoint_pre_eval.pt`. About 1 hour 45 minutes.
- [ ] **Step 2:** Check each run file has `time_budget_s: 600` and seven snapshots ending near 600 s; compare each 10-minute score with its 5-minute baseline.
- [ ] **Step 3:** `collection.json`: `{"title": "Ten-minute study", "kind": "study", "note": "The six baselines and the Folktales session's best recipe, trained for 10 minutes instead of 5."}`; rerun `uv run library/report.py`; add the results table and what it shows (catch-up of the bigger vocabularies; whether the agent's 5-minute win holds) to `library/README.md`; glossary rows for step, epoch and batch size. Commit, PR, merge.

### Task 4: The explorer defines steps and epochs and shows the study

**Files:** Modify `training/src/data.js`, `training/src/views.js`, `training/README.md`; test `test/training/Data.test.js`; sync `training/library/`.

**Interfaces:** Produces `budgetOf(run) -> seconds`, `passesAt(run, i, tokens) -> number|null`, `passesText(passes, datasetName) -> string`, `stepNote(run) -> string`; `validateIndex` accepts kind `study`; `recipeDiff` lists `training time` first when budgets differ.

- [ ] **Step 1: Failing tests** in `Data.test.js`:

```js
test('training time: a run without a budget is 5 minutes; a change of budget is a decision', () => {
  assert.equal(f('budgetOf')(run()), 300);
  const d = f('recipeDiff')(run(), run({ time_budget_s: 600 }));
  assert.deepEqual(plain(d[0]), { key: 'training time', a: '5 min', b: '10 min' });
});

test('passes through the data: steps times batch over the dataset size', () => {
  const tokens = { datasets: { tinystories: { own: { train_tokens: 500000000 } }, folktales: { own: { train_tokens: 3000000 } } } };
  const ts = run();                                   // TOTAL_BATCH_SIZE absent -> 32768
  ts.snapshots[5].step = 639;
  assert.equal(Math.round(f('passesAt')(ts, 5, tokens) * 1000), 42);
  assert.match(f('passesText')(f('passesAt')(ts, 5, tokens), 'TinyStories'), /about 4% of one pass through TinyStories/);
  const fk = run({ dataset: 'folktales' }); fk.snapshots[5].step = 620;
  assert.match(f('passesText')(f('passesAt')(fk, 5, tokens), 'Folktales'), /about 7 passes through Folktales/);
  assert.equal(f('passesAt')(run({ dataset: 'mine' }), 5, tokens), null, 'unknown dataset: no passes');
});

test('the step note defines a step and an epoch with the run’s batch size', () => {
  assert.match(f('stepNote')(run()), /A step is one update of the model’s weights, from a batch of 32,768 tokens/);
  assert.match(f('stepNote')(run()), /An epoch is one full pass through the training data/);
  assert.deepEqual(plain(f('validateIndex')({ schema: 1, collections: [{ id: 's', title: 'S', kind: 'study', runs: [], results: null }], tokens: 't', copies: 'c' })), []);
});
```

Run `node --test test/training/Data.test.js`. Expected: the new tests fail.

- [ ] **Step 2: Implement** in `data.js`:

```js
function budgetOf(run) { return (run && run.time_budget_s) || 300; }

function passesAt(run, i, tokens) {
  const snap = run.snapshots[i];
  const t = (((tokens || {}).datasets || {})[run.dataset] || {})[(run.tokenizer || {}).name];
  if (!snap || snap.step === undefined || !t || !t.train_tokens) return null;
  const batch = (run.recipe || {}).TOTAL_BATCH_SIZE || 32768;
  return (snap.step * batch) / t.train_tokens;
}

function passesText(p, datasetName) {
  if (p === null || p === undefined) return '';
  if (p < 0.95) return `about ${Math.max(1, Math.round(p * 100))}% of one pass through ${datasetName}`;
  return `about ${p < 1.95 ? p.toFixed(1) : Math.round(p)} passes through ${datasetName}`;
}

function stepNote(run) {
  const batch = ((run.recipe || {}).TOTAL_BATCH_SIZE || 32768).toLocaleString('en-GB');
  return `A step is one update of the model’s weights, from a batch of ${batch} tokens. An epoch is one full pass through the training data.`;
}
```

`validateIndex` accepts `['baselines', 'session', 'study']`; `recipeDiff` unshifts `{ key: 'training time', a: timeLabel(budgetOf(a)), b: timeLabel(budgetOf(b)) }` when the budgets differ. In `views.js`: Growth's caption adds the passes (`After 5 min of training (639 steps, about 4% of one pass through TinyStories).`), its legend line adds `stepNote(run)`; Compare's cost line adds the passes. Run the tests; sync the library at the Task 3 merge commit; check in a browser (Growth on a 10-minute run shows seven moments; Compare of a 5- and 10-minute run lists training time; Folktales says about 7 passes; the TinyStories figure agrees with the training log's "epoch 1"); README; commit; push.

### Task 5: Final review and hand-off

- [ ] Fresh reviewer (most capable model) over both repos' diffs with this plan and Review Focus; fix Critical and Important with a failing test first; ledger the rest; PRs per repo (the explorer's into PR #34's branch or a follow-up PR, whichever is open); preview READY.

---

## Self-review

Spec coverage: steps and epochs in the explorables (Task 4), the 10-minute study (Tasks 1, 3), the export with CLIP-named control tokens and parity (Task 2). Placeholders: none. Names: `budgetOf`, `passesAt`, `passesText`, `stepNote`, `snapshot_times_for`, `_time_budget_from_env`, `hf_tokenizer_json` match across tasks.
