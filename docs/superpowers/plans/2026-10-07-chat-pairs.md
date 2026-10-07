# Chat Page Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `chat.py` loads every model with the dataset and tokenizer it was trained with, lets a participant pick any two models, labels each with its facts, and shows how a model's writing grew during training.

**Architecture:** Pure helpers in `chat.py` work out each checkpoint's commit, run file, pair, vocabulary size and scoreboard row. `ModelStore` caches one tokenizer per pair and marks mismatched models unavailable. New routes `/growth` and `/vocab?model_id=` serve the run file's snapshots and a model's own vocabulary. The page gains two pickers, a facts line per pane, and a "Watch it learn" card. Built in the worked example, then ported to the starter.

**Tech Stack:** Python 3.10+, FastAPI, PyTorch 2.9.1, vanilla JS inside `chat.py`'s `_HTML` string, pytest.

**Spec:** `docs/superpowers/specs/2026-10-06-training-decisions-journey-design.md`, section "Component: chat page upgrade".

## Global Constraints

- `_HTML` is a normal Python string: a backslash in its JavaScript is written `\\` in the file (as the existing `'\\n'` shows).
- Pair resolution, exactly: run file `runs/<checkpoint stem>.json` → its `dataset` and `tokenizer.name`; else `checkpoint_pre_eval.pt` → the active pair; else the defaults (`tinystories`, `own`).
- A model is usable only if its tokenizer loads and its `transformer.wte.weight` row count equals the tokenizer's vocabulary size. Otherwise list it with `available: false` and a reason, and never generate with it.
- Public-demo mode (`SPACE_MODE=1`) keeps its clamps and banner unchanged.
- No new dependencies. Tests are CPU-only.
- Any `train.py` run in the worked example: restore `checkpoint_pre_eval.pt` with `git checkout -- checkpoint_pre_eval.pt` afterwards; any `prepare.py` run: restore the pair with `uv run prepare.py --dataset tinystories --tokenizer own`.
- Branch `feat/chat-pairs`. Merge with the usual PR flow.

## Review Focus

1. **Models from the May session** (no run files, default pair): load and generate exactly as before. Test in Task 1 (`test_archived_checkpoint_without_run_uses_defaults`).
2. **A model whose tokenizer is missing from the cache** (its pair was never prepared on this machine): listed as unavailable with a reason, not a crash at startup. Test in Task 2 (`test_missing_tokenizer_marks_model_unavailable`).
3. **A malformed run file:** treated as absent. Test in Task 1 (`test_malformed_run_file_is_ignored`).
4. **The public demo with only two checkpoints and no `runs/`:** pickers still work; the growth card stays hidden. Covered by Task 3's manual check of `/models` with no run files.
5. **`results.tsv` rows with blank or `missing` scores:** facts omit the score instead of failing. Test in Task 1 (`test_results_by_commit_tolerates_blank_scores`).

---

### Task 1: Pair, commit, run-file and scoreboard helpers

**Files:**
- Modify: `chat.py` (imports; new helpers after `_discover_checkpoints`)
- Create: `tests/test_chat_pairs.py`

**Interfaces:**
- Produces: `DATASET_BLURBS`, `_commit_from_name(name) -> str | None`, `_run_for_checkpoint(path, runs_dir="runs") -> dict | None`, `_pair_for_checkpoint(path, run) -> tuple[str, str]`, `_checkpoint_vocab_rows(path) -> int | None`, `_results_by_commit(path="results.tsv") -> dict[str, dict]`, `_growth_from_run(run) -> dict | None`.

- [ ] **Step 1: Failing tests** — create `tests/test_chat_pairs.py`:

```python
import json

import torch

import chat


def test_commit_from_name():
    assert chat._commit_from_name("20260523T175831-0700_e9fffd9.pt") == "e9fffd9"
    assert chat._commit_from_name("checkpoint_pre_eval.pt") is None


def _write_run(tmp_path, stem, dataset="folktales", tokenizer="phi3", **extra):
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    record = {"dataset": dataset, "tokenizer": {"name": tokenizer, "vocab_size": 32015}, **extra}
    (runs / f"{stem}.json").write_text(json.dumps(record), encoding="utf-8")
    return runs


def test_run_file_found_by_checkpoint_stem(tmp_path):
    runs = _write_run(tmp_path, "20261007T010000-0700_abc1234")
    run = chat._run_for_checkpoint("checkpoints/20261007T010000-0700_abc1234.pt", runs_dir=str(runs))
    assert run["dataset"] == "folktales"
    assert chat._pair_for_checkpoint("checkpoints/20261007T010000-0700_abc1234.pt", run) == ("folktales", "phi3")


def test_malformed_run_file_is_ignored(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "x_abc1234.json").write_text("{not json", encoding="utf-8")
    assert chat._run_for_checkpoint("checkpoints/x_abc1234.pt", runs_dir=str(runs)) is None


def test_archived_checkpoint_without_run_uses_defaults():
    assert chat._pair_for_checkpoint("checkpoints/20260523T175831-0700_e9fffd9.pt", None) == ("tinystories", "own")


def test_pre_eval_checkpoint_uses_the_active_pair(monkeypatch):
    monkeypatch.setattr(chat, "_resolve_dataset_name", lambda name=None: "folktales")
    monkeypatch.setattr(chat, "_resolve_tokenizer_name", lambda name=None: "gpt2")
    assert chat._pair_for_checkpoint("checkpoint_pre_eval.pt", None) == ("folktales", "gpt2")


def test_checkpoint_vocab_rows(tmp_path):
    path = tmp_path / "m.pt"
    torch.save({"transformer.wte.weight": torch.zeros(37, 4)}, path)
    assert chat._checkpoint_vocab_rows(str(path)) == 37
    assert chat._checkpoint_vocab_rows(str(tmp_path / "missing.pt")) is None


def test_results_by_commit_tolerates_blank_scores(tmp_path):
    tsv = tmp_path / "results.tsv"
    tsv.write_text(
        "timestamp\tcommit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "2026-05-23T15:57:43-07:00\t75027e8\t0.520096\t6.6\tkeep\tbaseline\n"
        "2026-05-23T16:00:00-07:00\tdeadbee\tmissing\t0.0\tcrash\toom\n",
        encoding="utf-8",
    )
    rows = chat._results_by_commit(str(tsv))
    assert rows["75027e8"] == {"val_bpb": 0.520096, "status": "keep", "description": "baseline"}
    assert rows["deadbee"]["val_bpb"] is None


def test_growth_from_run():
    run = {"prompts": ["Once upon a time"], "snapshots": [
        {"t_s": 0.0, "samples": ["noise"]}, {"t_s": 300.4, "samples": ["a story"], "final": True}]}
    growth = chat._growth_from_run(run)
    assert growth["prompts"] == ["Once upon a time"]
    assert [s["t_s"] for s in growth["snapshots"]] == [0.0, 300.4]
    assert chat._growth_from_run({"prompts": [], "snapshots": []}) is None
    assert chat._growth_from_run(None) is None
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_chat_pairs.py -q`
Expected: 8 failures, `AttributeError: module 'chat' has no attribute '_commit_from_name'` (and the other helpers).

- [ ] **Step 3: Implement** — in `chat.py` add `import re` after `import math`; replace `from prepare import Tokenizer` with:

```python
from prepare import (
    DEFAULT_DATASET,
    DEFAULT_TOKENIZER,
    TOKENIZER_SOURCES,
    Tokenizer,
    _resolve_dataset_name,
    _resolve_tokenizer_name,
)
```

and insert after `_discover_checkpoints`:

```python
# ---------------------------------------------------------------------------
# Which dataset and tokenizer each model was trained with
# ---------------------------------------------------------------------------

DATASET_BLURBS = {
    "tinystories": "short children’s stories",
    "folktales": "folk and fairy tales",
}


def _commit_from_name(name: str) -> str | None:
    match = re.search(r"_([0-9a-f]{7,40})\.pt$", name)
    return match.group(1) if match else None


def _run_for_checkpoint(path: str, runs_dir: str = "runs") -> dict | None:
    """The run file saved with a checkpoint: same <timestamp>_<commit> stem."""
    run_path = Path(runs_dir) / (Path(path).stem + ".json")
    try:
        return json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pair_for_checkpoint(path: str, run: dict | None) -> tuple[str, str]:
    """(dataset, tokenizer) a checkpoint was trained with.

    From its run file when there is one. checkpoint_pre_eval.pt is the latest run, so it
    uses the active pair. Archived checkpoints without a run file predate the choice.
    """
    if run:
        tokenizer = (run.get("tokenizer") or {}).get("name") or DEFAULT_TOKENIZER
        return run.get("dataset") or DEFAULT_DATASET, tokenizer
    if Path(path).name == "checkpoint_pre_eval.pt":
        return _resolve_dataset_name(None), _resolve_tokenizer_name(None)
    return DEFAULT_DATASET, DEFAULT_TOKENIZER


def _checkpoint_vocab_rows(path: str) -> int | None:
    """Rows of the embedding table, read without loading the whole checkpoint."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        return int(state["transformer.wte.weight"].shape[0])
    except Exception:
        return None


def _results_by_commit(path: str = "results.tsv") -> dict[str, dict]:
    rows: dict[str, dict] = {}
    p = Path(path)
    if not p.exists():
        return rows
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            commit = (r.get("commit") or "").strip()
            if not commit:
                continue
            try:
                score = float((r.get("val_bpb") or "").strip())
                score = score if math.isfinite(score) and score > 0 else None
            except ValueError:
                score = None
            rows[commit] = {
                "val_bpb": score,
                "status": (r.get("status") or "").strip(),
                "description": (r.get("description") or "").strip(),
            }
    return rows


def _growth_from_run(run: dict | None) -> dict | None:
    if not run or not run.get("prompts") or not run.get("snapshots"):
        return None
    return {
        "prompts": run["prompts"],
        "snapshots": [{"t_s": s.get("t_s"), "samples": s.get("samples", [])} for s in run["snapshots"]],
    }
```

- [ ] **Step 4: Run, expect pass** — `uv run pytest -q` → `67 passed`.

- [ ] **Step 5: Commit** — `git add chat.py tests/test_chat_pairs.py && git commit -m "chat: work out each model's dataset, tokenizer, run file and scoreboard row"`

---

### Task 2: Per-pair tokenizers in ModelStore; new routes

**Files:**
- Modify: `chat.py` (`ModelStore`, `_load`, `main`, `build_app`)
- Modify: `tests/test_chat_pairs.py` (append)

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces: `ModelStore(device, entries, active_path, tokenizer_loader=Tokenizer.from_directory, runs_dir="runs", results_path="results.tsv")`; `list_models()` entries gain `commit, dataset, tokenizer, tokenizer_source, vocab_size, params_m, val_bpb, status, description, has_growth, available, reason, dataset_blurb`; `get_bundle_by_id(id)` returns that model's own tokenizer and raises `KeyError` (unknown) or `ValueError` (unavailable); `tokenizer_for_id(id)`; `growth_for_id(id)`. Routes: `GET /growth?model_id=`, `GET /vocab?model_id=`.

- [ ] **Step 1: Failing tests** — append:

```python
class _Tok:
    def __init__(self, n):
        self.n = n

    def get_vocab_size(self):
        return self.n


def _entries(tmp_path, rows):
    out = []
    for i, (name, vocab_rows) in enumerate(rows):
        path = tmp_path / name
        torch.save({"transformer.wte.weight": torch.zeros(vocab_rows, 2)}, path)
        out.append({"id": f"m{i + 1}", "path": str(path), "label": name})
    return out


def test_mismatched_vocabulary_marks_model_unavailable(tmp_path):
    entries = _entries(tmp_path, [("a_aaaaaaa.pt", 8), ("b_bbbbbbb.pt", 9)])
    store = chat.ModelStore("cpu", entries, entries[0]["path"],
                            tokenizer_loader=lambda dataset, tokenizer: _Tok(8),
                            runs_dir=str(tmp_path / "runs"), results_path=str(tmp_path / "none.tsv"))
    listed = {m["id"]: m for m in store.list_models()}
    assert listed["m1"]["available"] is True
    assert listed["m2"]["available"] is False and "vocabulary" in listed["m2"]["reason"]


def test_missing_tokenizer_marks_model_unavailable(tmp_path):
    entries = _entries(tmp_path, [("a_aaaaaaa.pt", 8)])

    def loader(dataset, tokenizer):
        raise FileNotFoundError("no tokenizer.pkl")

    store = chat.ModelStore("cpu", entries, entries[0]["path"], tokenizer_loader=loader,
                            runs_dir=str(tmp_path / "runs"), results_path=str(tmp_path / "none.tsv"))
    m = store.list_models()[0]
    assert m["available"] is False and "prepare.py" in m["reason"]


def test_facts_come_from_the_run_file_and_the_scoreboard(tmp_path):
    entries = _entries(tmp_path, [("20261007T010000-0700_abc1234.pt", 8)])
    runs = _write_run(tmp_path, "20261007T010000-0700_abc1234", final={"params_m": 18.9},
                      prompts=["Once"], snapshots=[{"t_s": 0, "samples": ["x"]}])
    tsv = tmp_path / "results.tsv"
    tsv.write_text("timestamp\tcommit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
                   "2026-10-07T01:00:00-07:00\tabc1234\t1.389\t6.6\tkeep\tfolktales baseline\n", encoding="utf-8")
    store = chat.ModelStore("cpu", entries, entries[0]["path"], tokenizer_loader=lambda dataset, tokenizer: _Tok(8),
                            runs_dir=str(runs), results_path=str(tsv))
    m = store.list_models()[0]
    assert (m["dataset"], m["tokenizer"], m["params_m"]) == ("folktales", "phi3", 18.9)
    assert (m["val_bpb"], m["description"], m["has_growth"]) == (1.389, "folktales baseline", True)
    assert store.growth_for_id("m1")["prompts"] == ["Once"]
```

- [ ] **Step 2: Run, expect failure** — `TypeError` from the old `ModelStore` signature (3 tests).

- [ ] **Step 3: Implement** — replace the whole `ModelStore` class with:

```python
class ModelStore:
    """The models on disk, each with the tokenizer of the pair it was trained with."""

    def __init__(self, device: str, entries: list[dict], active_path: str,
                 tokenizer_loader=None, runs_dir: str = "runs", results_path: str = "results.tsv"):
        self._device = device
        self._loader = tokenizer_loader or (lambda dataset, tokenizer: Tokenizer.from_directory(dataset=dataset, tokenizer=tokenizer))
        self._tokenizers: dict[tuple[str, str], object] = {}
        self._cache: dict[str, GPT] = {}
        self._lock = threading.Lock()
        self._runs: dict[str, dict | None] = {}
        scoreboard = _results_by_commit(results_path)

        self._entries = []
        for entry in entries:
            run = _run_for_checkpoint(entry["path"], runs_dir)
            dataset, tokenizer_name = _pair_for_checkpoint(entry["path"], run)
            commit = _commit_from_name(entry["label"])
            row = scoreboard.get(commit or "", {})
            final = (run or {}).get("final") or {}
            facts = {
                **entry,
                "commit": commit,
                "dataset": dataset,
                "dataset_blurb": DATASET_BLURBS.get(dataset, dataset),
                "tokenizer": tokenizer_name,
                "tokenizer_source": TOKENIZER_SOURCES.get(tokenizer_name, ""),
                "params_m": final.get("params_m"),
                "val_bpb": row.get("val_bpb", final.get("val_bpb")),
                "status": row.get("status", ""),
                "description": row.get("description", ""),
                "has_growth": _growth_from_run(run) is not None,
            }
            facts["available"], facts["reason"], facts["vocab_size"] = self._check(facts)
            self._runs[entry["id"]] = run
            self._entries.append(facts)
        self._by_id = {e["id"]: e for e in self._entries}

        normalized_active = str(Path(active_path)).replace("\\", "/")
        usable = [e for e in self._entries if e["available"]]
        if not usable:
            raise RuntimeError("No usable checkpoint found. Run 'uv run prepare.py' and 'uv run train.py' first.")
        active = next((e for e in usable if e["path"] == normalized_active), usable[-1])
        self._active_id = active["id"]

    def _tokenizer(self, dataset: str, tokenizer_name: str):
        key = (dataset, tokenizer_name)
        if key not in self._tokenizers:
            try:
                self._tokenizers[key] = self._loader(dataset, tokenizer_name)
            except (FileNotFoundError, OSError, ValueError, KeyError):
                self._tokenizers[key] = None
        return self._tokenizers[key]

    def _check(self, facts: dict) -> tuple[bool, str, int | None]:
        tok = self._tokenizer(facts["dataset"], facts["tokenizer"])
        if tok is None:
            return False, (f"its tokenizer ({facts['dataset']}, {facts['tokenizer']}) is not prepared on this machine: "
                           f"uv run prepare.py --dataset {facts['dataset']} --tokenizer {facts['tokenizer']}"), None
        size = tok.get_vocab_size()
        rows = _checkpoint_vocab_rows(facts["path"])
        if rows is not None and rows != size:
            return False, f"its vocabulary has {rows:,} tokens but the {facts['tokenizer']} tokenizer has {size:,}", size
        return True, "", size

    @property
    def device(self) -> str:
        return self._device

    @property
    def active_id(self) -> str:
        return self._active_id

    def list_models(self) -> list[dict]:
        return [{**e, "active": e["id"] == self._active_id} for e in self._entries]

    def tokenizer_for_id(self, model_id: str | None):
        entry = self._by_id.get(model_id or self._active_id) or self._by_id[self._active_id]
        if not entry["available"]:
            entry = self._by_id[self._active_id]
        return self._tokenizer(entry["dataset"], entry["tokenizer"])

    def growth_for_id(self, model_id: str) -> dict | None:
        return _growth_from_run(self._runs.get(model_id))

    def get_bundle_by_id(self, model_id: str | None):
        with self._lock:
            entry = self._by_id.get(model_id or self._active_id)
            if entry is None:
                raise KeyError(f"Unknown model id: {model_id}")
            if not entry["available"]:
                raise ValueError(f"This model can't be loaded: {entry['reason']}.")
            model = self._cache.get(entry["path"])
            if model is None:
                model = _load_model_from_checkpoint(entry["path"], self._device)
                self._cache[entry["path"]] = model
            return model, self._tokenizer(entry["dataset"], entry["tokenizer"]), self._device, entry

    def get_active_bundle(self):
        return self.get_bundle_by_id(self._active_id)
```

Replace `_load` and the start of `main()` (from `model, tokenizer, device = _load(args.checkpoint)` through `app = build_app(model_store)`) with:

```python
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    entries = _discover_checkpoints(args.checkpoint)
    try:
        model_store = ModelStore(device=device, entries=entries, active_path=args.checkpoint)
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)
    for m in model_store.list_models():
        if not m["available"]:
            print(f"Skipping {m['label']}: {m['reason']}.")
    model_store.get_active_bundle()  # load the default model before the page opens
    print("Model ready.\n")
    app = build_app(model_store)
```

and delete the old `_load` function. In `build_app`, replace the model selection in `generate` with:

```python
        try:
            model, tokenizer, device, _ = model_store.get_bundle_by_id(req.model_id)
        except KeyError:
            model, tokenizer, device, _ = model_store.get_active_bundle()
        except ValueError as exc:
            message = str(exc)
            return StreamingResponse(iter([json.dumps({"t": message, "toks": []}) + "\n"]), media_type="text/plain")
```

Replace the `/vocab` route with:

```python
    @app.get("/vocab")
    def vocab(model_id: str | None = None):
        tokenizer = model_store.tokenizer_for_id(model_id)
        n = tokenizer.get_vocab_size()
        return {"entries": [{"id": i, "text": tokenizer.decode([i])} for i in range(n)]}

    @app.get("/growth")
    def growth(model_id: str):
        return model_store.growth_for_id(model_id) or {"prompts": [], "snapshots": []}
```

- [ ] **Step 4: Run, expect pass** — `uv run pytest -q` → `70 passed`; `uv run python -c "import chat"` succeeds.

- [ ] **Step 5: Commit** — `git commit -am "chat: each model loads with its own tokenizer; mismatches are listed, not decoded"` (add the test file).

---

### Task 3: The page — pickers, facts, header, vocabulary, Watch it learn

**Files:** Modify `chat.py` (`_HTML`: CSS, header, pane labels, JS).

- [ ] **Step 1: CSS** — add before the closing `</style>` of `_HTML`:

```css
  .pane-pick { font: inherit; font-size: 0.85rem; font-weight: 600; max-width: 100%; padding: 4px 6px;
               border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--text); }
  .pane-facts { font-size: 0.78rem; color: var(--muted); margin: -4px 0 8px; }
  .growth-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }
  .growth-times { display: flex; flex-wrap: wrap; gap: 4px; }
  .growth-note { font-size: 0.78rem; color: var(--muted); margin: 8px 0 0; }
```

- [ ] **Step 2: Header** — in the header paragraph replace `trained on short children&rsquo;s\n      stories.` with `trained on <span id="dataset-blurb">short children&rsquo;s stories</span>.` and `Output appears word&nbsp;by&nbsp;word in real time.` with `Output appears token&nbsp;by&nbsp;token in real time.`

- [ ] **Step 3: Pane labels** — replace `<div class="card-label" id="label-baseline">Baseline</div>` with `<select class="pane-pick" id="pick-baseline" aria-label="Left model"></select>`, and the `label-best` div likewise with `id="pick-best"` / `aria-label="Right model"`. Directly after each pane's `card-label-row` closing `</div>`, add `<div class="pane-facts" id="facts-baseline"></div>` (and `facts-best`).

- [ ] **Step 4: Growth card** — after `</div><!-- /gen-outputs -->` add:

```html
  <div class="card" id="card-growth" style="display:none">
    <div class="card-label">Watch it learn</div>
    <div class="growth-row">
      <select class="pane-pick" id="growth-model" aria-label="Model to watch"></select>
      <select class="pane-pick" id="growth-prompt" aria-label="Prompt"></select>
      <div class="growth-times" id="growth-times" role="group" aria-label="Moment in training"></div>
    </div>
    <div id="growth-text" class="output-box"></div>
    <p class="growth-note">What this model wrote at each moment of its 5 minutes of training, saved in its run file. Every model uses the same prompts and sampling settings, so the differences come from the model.</p>
  </div>
```

- [ ] **Step 5: JavaScript** — in `loadModels()` replace the `_commitToModelId` filename regex loop with `_models.forEach(m => { if (m.commit) _commitToModelId[m.commit] = m.id; });`. In `loadResults()`, replace the four lines that set `label-baseline` / `label-best` text and the `else` branch's label line with calls to `setupPickers(_baselineModelId, _bestModelId)`; in the `else` branch call `setupPickers(null, null)`. Add these functions before `// ---- Vocabulary browser ----`:

```js
  function factsText(m) {
    const parts = [];
    if (m.val_bpb != null) parts.push('score ' + m.val_bpb.toFixed(6));
    parts.push(m.dataset === 'folktales' ? 'Folktales' : 'TinyStories');
    if (m.vocab_size) parts.push(m.tokenizer + ' tokenizer (' + m.vocab_size.toLocaleString() + ' tokens)');
    if (m.params_m) parts.push(m.params_m + ' M parameters');
    return parts.join(' \\u00b7 ');
  }
  function modelName(m) { return (m.description || m.label) + (m.commit ? ' \\u2014 ' + m.commit : ''); }
  function fillPick(sel, chosen) {
    sel.innerHTML = '';
    _models.forEach(m => {
      const o = document.createElement('option');
      o.value = m.id;
      o.textContent = modelName(m) + (m.available ? '' : ' (can\\u2019t load)');
      o.disabled = !m.available;
      o.title = m.available ? factsText(m) : m.reason;
      sel.appendChild(o);
    });
    if (chosen) sel.value = chosen;
  }
  function showFacts(which) {
    const m = _models.find(x => x.id === $('pick-' + which).value);
    $('facts-' + which).textContent = m ? factsText(m) : '';
    if (which === 'baseline' && m) { $('dataset-blurb').textContent = m.dataset_blurb; _vocabData = null; }
  }
  function setupPickers(leftId, rightId) {
    const usable = _models.filter(m => m.available);
    const fallback = (usable[usable.length - 1] || {}).id;
    fillPick($('pick-baseline'), leftId || fallback);
    fillPick($('pick-best'), rightId || fallback);
    ['baseline', 'best'].forEach(w => { $('pick-' + w).onchange = () => showFacts(w); showFacts(w); });
    setupGrowth();
  }

  // ---- Watch it learn ----
  let _growth = null;
  async function setupGrowth() {
    const withGrowth = _models.filter(m => m.has_growth && m.available);
    if (!withGrowth.length) { $('card-growth').style.display = 'none'; return; }
    $('card-growth').style.display = '';
    const sel = $('growth-model');
    sel.innerHTML = '';
    withGrowth.forEach(m => { const o = document.createElement('option'); o.value = m.id; o.textContent = modelName(m); sel.appendChild(o); });
    sel.onchange = loadGrowth;
    $('growth-prompt').onchange = () => showGrowth(0);
    await loadGrowth();
  }
  async function loadGrowth() {
    _growth = await (await fetch('/growth?model_id=' + encodeURIComponent($('growth-model').value))).json();
    const p = $('growth-prompt');
    p.innerHTML = '';
    _growth.prompts.forEach((text, i) => { const o = document.createElement('option'); o.value = i; o.textContent = text; p.appendChild(o); });
    const times = $('growth-times');
    times.innerHTML = '';
    _growth.snapshots.forEach((s, i) => {
      const b = document.createElement('button');
      b.className = 'toggle-btn';
      b.textContent = s.t_s < 60 ? Math.round(s.t_s) + ' s' : (s.t_s / 60).toFixed(s.t_s % 60 ? 1 : 0) + ' min';
      b.onclick = () => showGrowth(i);
      times.appendChild(b);
    });
    showGrowth(_growth.snapshots.length - 1);
  }
  function showGrowth(i) {
    if (!_growth || !_growth.snapshots.length) return;
    const prompt = parseInt($('growth-prompt').value || '0');
    Array.from($('growth-times').children).forEach((b, j) => {
      b.classList.toggle('active', j === i); b.setAttribute('aria-pressed', j === i ? 'true' : 'false');
    });
    $('growth-text').textContent = _growth.prompts[prompt] + (_growth.snapshots[i].samples[prompt] || '');
  }
```

In `generate()`, replace `runGenerate(prompt, _baselineModelId, 'baseline')` with `runGenerate(prompt, $('pick-baseline').value, 'baseline')` and the `best` call likewise with `$('pick-best').value`. In `loadVocab()`, replace `fetch('/vocab')` with `fetch('/vocab?model_id=' + encodeURIComponent($('pick-baseline').value))`.

- [ ] **Step 6: Verify on real models**

```bash
uv run pytest -q                              # 70 passed
uv run prepare.py --dataset folktales --tokenizer own | tail -1
uv run train.py > run.log 2>&1                # ~8 min; writes runs/<stem>.json
STEM=$(basename "$(ls runs/*.json | tail -1)" .json); cp checkpoint_pre_eval.pt "checkpoints/$STEM.pt"
uv run prepare.py --dataset tinystories --tokenizer own | tail -1
git checkout -- checkpoint_pre_eval.pt
uv run chat.py --no-browser &                 # then, against its port:
curl -s localhost:8000/models | python -c "import json,sys; [print(m['id'], m['dataset'], m['tokenizer'], m['available'], m['has_growth'], m['val_bpb']) for m in json.load(sys.stdin)['models']]"
curl -s "localhost:8000/growth?model_id=<folktales id>" | python -c "import json,sys; g=json.load(sys.stdin); print(len(g['snapshots']), g['prompts'][0])"
curl -s -X POST localhost:8000/generate -H 'Content-Type: application/json' -d '{"prompt":"The old king said","max_tokens":40,"temperature":0.8,"top_k":40,"model_id":"<folktales id>"}' | tail -1
```

Expected: May models list as `tinystories own True False`; the Folktales model as `folktales own True True` (its own 8,192-token tokenizer, built from Folktales); `/growth` returns 6 snapshots; `/generate` on the Folktales model writes in folk-tale style while the active pair is TinyStories (proof that each model uses its own tokenizer). Open the page once and look at both panes, the facts lines and Watch it learn. Then stop the server and delete the Folktales checkpoint and run file (verification artifacts, not commits).

- [ ] **Step 7: Commit** — `git commit -am "chat: pick any two models, label them with their facts, watch a model learn"`

---

### Task 4: Port to the starter kit

The starter's `chat.py` differs from the worked example's in a few lines. Apply the same Task 1–3 changes, copy `tests/test_chat_pairs.py`, run `uv run pytest -q` (expect 70 passed) and `uv run chat.py --no-browser` against the default model, then commit on branch `feat/chat-pairs`.
