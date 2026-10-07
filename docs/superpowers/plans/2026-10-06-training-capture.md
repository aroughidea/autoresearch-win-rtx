# Training Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every training run records what the model writes at 0, 10, 30, 60 and 120 seconds and at the end, on fixed prompts, and saves one small JSON run file per experiment, without changing training or the score.

**Architecture:** A new `capture.py` (off-limits to the agent, like `prepare.py`) holds all logic: a deterministic sampler with its own random generator, a `RunCapture` object that takes snapshots on a training-time schedule, and an atomic JSON writer. `train.py` gains only hook calls: one before each step's timer starts, one at the end of training, one after the score. `program.md` tells the agent to commit the run file with its scoreboard row. Built in the worked example first, then ported to the starter kit.

**Tech Stack:** Python 3.10+ (project pins signed CPython 3.14 on Windows), PyTorch 2.9.1, pytest (new, dev-only), uv, git.

**Spec:** `docs/superpowers/specs/2026-10-06-training-decisions-journey-design.md`, component 1 "Capture", the run-file section, and "Check results".

## Global Constraints

- Training must be unchanged: the 5-minute budget (`TIME_BUDGET = 300` in `prepare.py`) and the score. Capture runs only before `t0 = time.time()` in the loop or after the loop, and samples with its own `torch.Generator`, never the global RNG.
- Snapshot times: `0, 10, 30, 60, 120` seconds of training time, plus a final snapshot at the end of training.
- Prompts, exactly: `"Once upon a time"`, `"The old king said"`, `"In the dark forest"`, `"The little girl found a"`.
- Decoding, exactly: `seed 1234` (prompt *i* uses `1234 + i`), `temperature 0.8`, `top_k 40`, `min_chars 320`, `max_tokens 300`.
- Run file path: `runs/<YYYYMMDDTHHMMSS-ZZZZ>_<commit>.json`, timestamp from the commit time, colon-free (colons break Windows filenames). No git: `_nogit` with the current time.
- Keep/discard status and the agent's description are NOT in the run file; they stay in `results.tsv`, joined on the commit.
- All file I/O is explicit UTF-8 (`encoding="utf-8"`); Windows defaults to cp1252.
- Capture must never raise into training. Any sampling error disables capture for that run and is recorded as `capture_error` in the run file.
- No new runtime dependencies (`program.md` forbids the agent from adding any). pytest is a dev dependency only.
- `checkpoint_pre_eval.pt` is tracked in git (LFS). Every `train.py` run overwrites it; restore it with `git checkout -- checkpoint_pre_eval.pt` after any verification run.
- Before any GPU step: `nvidia-smi --query-gpu=memory.used --format=csv,noheader` must show under 3,000 MiB used. Chrome and `KinectService` have held 8 GB on this machine before; ask TJ to close them rather than closing them yourself.
- VS Code exports `VIRTUAL_ENV` pointing at this repo's `.venv`. In any other directory (a worktree, the starter clone), run `unset VIRTUAL_ENV` first, or `uv` installs into the wrong environment.
- Commits go on branch `feat/training-capture`. Do not push or open a PR without TJ's go-ahead.

## Review Focus

1. **A capture failure mid-run** (sampling raises, odd tokenizer output): training must finish and print `val_bpb` as normal, with the error noted in the run file. Test in Task 3 (`test_sampling_error_disables_capture_but_never_raises`).
2. **An OOM retry** (`main()` retries `_run_training_once` with a smaller batch): snapshots from the failed attempt must not leak into the run file. Test in Task 3 (`test_begin_attempt_resets_snapshots`).
3. **No git** (starter downloaded as a ZIP, a copied Colab folder): the run file is still written, named `_nogit`. Tests in Task 1 (`test_git_commit_info_outside_a_repo`) and Task 3 (`test_finish_without_git_uses_nogit`).
4. **Non-ASCII and broken characters in samples** (untrained models emit Chinese, Cyrillic and partial bytes, as the check showed): the file must write and read back as UTF-8 on Windows. Test in Task 3 (`test_unicode_samples_round_trip`).
5. **NaN or infinite loss or score**: the run file must still be valid JSON, with `null`. Test in Task 3 (`test_nan_values_become_null`).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `capture.py` | Create | All capture logic: timestamps and run ids, git lookup, sampler, `RunCapture`, `recipe_from`, `runs_dir_from_env`. Agent may not modify. |
| `tests/test_capture.py` | Create | CPU-only unit tests with a fake model and tokenizer. |
| `runs/.gitkeep` | Create | Keeps `runs/` present so the agent's `git add results.tsv runs/` never fails. |
| `pyproject.toml` | Modify | pytest dev dependency; pytest finds `capture.py` at the repo root. |
| `train.py` | Modify | Import plus four hook sites (`begin_attempt`, `on_step`, `on_train_end`, `finish`). |
| `program.md` | Modify | `capture.py` read-only; keep the hooks; commit the run file in step 9. |
| `README.md` | Modify | Project structure lists `capture.py` and `runs/`. |

---

### Task 1: Branch, test runner, and run-id helpers

**Files:**
- Create: `capture.py`
- Create: `tests/test_capture.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `capture.compact_timestamp(moment: datetime) -> str`, `capture.make_run_id(commit: str | None, moment: datetime) -> str`, `capture.git_commit_info(cwd=None) -> tuple[str | None, datetime | None]`.

- [ ] **Step 1: Create the branch and commit the design docs**

```bash
cd /c/Users/tj/repos/autoresearch-win-rtx
git switch -c feat/training-capture
git add docs/superpowers/specs/2026-10-06-training-decisions-journey-design.md docs/superpowers/plans/2026-10-06-training-capture.md
git commit -m "docs: Training Decisions design spec and capture plan"
```

- [ ] **Step 2: Add pytest as a dev dependency and point it at the repo root**

```bash
uv add --dev pytest
```

Then append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Run: `uv run pytest --version`
Expected: `pytest 8.x` or newer.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_capture.py`:

```python
import re
import subprocess
from datetime import datetime, timedelta, timezone

import capture


PDT = timezone(timedelta(hours=-7))


def test_compact_timestamp_negative_offset():
    moment = datetime(2026, 5, 23, 15, 57, 43, tzinfo=PDT)
    assert capture.compact_timestamp(moment) == "20260523T155743-0700"


def test_compact_timestamp_positive_half_hour_offset():
    moment = datetime(2026, 5, 23, 15, 57, 43, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert capture.compact_timestamp(moment) == "20260523T155743+0530"


def test_compact_timestamp_naive_gets_local_offset():
    stamp = capture.compact_timestamp(datetime(2026, 5, 23, 15, 57, 43))
    assert re.fullmatch(r"\d{8}T\d{6}[+-]\d{4}", stamp)


def test_make_run_id_with_commit():
    moment = datetime(2026, 5, 23, 15, 57, 43, tzinfo=PDT)
    assert capture.make_run_id("75027e8", moment) == "20260523T155743-0700_75027e8"


def test_make_run_id_without_git_is_colon_free():
    run_id = capture.make_run_id(None, datetime(2026, 5, 23, 15, 57, 43, tzinfo=PDT))
    assert run_id.endswith("_nogit")
    assert ":" not in run_id


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_git_commit_info_in_a_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    expected = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()

    commit, committed_at = capture.git_commit_info(cwd=tmp_path)

    assert commit == expected
    assert committed_at.tzinfo is not None


def test_git_commit_info_outside_a_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert capture.git_commit_info(cwd=tmp_path) == (None, None)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_capture.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'capture'`.

- [ ] **Step 5: Write the helpers**

Create `capture.py`:

```python
"""
Run capture for the Training Decisions explorer.

Records what the model writes at fixed moments of training, on fixed prompts with
fixed decoding, and saves one small JSON run file per experiment in runs/.

A captured run trains exactly like an uncaptured one: train.py calls this module
before each step's timer starts (so sampling never counts toward the 5-minute
budget), and sampling uses its own random generator, never the global one.

The agent must not modify this file (see program.md).
"""

import contextlib
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch


def compact_timestamp(moment):
    """2026-05-23 15:57:43 -07:00 -> '20260523T155743-0700'. Colon-free, safe in Windows filenames."""
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.strftime("%Y%m%dT%H%M%S%z")


def make_run_id(commit, moment):
    return f"{compact_timestamp(moment)}_{commit or 'nogit'}"


def git_commit_info(cwd=None):
    """Short hash and commit time of HEAD, or (None, None) outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h%n%cI"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or len(lines) != 2:
        return None, None
    try:
        return lines[0], datetime.fromisoformat(lines[1])
    except ValueError:
        return lines[0], None
```

(`contextlib`, `json`, `math`, `os`, `time`, `Path` and `torch` are used from Task 2 on. Keep the imports now so later tasks only add code.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_capture.py -v`
Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
git add capture.py tests/test_capture.py pyproject.toml uv.lock
git commit -m "capture: run-id and git helpers, pytest setup"
```

---

### Task 2: Deterministic sampler

**Files:**
- Modify: `capture.py` (append)
- Modify: `tests/test_capture.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `capture.sample_continuation(model, tokenizer, prompt: str, *, seed: int, temperature: float, top_k: int, min_chars: int, max_tokens: int, device, autocast_ctx=None) -> str`. `model(idx)` must return logits `(1, T, vocab)` and expose `model.config.sequence_len`. `tokenizer` must provide `get_bos_token_id()`, `get_eos_token_id()`, `get_vocab_size()`, `encode(str) -> list[int]`, `decode(list[int]) -> str`. These match `train.GPT` and `prepare.Tokenizer`.
- Also produces, for Task 3's tests: `FakeTokenizer` and `FakeModel` in `tests/test_capture.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capture.py`:

```python
from types import SimpleNamespace

import torch


class FakeTokenizer:
    """Byte-level: ids 0-255 are bytes, 256 is BOS, 257 is EOS."""

    BOS, EOS = 256, 257

    def get_bos_token_id(self):
        return self.BOS

    def get_eos_token_id(self):
        return self.EOS

    def get_vocab_size(self):
        return 258

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class FakeModel(torch.nn.Module):
    """Next-token logits depend only on the last token, from a fixed random table."""

    def __init__(self, seq_len=64, eos_bias=0.0, fail=False):
        super().__init__()
        weights = torch.randn(258, 258, generator=torch.Generator().manual_seed(0))
        self.table = torch.nn.Embedding.from_pretrained(weights)
        self.config = SimpleNamespace(sequence_len=seq_len)
        self.eos_bias = eos_bias
        self.fail = fail
        self.calls = 0
        self.max_len_seen = 0

    def forward(self, idx):
        self.calls += 1
        self.max_len_seen = max(self.max_len_seen, idx.size(1))
        if self.fail:
            raise RuntimeError("boom")
        logits = self.table(idx).clone()
        logits[..., FakeTokenizer.EOS] += self.eos_bias
        return logits


def _sample(model, prompt="Once upon a time", **overrides):
    kwargs = dict(seed=1234, temperature=0.8, top_k=40, min_chars=40, max_tokens=60, device="cpu")
    kwargs.update(overrides)
    return capture.sample_continuation(model, FakeTokenizer(), prompt, **kwargs)


def test_same_seed_same_text():
    model = FakeModel()
    assert _sample(model) == _sample(model)


def test_different_seed_different_text():
    model = FakeModel()
    assert _sample(model, seed=1) != _sample(model, seed=2)


def test_global_rng_untouched():
    torch.manual_seed(7)
    before = torch.get_rng_state()
    _sample(FakeModel())
    assert torch.equal(before, torch.get_rng_state())


def test_training_mode_restored():
    model = FakeModel()
    model.train()
    _sample(model)
    assert model.training is True


def test_stops_at_eos():
    assert _sample(FakeModel(eos_bias=1e4)) == ""


def test_stops_once_min_chars_reached():
    text = _sample(FakeModel(), min_chars=10, max_tokens=300)
    assert 10 <= len(text) <= 12


def test_respects_max_tokens():
    text = _sample(FakeModel(), min_chars=10_000, max_tokens=5)
    assert len(text) <= 5  # each generated byte decodes to at most one character


def test_crops_context_to_sequence_len():
    model = FakeModel(seq_len=8)
    _sample(model, prompt="a prompt much longer than eight tokens")
    assert model.max_len_seen <= 8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_capture.py -v`
Expected: the 8 new tests FAIL with `AttributeError: module 'capture' has no attribute 'sample_continuation'`; the 7 from Task 1 still pass.

- [ ] **Step 3: Write the sampler**

Append to `capture.py`:

```python
@torch.no_grad()
def sample_continuation(model, tokenizer, prompt, *, seed, temperature, top_k, min_chars,
                        max_tokens, device, autocast_ctx=None):
    """Continue `prompt` until EOS, `min_chars` characters, or `max_tokens` tokens.

    Uses a private torch.Generator so the global RNG (and so training) is untouched.
    Returns only the continuation, not the prompt.
    """
    was_training = model.training
    model.eval()
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        ids = [tokenizer.get_bos_token_id()] + list(tokenizer.encode(prompt))
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        start = idx.size(1)
        eos = tokenizer.get_eos_token_id()
        window = model.config.sequence_len
        ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
        text = ""
        for _ in range(max_tokens):
            with ctx:
                logits = model(idx[:, -window:])
            logits = logits[:, -1, :].float() / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
            if next_id.item() == eos:
                break
            idx = torch.cat((idx, next_id), dim=1)
            text = tokenizer.decode(idx[0, start:].tolist())
            if len(text) >= min_chars:
                break
        return text
    finally:
        model.train(was_training)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_capture.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add capture.py tests/test_capture.py
git commit -m "capture: deterministic sampler with its own random generator"
```

---

### Task 3: RunCapture, recipe and run-file writer

**Files:**
- Modify: `capture.py` (append)
- Modify: `tests/test_capture.py` (append)

**Interfaces:**
- Consumes: `make_run_id`, `git_commit_info` (Task 1); `sample_continuation` (Task 2); `FakeTokenizer`, `FakeModel` (Task 2 tests).
- Produces, used by Task 4 in `train.py`:
  - `RunCapture(tokenizer, *, dataset: str, runs_dir, device, tokenizer_name="own", snapshot_times=SNAPSHOT_TIMES, prompts=PROMPTS, decoding=None, log=print)`
  - `RunCapture.begin_attempt(autocast_ctx=None) -> None`
  - `RunCapture.on_step(model, training_seconds: float, step: int, train_loss: float | None) -> None`
  - `RunCapture.on_train_end(model, training_seconds: float, step: int, train_loss: float | None) -> None`
  - `RunCapture.finish(*, val_bpb, peak_vram_mb, training_seconds, num_steps, num_params, recipe: dict, commit=None, committed_at=None) -> Path | None`
  - `recipe_from(namespace: dict, config=None) -> dict`
  - `runs_dir_from_env(smoke_test: bool, env=None) -> str | None`
  - Constants `SCHEMA = 1`, `SNAPSHOT_TIMES`, `PROMPTS`, `DECODING`, `RECIPE_KEYS`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capture.py`:

```python
import json


FAST = {"seed": 1234, "temperature": 0.8, "top_k": 40, "min_chars": 5, "max_tokens": 8}
WHEN = datetime(2026, 5, 23, 15, 57, 43, tzinfo=PDT)


def _capture(tmp_path, runs_dir="use_tmp", **overrides):
    kwargs = dict(
        dataset="tinystories",
        runs_dir=tmp_path if runs_dir == "use_tmp" else runs_dir,
        device="cpu",
        snapshot_times=(0, 10, 30),
        decoding=FAST,
        log=lambda message: None,
    )
    kwargs.update(overrides)
    return capture.RunCapture(FakeTokenizer(), **kwargs)


def _finish(cap, **overrides):
    kwargs = dict(
        val_bpb=0.520082, peak_vram_mb=6799.2, training_seconds=300.4, num_steps=641,
        num_params=18_900_000, recipe={"DEPTH": 6}, commit="abc1234", committed_at=WHEN,
    )
    kwargs.update(overrides)
    return cap.finish(**kwargs)


def test_snapshots_follow_training_time(tmp_path):
    cap, model = _capture(tmp_path), FakeModel()
    cap.on_step(model, 0.0, 0, None)
    cap.on_step(model, 5.0, 3, 4.2)
    assert [s["t_s"] for s in cap.snapshots] == [0.0]
    cap.on_step(model, 31.0, 9, 3.1)
    assert [s["t_s"] for s in cap.snapshots] == [0.0, 10.0, 30.0]
    assert [s["step"] for s in cap.snapshots] == [0, 9, 9]
    assert all(len(s["samples"]) == 4 for s in cap.snapshots)


def test_on_train_end_adds_final_snapshot(tmp_path):
    cap, model = _capture(tmp_path), FakeModel()
    cap.on_step(model, 0.0, 0, None)
    cap.on_train_end(model, 300.44, 641, 1.497)
    final = cap.snapshots[-1]
    assert final["t_s"] == 300.4 and final["final"] is True and final["train_loss"] == 1.497


def test_begin_attempt_resets_snapshots(tmp_path):
    cap, model = _capture(tmp_path), FakeModel()
    cap.on_step(model, 31.0, 9, 3.1)
    cap.begin_attempt()
    assert cap.snapshots == []
    cap.on_step(model, 0.0, 0, None)
    assert [s["t_s"] for s in cap.snapshots] == [0.0]


def test_disabled_capture_does_nothing(tmp_path):
    cap, model = _capture(tmp_path, runs_dir=None), FakeModel()
    cap.on_step(model, 31.0, 9, 3.1)
    cap.on_train_end(model, 300.0, 641, 1.5)
    assert model.calls == 0
    assert _finish(cap) is None
    assert list(tmp_path.iterdir()) == []


def test_sampling_error_disables_capture_but_never_raises(tmp_path):
    cap, model = _capture(tmp_path), FakeModel(fail=True)
    cap.on_step(model, 0.0, 0, None)
    assert cap.error.startswith("RuntimeError")
    calls = model.calls
    cap.on_step(model, 31.0, 9, 3.1)
    assert model.calls == calls
    record = json.loads(_finish(cap).read_text(encoding="utf-8"))
    assert record["capture_error"].startswith("RuntimeError")
    assert record["final"]["val_bpb"] == 0.520082


def test_finish_writes_named_run_file(tmp_path):
    cap, model = _capture(tmp_path), FakeModel()
    cap.on_step(model, 0.0, 0, None)
    cap.on_train_end(model, 300.4, 641, 1.497)
    path = _finish(cap)
    assert path.name == "20260523T155743-0700_abc1234.json"
    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema"] == 1
    assert record["run_id"] == "20260523T155743-0700_abc1234"
    assert record["commit"] == "abc1234"
    assert record["dataset"] == "tinystories"
    assert record["tokenizer"] == {"name": "own", "vocab_size": 258}
    assert record["prompts"] == list(capture.PROMPTS)
    assert record["decoding"] == FAST
    assert record["recipe"] == {"DEPTH": 6}
    assert [s["t_s"] for s in record["snapshots"]] == [0.0, 300.4]
    assert record["final"]["val_bpb"] == 0.520082
    assert record["final"]["params_m"] == 18.9
    assert record["final"]["num_steps"] == 641
    assert "sampling_s" in record["final"]
    assert "status" not in record and "description" not in record


def test_finish_overwrites_same_run_id(tmp_path):
    cap = _capture(tmp_path)
    _finish(cap, val_bpb=0.6)
    path = _finish(cap, val_bpb=0.5)
    assert json.loads(path.read_text(encoding="utf-8"))["final"]["val_bpb"] == 0.5
    assert len(list(tmp_path.iterdir())) == 1


def test_finish_without_git_uses_nogit(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "git_commit_info", lambda cwd=None: (None, None))
    path = _finish(_capture(tmp_path), commit=None, committed_at=None)
    assert path.name.endswith("_nogit.json")


def test_unicode_samples_round_trip(tmp_path):
    cap, model = _capture(tmp_path), FakeModel()
    cap.on_step(model, 0.0, 0, None)
    cap.snapshots[0]["samples"][0] = "Wasser система 然 \ufffd"
    record = json.loads(_finish(cap).read_text(encoding="utf-8"))
    assert record["snapshots"][0]["samples"][0] == "Wasser система 然 \ufffd"


def test_nan_values_become_null(tmp_path):
    cap, model = _capture(tmp_path), FakeModel()
    cap.on_step(model, 0.0, 0, float("nan"))
    record = json.loads(_finish(cap, val_bpb=float("inf")).read_text(encoding="utf-8"))
    assert record["snapshots"][0]["train_loss"] is None
    assert record["final"]["val_bpb"] is None


def test_recipe_from_keeps_known_keys_and_config():
    namespace = {"DEPTH": 6, "MATRIX_LR": 0.045, "ADAM_BETAS": (0.8, 0.95), "UNRELATED": 1}
    config = SimpleNamespace(n_layer=6, n_embd=384, n_head=3, n_kv_head=1, vocab_size=8192, sequence_len=2048)
    recipe = capture.recipe_from(namespace, config)
    assert recipe["DEPTH"] == 6 and recipe["MATRIX_LR"] == 0.045
    assert recipe["ADAM_BETAS"] == [0.8, 0.95]
    assert recipe["n_embd"] == 384 and recipe["vocab_size"] == 8192
    assert "UNRELATED" not in recipe and "WARMDOWN_RATIO" not in recipe


def test_runs_dir_from_env():
    assert capture.runs_dir_from_env(False, env={}) == "runs"
    assert capture.runs_dir_from_env(True, env={}) is None
    assert capture.runs_dir_from_env(False, env={"AUTORESEARCH_RUNS_DIR": "off"}) is None
    assert capture.runs_dir_from_env(True, env={"AUTORESEARCH_RUNS_DIR": "D:/runs"}) == "D:/runs"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_capture.py -v`
Expected: the 12 new tests FAIL with `AttributeError: module 'capture' has no attribute 'RunCapture'` (or `recipe_from` / `runs_dir_from_env`); the 15 earlier tests pass.

- [ ] **Step 3: Write RunCapture and helpers**

Append to `capture.py`:

```python
SCHEMA = 1
SNAPSHOT_TIMES = (0, 10, 30, 60, 120)
PROMPTS = ("Once upon a time", "The old king said", "In the dark forest", "The little girl found a")
DECODING = {"seed": 1234, "temperature": 0.8, "top_k": 40, "min_chars": 320, "max_tokens": 300}
RECIPE_KEYS = (
    "DEPTH", "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN", "N_KV_HEAD", "TOTAL_BATCH_SIZE",
    "EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR", "WEIGHT_DECAY", "ADAM_BETAS",
    "WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC",
)
CONFIG_KEYS = ("n_layer", "n_embd", "n_head", "n_kv_head", "vocab_size", "sequence_len")


def _finite(value, digits=4):
    """A rounded float, or None for missing, NaN or infinite values (JSON has no NaN)."""
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _jsonable(value):
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return _finite(value, 8)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def recipe_from(namespace, config=None):
    """The recipe as far as it can be read. Keys the agent renamed or removed are simply absent."""
    recipe = {k: _jsonable(namespace[k]) for k in RECIPE_KEYS if k in namespace}
    if config is not None:
        recipe.update({k: _jsonable(getattr(config, k)) for k in CONFIG_KEYS if hasattr(config, k)})
    return recipe


def runs_dir_from_env(smoke_test, env=None):
    """AUTORESEARCH_RUNS_DIR wins ('off' disables). Otherwise runs/ for real runs, nothing for smoke tests."""
    env = os.environ if env is None else env
    value = env.get("AUTORESEARCH_RUNS_DIR", "").strip()
    if value.lower() == "off":
        return None
    if value:
        return value
    return None if smoke_test else "runs"


class RunCapture:
    """Takes writing snapshots on a training-time schedule and writes one run file."""

    def __init__(self, tokenizer, *, dataset, runs_dir, device, tokenizer_name="own",
                 snapshot_times=SNAPSHOT_TIMES, prompts=PROMPTS, decoding=None, log=print):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.runs_dir = Path(runs_dir) if runs_dir is not None else None
        self.device = device
        self.tokenizer_name = tokenizer_name
        self.snapshot_times = tuple(sorted(float(t) for t in snapshot_times))
        self.prompts = tuple(prompts)
        self.decoding = dict(DECODING if decoding is None else decoding)
        self.log = log
        self.error = None
        self.begin_attempt()

    @property
    def enabled(self):
        return self.runs_dir is not None and self.error is None

    def begin_attempt(self, autocast_ctx=None):
        """Start (or restart, after an OOM retry) a training attempt."""
        self.autocast_ctx = autocast_ctx
        self.pending = list(self.snapshot_times)
        self.snapshots = []
        self.sampling_seconds = 0.0

    def on_step(self, model, training_seconds, step, train_loss):
        if not self.enabled:
            return
        while self.enabled and self.pending and training_seconds >= self.pending[0]:
            self._take(model, self.pending.pop(0), step, train_loss)

    def on_train_end(self, model, training_seconds, step, train_loss):
        self.pending = []
        if self.enabled:
            self._take(model, training_seconds, step, train_loss, final=True)

    def _take(self, model, t_s, step, train_loss, final=False):
        started = time.time()
        d = self.decoding
        try:
            samples = [
                sample_continuation(
                    model, self.tokenizer, prompt,
                    seed=d["seed"] + i, temperature=d["temperature"], top_k=d["top_k"],
                    min_chars=d["min_chars"], max_tokens=d["max_tokens"],
                    device=self.device, autocast_ctx=self.autocast_ctx,
                )
                for i, prompt in enumerate(self.prompts)
            ]
        except Exception as exc:  # capture must never break training
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(f"capture: disabled for this run after an error: {self.error}")
            return
        took = time.time() - started
        self.sampling_seconds += took
        entry = {"t_s": round(float(t_s), 1), "step": int(step), "train_loss": _finite(train_loss), "samples": samples}
        if final:
            entry["final"] = True
        self.snapshots.append(entry)
        self.log(f"capture: t={entry['t_s']}s step={step} sampled in {took:.1f}s")

    def finish(self, *, val_bpb, peak_vram_mb, training_seconds, num_steps, num_params, recipe,
               commit=None, committed_at=None):
        """Write runs/<run_id>.json. Returns its path, or None when capture is off or the write failed."""
        if self.runs_dir is None:
            return None
        if commit is None and committed_at is None:
            commit, committed_at = git_commit_info()
        run_id = make_run_id(commit, committed_at or datetime.now().astimezone())
        record = {
            "schema": SCHEMA,
            "run_id": run_id,
            "commit": commit,
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": self.dataset,
            "tokenizer": {"name": self.tokenizer_name, "vocab_size": int(self.tokenizer.get_vocab_size())},
            "recipe": recipe,
            "decoding": self.decoding,
            "prompts": list(self.prompts),
            "snapshot_times": list(self.snapshot_times),
            "snapshots": self.snapshots,
            "final": {
                "val_bpb": _finite(val_bpb, 6),
                "training_s": _finite(training_seconds, 1),
                "peak_vram_mb": _finite(peak_vram_mb, 1),
                "num_steps": int(num_steps),
                "params_m": round(num_params / 1e6, 1),
                "sampling_s": round(self.sampling_seconds, 1),
            },
        }
        if self.error:
            record["capture_error"] = self.error
        path = self.runs_dir / f"{run_id}.json"
        tmp = path.with_name(path.name + ".tmp")
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            self.log(f"capture: could not write {path}: {exc}")
            return None
        self.log(f"capture: wrote {path}")
        return path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_capture.py -v`
Expected: 27 passed.

- [ ] **Step 5: Commit**

```bash
git add capture.py tests/test_capture.py
git commit -m "capture: RunCapture schedule, recipe and atomic run-file writer"
```

---

### Task 4: Hook capture into train.py

**Files:**
- Modify: `train.py` (import after the `from prepare import (...)` block; `_run_training_once` at the `def` line, after `autocast_ctx = ...`, around the loop's first lines, after `debiased_smooth_loss`, before `return {`; `main()` after the tokenizer prints, in the `_run_training_once(...)` call, before `return 0`)

**Interfaces:**
- Consumes: `RunCapture`, `recipe_from`, `runs_dir_from_env` from Task 3.
- Produces: real runs write `runs/<run_id>.json`; smoke tests write nothing unless `AUTORESEARCH_RUNS_DIR` is set; `AUTORESEARCH_RUNS_DIR=off` disables capture.

- [ ] **Step 1: Import**

Replace:

```python
    evaluate_bpb,
    make_dataloader,
)
```

with:

```python
    evaluate_bpb,
    make_dataloader,
)
from capture import RunCapture, recipe_from, runs_dir_from_env
```

- [ ] **Step 2: Thread `capture` through `_run_training_once`**

Replace `def _run_training_once(runtime, tokenizer, config, device_batch_size, smoke_test):` with:

```python
def _run_training_once(runtime, tokenizer, config, device_batch_size, smoke_test, capture):
```

Replace:

```python
    autocast_ctx = torch.amp.autocast(device_type=runtime.device_type, dtype=runtime.amp_dtype)
```

(the first occurrence, inside `_run_training_once`) with:

```python
    autocast_ctx = torch.amp.autocast(device_type=runtime.device_type, dtype=runtime.amp_dtype)
    capture.begin_attempt(autocast_ctx)
```

Check with `grep -n "capture.begin_attempt" train.py` that it landed inside `_run_training_once` (line number just after `def _run_training_once`), not in `_benchmark_train_candidate`.

- [ ] **Step 3: The step hook, before the timer**

Replace:

```python
    total_training_time = 0.0
    step = 0

    while True:
        torch.cuda.synchronize()
        t0 = time.time()
```

with:

```python
    total_training_time = 0.0
    step = 0
    last_loss = None

    while True:
        # Capture hook: runs before t0, so sampling never counts toward the time budget.
        capture.on_step(model, total_training_time, step, last_loss)
        torch.cuda.synchronize()
        t0 = time.time()
```

Replace:

```python
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
```

with:

```python
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        last_loss = debiased_smooth_loss
```

- [ ] **Step 4: The end-of-training hook**

Replace:

```python
    print()
    return {
        "model": model,
```

with:

```python
    print()
    capture.on_train_end(model, total_training_time, step, last_loss)
    return {
        "model": model,
```

- [ ] **Step 5: Create the capture in `main()` and pass it in**

Replace:

```python
    print(f"Dataset: {tokenizer.dataset}")
```

(the one in `main()`) with:

```python
    print(f"Dataset: {tokenizer.dataset}")
    capture = RunCapture(
        tokenizer,
        dataset=tokenizer.dataset,
        runs_dir=runs_dir_from_env(args.smoke_test),
        device=runtime.device,
    )
```

Replace:

```python
                device_batch_size=train_batch_size,
                smoke_test=args.smoke_test,
            )
```

with:

```python
                device_batch_size=train_batch_size,
                smoke_test=args.smoke_test,
                capture=capture,
            )
```

- [ ] **Step 6: Write the run file after the score**

Replace:

```python
    if args.smoke_test:
        print("smoke_test:       true")
    return 0
```

with:

```python
    if args.smoke_test:
        print("smoke_test:       true")
    capture.finish(
        val_bpb=val_bpb,
        peak_vram_mb=peak_vram_mb,
        training_seconds=total_training_time,
        num_steps=step,
        num_params=num_params,
        recipe=recipe_from(globals(), config),
    )
    return 0
```

- [ ] **Step 7: Unit tests still pass and train.py still imports**

Run: `uv run pytest -q && uv run python -c "import train; print('import ok')"`
Expected: `27 passed` and `import ok`.

- [ ] **Step 8: GPU smoke test writes a run file**

Check the GPU first (Global Constraints). Then:

```bash
SMOKE_DIR="$(cygpath -w "$(mktemp -d)")"   # Windows path: Python cannot read /tmp/...
AUTORESEARCH_RUNS_DIR="$SMOKE_DIR" uv run train.py --smoke-test > smoke.log 2>&1; echo "exit $?"
tr '\r' '\n' < smoke.log | grep -E "^capture:|^val_bpb:"
uv run python -c "
import json, pathlib, sys
files = list(pathlib.Path(sys.argv[1]).glob('*.json'))
assert len(files) == 1, files
r = json.loads(files[0].read_text(encoding='utf-8'))
print(files[0].name, [s['t_s'] for s in r['snapshots']], r['final'])
assert [len(s['samples']) for s in r['snapshots']] == [4, 4]
assert all(isinstance(x, str) for s in r['snapshots'] for x in s['samples'])
" "$SMOKE_DIR"
git checkout -- checkpoint_pre_eval.pt
```

Expected: `exit 0`; two `capture: t=...` lines (t=0.0 and the final one) and one `capture: wrote ...`; the check prints a name like `20261007T...-0700_<hash>.json` with two snapshots. (A smoke test counts no training time: its 3 steps sit inside the 11 warm-up steps the clock skips. So it has only the 0 s and final snapshots.)

- [ ] **Step 9: Smoke test without the variable writes nothing**

```bash
uv run train.py --smoke-test > smoke.log 2>&1; echo "exit $?"
tr '\r' '\n' < smoke.log | grep -c "^capture:"
ls runs/
git checkout -- checkpoint_pre_eval.pt
rm -f smoke.log
```

Expected: `exit 0`; `0` capture lines; `runs/` does not exist yet (or holds only `.gitkeep` once Task 5 is done).

- [ ] **Step 10: Commit**

```bash
git status --short   # expect only train.py modified
git add train.py
git commit -m "train: capture writing snapshots and a run file per experiment"
```

---

### Task 5: Agent instructions and docs

**Files:**
- Create: `runs/.gitkeep`
- Modify: `program.md` (setup step 3 file list; "What you CANNOT do"; loop step 9)
- Modify: `README.md` ("Project structure" block)

**Interfaces:**
- Consumes: the hook names from Task 4 (`capture.begin_attempt`, `capture.on_step`, `capture.on_train_end`, `capture.finish`).
- Produces: an agent loop that commits `runs/<run_id>.json` with each scoreboard row.

- [ ] **Step 1: Keep `runs/` in git**

```bash
mkdir -p runs && touch runs/.gitkeep
```

- [ ] **Step 2: program.md, in-scope files**

Replace:

```markdown
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
```

with:

```markdown
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `capture.py` — records what the model writes during training and saves one run file per experiment in `runs/`. Do not modify.
```

- [ ] **Step 3: program.md, what the agent cannot do**

Replace:

```markdown
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
```

with:

```markdown
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Modify `capture.py`, or remove or change the `capture.` calls in `train.py` (`capture.begin_attempt`, `capture.on_step`, `capture.on_train_end`, `capture.finish`) and the `RunCapture(...)` setup in `main()`. They record each run for the Training Decisions explorer. They run outside the timed part of the loop and do not affect training or val_bpb.
```

- [ ] **Step 4: program.md, loop step 9**

Replace:

```markdown
9. Record the result in results.tsv, then commit it: `git add results.tsv && git commit -m "log: <description> <status>"`. Do this AFTER any git reset so the TSV update is never undone.
```

with:

```markdown
9. Record the result in results.tsv, then commit it together with the run file: `git add results.tsv runs/ && git commit -m "log: <description> <status>"`. Do this AFTER any git reset so neither is undone. (`train.py` writes the run file to `runs/<timestamp>_<commit>.json`. It stays untracked until this step, so the reset in step 8 does not remove it.)
```

- [ ] **Step 5: README project structure**

Replace:

```
prepare.py        — constants, data prep + runtime utilities (do not modify)
train.py          — model, optimizer, training loop (agent modifies this)
```

with:

```
prepare.py        — constants, data prep + runtime utilities (do not modify)
capture.py        — records what the model writes during training (do not modify)
train.py          — model, optimizer, training loop (agent modifies this)
```

and replace:

```
results.tsv       — experiment log written by the agent (one row per run)
```

with:

```
results.tsv       — experiment log written by the agent (one row per run)
runs/             — one JSON run file per experiment: writing samples over training time
```

- [ ] **Step 6: Check the edits**

Run: `grep -n "capture" program.md README.md && git diff --stat`
Expected: the three `program.md` edits and two `README.md` lines appear; diff touches only `program.md`, `README.md`, `runs/.gitkeep`.

- [ ] **Step 7: Commit**

```bash
git add runs/.gitkeep program.md README.md
git commit -m "program: capture.py is read-only; commit run files with the scoreboard"
```

---

### Task 6: Full five-minute verification

**Files:** none changed. This task proves the Global Constraints on real hardware.

**Interfaces:**
- Consumes: everything above.
- Produces: evidence for the PR description: score match, snapshot count, capture overhead.

- [ ] **Step 1: Check the GPU, then run one real experiment**

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # must be under 3000 MiB
uv run train.py > run.log 2>&1; echo "exit $?"
tr '\r' '\n' < run.log | grep -E "^capture:|^val_bpb:|^training_seconds:|^total_seconds:"
```

Expected: `exit 0`; six `capture: t=` lines (0, 10, 30, 60, 120, about 300) and one `capture: wrote runs/...json`; `training_seconds` 300.0 to 301.0.

- [ ] **Step 2: Score unchanged**

`train.py` on this branch is the session best (commit `e9fffd9`, `val_bpb 0.518708` in `results.tsv`). The check reproduced a historical score to within 0.00002, so allow 0.0005.

```bash
uv run python -c "
import re, sys
v = float(re.search(r'^val_bpb:\s+([0-9.]+)', open('run.log', encoding='utf-8', errors='replace').read(), re.M).group(1))
print('val_bpb', v, 'delta', round(v - 0.518708, 6))
sys.exit(0 if abs(v - 0.518708) <= 0.0005 else 1)
"; echo "within tolerance: exit $?"
```

Expected: `within tolerance: exit 0`. If it fails: rerun once with `AUTORESEARCH_RUNS_DIR=off`. If that uncaptured run also misses, the drift is the machine, not capture; record both numbers and tell TJ. If only the captured run misses, capture is changing training: stop and investigate the RNG and timing hooks.

- [ ] **Step 3: Run-file contents and overhead**

```bash
uv run python -c "
import json, glob
path = sorted(glob.glob('runs/*.json'))[-1]
r = json.load(open(path, encoding='utf-8'))
print(path)
print('times', [s['t_s'] for s in r['snapshots']])
print('final', r['final'])
print('recipe', r['recipe'])
for s in r['snapshots']:
    print(s['t_s'], repr(s['samples'][0][:90]))
assert len(r['snapshots']) == 6 and 'capture_error' not in r
assert r['final']['sampling_s'] < 90
"
```

Expected: six snapshots; the 0 s sample is random vocabulary and the last reads as a story; `sampling_s` under 90 seconds. Record `sampling_s` and the printed samples for TJ.

- [ ] **Step 4: Clean up; this run is not an experiment of record**

```bash
rm runs/*.json
git checkout -- checkpoint_pre_eval.pt
git status --short
```

Expected: `git status --short` prints nothing (`run.log` is gitignored).

---

### Task 7: Port to the starter kit

**Files (in `C:\Users\tj\repos\autoresearch-starter`, branch `main` upstream):**
- Create: `capture.py`, `tests/test_capture.py`, `runs/.gitkeep` (copied from the worked example)
- Modify: `pyproject.toml`, `uv.lock`, `train.py`, `program.md`, `README.md`

**Interfaces:**
- Consumes: the final `capture.py` and `tests/test_capture.py` from Tasks 1–3, and the exact edits from Tasks 4–5.
- Produces: the starter kit writes the same run files, so a learner's overnight run loads into the explorer.

- [ ] **Step 1: Clone and branch**

```bash
unset VIRTUAL_ENV
cd /c/Users/tj/repos
[ -d autoresearch-starter ] || gh repo clone aroughidea/autoresearch-starter
cd autoresearch-starter
git switch main && git pull --ff-only
git switch -c feat/training-capture
```

- [ ] **Step 2: Copy the module, tests and folder**

```bash
unset VIRTUAL_ENV
cp ../autoresearch-win-rtx/capture.py .
mkdir -p tests runs
cp ../autoresearch-win-rtx/tests/test_capture.py tests/
touch runs/.gitkeep
uv add --dev pytest
```

Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Run: `uv run pytest -q`
Expected: `27 passed`.

- [ ] **Step 3: Apply the train.py hooks**

The starter's `train.py` differs from the worked example only in three constants (`WINDOW_PATTERN`, `MATRIX_LR`, `WARMDOWN_RATIO`). Make exactly the nine replacements of Task 4 Steps 1–6. Then confirm the two files differ only in those constants:

```bash
diff --strip-trailing-cr ../autoresearch-win-rtx/train.py train.py
```

Expected: only the `WINDOW_PATTERN`, `MATRIX_LR` and `WARMDOWN_RATIO` lines differ.

- [ ] **Step 4: Apply the program.md edits**

Make the three `program.md` replacements of Task 5 Steps 2–4. The starter's lines 13, 31 and 107 carry the same original text.

Run: `grep -n "capture" program.md`
Expected: three hits (in-scope file, cannot-do rule, step 9).

- [ ] **Step 5: README "What you get"**

Replace:

```markdown
- **`results.tsv`** — your scoreboard. Starts empty; gains one row per experiment: score, VRAM, keep/discard, description.
```

with:

```markdown
- **`results.tsv`** — your scoreboard. Starts empty; gains one row per experiment: score, VRAM, keep/discard, description.
- **`capture.py`** and **`runs/`** — every experiment saves what your model wrote at 0, 10, 30, 60 and 120 seconds and at the end, as a small JSON file in `runs/`. Load the folder into the Training Decisions explorer to watch your models grow and compare them. **Off-limits during experiments**, like `prepare.py`.
```

- [ ] **Step 6: Smoke test**

Check the GPU first. Then:

```bash
unset VIRTUAL_ENV
SMOKE_DIR="$(cygpath -w "$(mktemp -d)")"   # Windows path: Python cannot read /tmp/...
AUTORESEARCH_RUNS_DIR="$SMOKE_DIR" uv run train.py --smoke-test > smoke.log 2>&1; echo "exit $?"
tr '\r' '\n' < smoke.log | grep -E "^capture:|^val_bpb:"
ls "$SMOKE_DIR"
rm -f smoke.log checkpoint_pre_eval.pt
```

Expected: `exit 0`, two snapshot lines and a `capture: wrote` line, one `.json` file. (`*.pt` is gitignored in the starter, so the checkpoint is deleted rather than restored.)

- [ ] **Step 7: Commit, then stop**

```bash
git add capture.py tests/test_capture.py runs/.gitkeep pyproject.toml uv.lock train.py program.md README.md
git commit -m "capture: writing snapshots and a run file per experiment"
git log --oneline -3
```

Then STOP. Report to TJ: the branch names in both repos, the Task 6 evidence (score delta, `sampling_s`, the six samples), and ask before pushing either branch or opening PRs.
