# Dataset and Tokenizer Options Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A learner picks a dataset (TinyStories or Folktales) and a tokenizer (built from the data, Phi-3/Llama 2, or GPT-2) with one `prepare.py` command; training, sampling, chat and run files then use that pair, and the agent never changes it.

**Architecture:** All logic lives in `prepare.py` (read-only to the agent): name resolution (flag, then env var, then the cache's `active_*.txt` file, then default), per-tokenizer cache folders, builders for the two standard tokenizers, and a Folktales downloader. `Tokenizer.from_directory()` follows the active pair and carries `name` and `source`, so `train.py`, `generate.py` and `chat.py` need no change. `capture.py` writes the tokenizer's name and source into run files. Built in the worked example, then ported to the starter.

**Tech Stack:** Python 3.10+, PyTorch 2.9.1, tiktoken, rustbpe, Hugging Face `tokenizers` (new dependency), pyarrow, pytest.

**Spec:** `docs/superpowers/specs/2026-10-06-training-decisions-journey-design.md`, section "Component 2 design: dataset and tokenizer options", plus "Check results".

## Global Constraints

- `train.py` is not modified. It is the agent's file.
- Tokenizer names, exactly: `own` (default), `phi3`, `gpt2`. Dataset names: `tinystories` (default), `folktales`.
- Resolution order for both: command-line flag, then `AUTORESEARCH_DATASET` / `AUTORESEARCH_TOKENIZER`, then `active_dataset.txt` / `active_tokenizer.txt` in the cache root, then the default. A missing `active_tokenizer.txt` means `own` (caches prepared before this change keep working).
- Cache folders: `datasets/<dataset>/tokenizer/` for `own` (unchanged); `datasets/<dataset>/tokenizer-<name>/` otherwise.
- Standard tokenizers get the four `SPECIAL_TOKENS` appended: `phi3` loads as 32,015 ids, `gpt2` as 50,261.
- Sources: Phi-3 `https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/tokenizer.json`; Folktales `https://huggingface.co/datasets/merve/folk-mythology-tales/resolve/main/merged_clean.txt`. Folktales documents: paragraphs packed to at most 1,500 characters; splits `val (0, 300)`, `train (300, None)`, `test (0, 0)`.
- The evaluation metric is unchanged. Only the plumbing that picks the right `token_bytes.pt` changes.
- All file I/O is explicit UTF-8.
- Every GPU step: `nvidia-smi --query-gpu=memory.used --format=csv,noheader` under 3,000 MiB first. Restore `checkpoint_pre_eval.pt` with `git checkout -- checkpoint_pre_eval.pt` after any `train.py` run in the worked example.
- Any step that runs `prepare.py` for real changes the user's active pair in `%LOCALAPPDATA%\autoresearch`. The last GPU step restores it: `uv run prepare.py --dataset tinystories --tokenizer own`.
- In another directory (the starter), `unset VIRTUAL_ENV` first.
- Commits on branch `feat/dataset-tokenizer-options`. Push only with TJ's go-ahead.

## Review Focus

1. **A cache prepared before this change** (no `active_tokenizer.txt`, `tokenizer/` folder only): everything resolves to `own` and works unchanged. Test in Task 1 (`test_tokenizer_default_is_own`).
2. **`uv run prepare.py` writing a pickle that `train.py` cannot load.** The script runs as `__main__`, so pickle could record `__main__.HFEncoding` (this happened in the check). Covered by the import in `_phi3_encoding` and verified end to end in Task 5 (prepare `phi3`, then train).
3. **A mistyped `AUTORESEARCH_TOKENIZER`:** fail loudly, never silently train with another vocabulary. Test in Task 1 (`test_unknown_tokenizer_env_fails_loudly`).
4. **Scoring a model with the wrong `token_bytes.pt`** (active pair changed between training and eval, or a non-default pair): `evaluate_bpb` must use the tokenizer object's own name. Test in Task 3 (`test_get_token_bytes_follows_the_named_tokenizer`).
5. **Offline machines:** tests needing the GPT-2 vocabulary download skip instead of failing. Test in Task 2 (`test_gpt2_encoding_adds_reserved_tokens` skips offline).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `prepare.py` | Modify | Names and resolution; cache folders; standard tokenizers; Folktales; `main(argv)` with `--tokenizer`. |
| `capture.py` | Modify | Run file's `tokenizer` field gains `name` from the tokenizer object and `source`. |
| `tests/test_prepare_options.py` | Create | CPU-only tests of the above (one test needs a network download, skipped offline). |
| `tests/test_capture.py` | Modify | Expectation for the new `source` key; a named-tokenizer test. |
| `pyproject.toml`, `uv.lock` | Modify | Add `tokenizers`. |
| `program.md` | Modify | Cache check mentions `tokenizer-<name>/` and `active_tokenizer.txt`; the agent must not change the pair. |
| `README.md` | Modify | "What about using different datasets?" becomes the dataset and tokenizer choice. |

---

### Task 1: Tokenizer names, resolution and cache folders

**Files:**
- Modify: `prepare.py` (imports; constants after `DATASET_CHOICES`; `ACTIVE_TOKENIZER_PATH` after `ACTIVE_DATASET_PATH`; new functions after `_set_active_dataset`; `_tokenizer_dir`)
- Create: `tests/test_prepare_options.py`

**Interfaces:**
- Produces: `TOKENIZER_CHOICES`, `DEFAULT_TOKENIZER`, `TOKENIZER_SOURCES`, `ACTIVE_TOKENIZER_PATH`, `_resolve_tokenizer_name(tokenizer_name=None) -> str`, `_set_active_tokenizer(name) -> None`, `_tokenizer_dir(dataset_name=None, tokenizer_name=None) -> str`.

- [ ] **Step 1: Create the branch**

```bash
cd /c/Users/tj/repos/autoresearch-win-rtx
git switch feat/dataset-tokenizer-options   # created with the spec commit ef31796
git add docs/superpowers/plans/2026-10-07-dataset-tokenizer-options.md
git commit -m "docs: dataset and tokenizer options plan"
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_prepare_options.py`:

```python
import os

import pytest

import prepare


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTORESEARCH_TOKENIZER", raising=False)
    monkeypatch.delenv("AUTORESEARCH_DATASET", raising=False)
    monkeypatch.setattr(prepare, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(prepare, "DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setattr(prepare, "ACTIVE_DATASET_PATH", str(tmp_path / "active_dataset.txt"))
    monkeypatch.setattr(prepare, "ACTIVE_TOKENIZER_PATH", str(tmp_path / "active_tokenizer.txt"))
    return tmp_path


def test_tokenizer_default_is_own(clean_env):
    assert prepare._resolve_tokenizer_name() == "own"


def test_flag_beats_env_beats_active_file(clean_env, monkeypatch):
    (clean_env / "active_tokenizer.txt").write_text("gpt2\n", encoding="utf-8")
    assert prepare._resolve_tokenizer_name() == "gpt2"
    monkeypatch.setenv("AUTORESEARCH_TOKENIZER", "phi3")
    assert prepare._resolve_tokenizer_name() == "phi3"
    assert prepare._resolve_tokenizer_name("own") == "own"


def test_unknown_tokenizer_flag_is_rejected(clean_env):
    with pytest.raises(ValueError):
        prepare._resolve_tokenizer_name("llama3")


def test_unknown_tokenizer_env_fails_loudly(clean_env, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_TOKENIZER", "gtp2")
    with pytest.raises(ValueError):
        prepare._resolve_tokenizer_name()


def test_set_active_tokenizer_is_read_back(clean_env):
    prepare._set_active_tokenizer("phi3")
    assert prepare._resolve_tokenizer_name() == "phi3"


def test_tokenizer_folders(clean_env):
    own = prepare._tokenizer_dir("tinystories", "own")
    phi3 = prepare._tokenizer_dir("folktales", "phi3")
    assert own.endswith(os.path.join("tinystories", "tokenizer"))
    assert phi3.endswith(os.path.join("folktales", "tokenizer-phi3"))


def test_every_tokenizer_has_a_source():
    assert set(prepare.TOKENIZER_SOURCES) == set(prepare.TOKENIZER_CHOICES)
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_prepare_options.py -q`
Expected: failures with `AttributeError: module 'prepare' has no attribute '_resolve_tokenizer_name'` (or `TOKENIZER_SOURCES`, `_set_active_tokenizer`); `test_tokenizer_folders` fails with a `TypeError` (`_tokenizer_dir` takes one argument).

- [ ] **Step 4: Implement**

In `prepare.py`, add `import re` after `import pickle`.

Replace `DATASET_CHOICES = ("tinystories",)` with:

```python
DATASET_CHOICES = ("tinystories", "folktales")

TOKENIZER_CHOICES = ("own", "phi3", "gpt2")
DEFAULT_TOKENIZER = "own"
TOKENIZER_SOURCES = {
    "own": "BPE trained on the dataset (rustbpe, 8,192 tokens)",
    "phi3": "microsoft/Phi-3-mini-4k-instruct tokenizer.json (Llama 2 vocabulary, 32,011 tokens)",
    "gpt2": "OpenAI GPT-2 via tiktoken (50,257 tokens)",
}
PHI3_TOKENIZER_URL = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/tokenizer.json"
FOLKTALES_TXT_URL = "https://huggingface.co/datasets/merve/folk-mythology-tales/resolve/main/merged_clean.txt"
```

Replace `ACTIVE_DATASET_PATH = os.path.join(CACHE_DIR, "active_dataset.txt")` with:

```python
ACTIVE_DATASET_PATH = os.path.join(CACHE_DIR, "active_dataset.txt")
ACTIVE_TOKENIZER_PATH = os.path.join(CACHE_DIR, "active_tokenizer.txt")
```

After the `_set_active_dataset` function, add:

```python
def _normalize_tokenizer_name(tokenizer_name):
    if tokenizer_name is None:
        return None
    value = tokenizer_name.strip().lower()
    if value not in TOKENIZER_CHOICES:
        raise ValueError(f"Unknown tokenizer '{tokenizer_name}'. Expected one of {TOKENIZER_CHOICES}.")
    return value


def _resolve_tokenizer_name(tokenizer_name=None):
    """Flag, then AUTORESEARCH_TOKENIZER, then active_tokenizer.txt, then 'own'.

    A mistyped environment variable raises rather than silently training with another vocabulary.
    """
    explicit = _normalize_tokenizer_name(tokenizer_name)
    if explicit is not None:
        return explicit
    env_value = os.environ.get("AUTORESEARCH_TOKENIZER")
    if env_value:
        return _normalize_tokenizer_name(env_value)
    if os.path.exists(ACTIVE_TOKENIZER_PATH):
        with open(ACTIVE_TOKENIZER_PATH, "r", encoding="utf-8") as f:
            value = f.read().strip().lower()
        if value in TOKENIZER_CHOICES:
            return value
    return DEFAULT_TOKENIZER


def _set_active_tokenizer(tokenizer_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(ACTIVE_TOKENIZER_PATH, "w", encoding="utf-8") as f:
        f.write(tokenizer_name + "\n")
```

Replace:

```python
def _tokenizer_dir(dataset_name=None):
    return os.path.join(_dataset_root(dataset_name), "tokenizer")
```

with:

```python
def _tokenizer_dir(dataset_name=None, tokenizer_name=None):
    name = _resolve_tokenizer_name(tokenizer_name)
    folder = "tokenizer" if name == DEFAULT_TOKENIZER else f"tokenizer-{name}"
    return os.path.join(_dataset_root(dataset_name), folder)
```

- [ ] **Step 5: Run to verify they pass, and nothing else broke**

Run: `uv run pytest -q`
Expected: `45 passed` (38 existing + 7 new).

- [ ] **Step 6: Commit**

```bash
git add prepare.py tests/test_prepare_options.py
git commit -m "prepare: tokenizer names, resolution order and per-tokenizer cache folders"
```

---

### Task 2: Standard tokenizers (Phi-3, GPT-2)

**Files:**
- Modify: `prepare.py` (new code before the `# Runtime utilities` banner; `train_tokenizer`)
- Modify: `pyproject.toml`, `uv.lock` (add `tokenizers`)
- Modify: `tests/test_prepare_options.py` (append)

**Interfaces:**
- Consumes: `_resolve_tokenizer_name`, `_tokenizer_dir`, `TOKENIZER_SOURCES` (Task 1).
- Produces: `HFEncoding(json_str, reserved)` (tiktoken-shaped: `n_vocab`, `encode_single_token`, `encode_ordinary`, `encode_ordinary_batch`, `decode`); `_piece_byte_length(piece, special) -> int`; `_gpt2_encoding()`; `_phi3_encoding()`; `train_tokenizer(dataset_name=None, tokenizer_name=None)`.

- [ ] **Step 1: Add the dependency**

Run: `uv add tokenizers`
Expected: `pyproject.toml` dependencies gain `"tokenizers>=0.23..."`.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_prepare_options.py`:

```python
import pickle


def _tiny_hf_json():
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "once": 1, "upon": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok.to_str()


def test_hfencoding_adds_reserved_tokens_and_pickles():
    enc = prepare.HFEncoding(_tiny_hf_json(), prepare.SPECIAL_TOKENS)
    again = pickle.loads(pickle.dumps(enc))
    assert again.n_vocab == 3 + len(prepare.SPECIAL_TOKENS)
    assert again.encode_single_token(prepare.BOS_TOKEN) == 3
    assert again.encode_ordinary("once upon") == [1, 2]
    assert again.encode_ordinary_batch(["once", "upon"]) == [[1], [2]]


def test_piece_byte_length():
    assert prepare._piece_byte_length("\u2581Once", special=False) == 5   # " Once"
    assert prepare._piece_byte_length("ily", special=False) == 3
    assert prepare._piece_byte_length("<0x0A>", special=False) == 1       # byte fallback
    assert prepare._piece_byte_length("<s>", special=True) == 0


def test_gpt2_encoding_adds_reserved_tokens():
    try:
        enc = prepare._gpt2_encoding()
    except Exception as exc:  # the GPT-2 vocabulary downloads once; skip offline
        pytest.skip(f"GPT-2 vocabulary unavailable: {exc}")
    assert enc.n_vocab == 50257 + len(prepare.SPECIAL_TOKENS)
    assert enc.encode_single_token(prepare.BOS_TOKEN) == 50257
    ids = enc.encode_ordinary("Once upon a time")
    assert len(ids) == 4 and enc.decode(ids) == "Once upon a time"
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_prepare_options.py -q`
Expected: the 3 new tests fail with `AttributeError: module 'prepare' has no attribute 'HFEncoding'` / `_piece_byte_length` / `_gpt2_encoding`.

- [ ] **Step 4: Implement**

Insert before the `# Runtime utilities (imported by train.py)` banner in `prepare.py`:

```python
class HFEncoding:
    """A Hugging Face tokenizer.json behind the small tiktoken-shaped interface Tokenizer uses."""

    def __init__(self, json_str, reserved):
        self._json = json_str
        self._reserved = list(reserved)
        self._build()

    def _build(self):
        from tokenizers import Tokenizer as HFTokenizer

        self.tok = HFTokenizer.from_str(self._json)
        self.tok.add_special_tokens(self._reserved)

    def __getstate__(self):
        return {"_json": self._json, "_reserved": self._reserved}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build()

    @property
    def n_vocab(self):
        return self.tok.get_vocab_size(with_added_tokens=True)

    def encode_single_token(self, text):
        token_id = self.tok.token_to_id(text)
        if token_id is None:
            raise KeyError(text)
        return token_id

    def encode_ordinary(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_ordinary_batch(self, texts, num_threads=8):
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=False)


def _piece_byte_length(piece, special):
    """Bytes of text a SentencePiece-style token stands for (0 for special tokens)."""
    if special or piece is None:
        return 0
    if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
        return 1
    return len(piece.replace("\u2581", " ").encode("utf-8"))


def _gpt2_encoding():
    base = tiktoken.get_encoding("gpt2")
    specials = dict(base._special_tokens)
    for i, name in enumerate(SPECIAL_TOKENS):
        specials[name] = base.n_vocab + i
    return tiktoken.Encoding(
        name="gpt2-reserved",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=specials,
    )


def _phi3_encoding():
    # Import through the module name so pickle records prepare.HFEncoding even when this file
    # runs as __main__ (`uv run prepare.py`); otherwise train.py cannot load the tokenizer.
    from prepare import HFEncoding as hf_encoding_class

    response = requests.get(PHI3_TOKENIZER_URL, timeout=120)
    response.raise_for_status()
    return hf_encoding_class(response.text, SPECIAL_TOKENS)


def _build_standard_tokenizer(tokenizer_name, tokenizer_dir, dataset):
    """Fetch a published vocabulary instead of training one."""
    if tokenizer_name == "gpt2":
        enc = _gpt2_encoding()
        special_ids = {enc.encode_single_token(t) for t in enc.special_tokens_set}
        token_bytes = [0 if i in special_ids else len(enc.decode_single_token_bytes(i)) for i in range(enc.n_vocab)]
    elif tokenizer_name == "phi3":
        enc = _phi3_encoding()
        special_ids = set(enc.tok.get_added_tokens_decoder().keys())
        token_bytes = [_piece_byte_length(enc.tok.id_to_token(i), i in special_ids) for i in range(enc.n_vocab)]
    else:
        raise ValueError(tokenizer_name)
    os.makedirs(tokenizer_dir, exist_ok=True)
    with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(enc, f)
    torch.save(torch.tensor(token_bytes, dtype=torch.int32), os.path.join(tokenizer_dir, "token_bytes.pt"))
    with open(os.path.join(tokenizer_dir, "dataset.txt"), "w", encoding="utf-8") as f:
        f.write(dataset + "\n")
    sample = "Once upon a time, Lily found a shiny shell."
    print(f"Tokenizer: {tokenizer_name} ready (vocab_size={enc.n_vocab}); "
          f"'{sample}' is {len(enc.encode_ordinary(sample))} tokens")
```

In `train_tokenizer`, replace:

```python
def train_tokenizer(dataset_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    tokenizer_dir = _tokenizer_dir(dataset)
```

with:

```python
def train_tokenizer(dataset_name=None, tokenizer_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    tokenizer = _resolve_tokenizer_name(tokenizer_name)
    tokenizer_dir = _tokenizer_dir(dataset, tokenizer)
```

and replace:

```python
    os.makedirs(tokenizer_dir, exist_ok=True)

    parquet_files = list_parquet_files(dataset)
```

with:

```python
    os.makedirs(tokenizer_dir, exist_ok=True)
    if tokenizer != DEFAULT_TOKENIZER:
        _build_standard_tokenizer(tokenizer, tokenizer_dir, dataset)
        return

    parquet_files = list_parquet_files(dataset)
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest -q`
Expected: `48 passed` (the GPT-2 test may show as skipped offline: then `47 passed, 1 skipped`).

- [ ] **Step 6: Commit**

```bash
git add prepare.py pyproject.toml uv.lock tests/test_prepare_options.py
git commit -m "prepare: Phi-3 and GPT-2 tokenizers with reserved control tokens"
```

---

### Task 3: Tokenizer objects carry their name; scoring and run files follow it

**Files:**
- Modify: `prepare.py` (`Tokenizer.__init__`, `Tokenizer.from_directory`, `get_token_bytes`, one line in `evaluate_bpb`)
- Modify: `capture.py` (the `tokenizer` field in `_write`)
- Modify: `tests/test_prepare_options.py`, `tests/test_capture.py`

**Interfaces:**
- Consumes: `_resolve_tokenizer_name`, `_tokenizer_dir`, `TOKENIZER_SOURCES` (Task 1).
- Produces: `Tokenizer(enc, dataset, name="own")` with `.name` and `.source`; `Tokenizer.from_directory(tokenizer_dir=None, dataset=None, tokenizer=None)`; `get_token_bytes(device="cpu", dataset=None, tokenizer=None)`; run file `"tokenizer": {"name", "source", "vocab_size"}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prepare_options.py`:

```python
import tiktoken
import torch


def _tiny_tiktoken():
    ranks = {bytes([i]): i for i in range(256)}
    specials = {name: 256 + i for i, name in enumerate(prepare.SPECIAL_TOKENS)}
    return tiktoken.Encoding(name="tiny", pat_str=r"\S+|\s+", mergeable_ranks=ranks, special_tokens=specials)


def test_from_directory_names_the_tokenizer(tmp_path):
    with open(tmp_path / "tokenizer.pkl", "wb") as f:
        pickle.dump(_tiny_tiktoken(), f)
    tok = prepare.Tokenizer.from_directory(tokenizer_dir=str(tmp_path), dataset="tinystories", tokenizer="phi3")
    assert tok.name == "phi3"
    assert tok.source == prepare.TOKENIZER_SOURCES["phi3"]
    assert tok.get_vocab_size() == 260


def test_get_token_bytes_follows_the_named_tokenizer(clean_env):
    for name, value in (("own", 1), ("gpt2", 2)):
        folder = prepare._tokenizer_dir("tinystories", name)
        os.makedirs(folder)
        torch.save(torch.tensor([value], dtype=torch.int32), os.path.join(folder, "token_bytes.pt"))
    prepare._set_active_tokenizer("own")
    assert prepare.get_token_bytes(dataset="tinystories", tokenizer="gpt2").item() == 2
    assert prepare.get_token_bytes(dataset="tinystories").item() == 1
```

In `tests/test_capture.py`, change the existing assertion in `test_finish_writes_named_run_file`:

```python
    assert record["tokenizer"] == {"name": "own", "vocab_size": 258}
```

to:

```python
    assert record["tokenizer"] == {"name": "own", "source": None, "vocab_size": 258}
```

and append:

```python
def test_run_file_records_the_tokenizer_objects_name_and_source(tmp_path):
    named = FakeTokenizer()
    named.name, named.source = "phi3", "microsoft/Phi-3-mini-4k-instruct tokenizer.json"
    cap = capture.RunCapture(named, dataset="folktales", runs_dir=tmp_path, device="cpu",
                             snapshot_times=(0,), decoding=FAST, log=lambda m: None)
    record = json.loads(_finish(cap).read_text(encoding="utf-8"))
    assert record["tokenizer"] == {"name": "phi3", "source": named.source, "vocab_size": 258}
    assert record["dataset"] == "folktales"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest -q`
Expected: 4 failures: `test_from_directory_names_the_tokenizer` (`TypeError: ... unexpected keyword argument 'tokenizer'`), `test_get_token_bytes_follows_the_named_tokenizer` (`TypeError`), `test_finish_writes_named_run_file` and `test_run_file_records_the_tokenizer_objects_name_and_source` (the `source` key is missing).

- [ ] **Step 3: Implement**

In `prepare.py`, replace:

```python
    def __init__(self, enc, dataset):
        self.enc = enc
        self.dataset = _resolve_dataset_name(dataset)
```

with:

```python
    def __init__(self, enc, dataset, name=DEFAULT_TOKENIZER):
        self.enc = enc
        self.dataset = _resolve_dataset_name(dataset)
        self.name = name
        self.source = TOKENIZER_SOURCES[name]
```

Replace:

```python
    def from_directory(cls, tokenizer_dir=None, dataset=None):
        dataset_name = _resolve_dataset_name(dataset)
        resolved_dir = tokenizer_dir if tokenizer_dir is not None else _tokenizer_dir(dataset_name)
        with open(os.path.join(resolved_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc, dataset=dataset_name)
```

with:

```python
    def from_directory(cls, tokenizer_dir=None, dataset=None, tokenizer=None):
        dataset_name = _resolve_dataset_name(dataset)
        tokenizer_name = _resolve_tokenizer_name(tokenizer)
        resolved_dir = tokenizer_dir if tokenizer_dir is not None else _tokenizer_dir(dataset_name, tokenizer_name)
        with open(os.path.join(resolved_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc, dataset=dataset_name, name=tokenizer_name)
```

Replace:

```python
def get_token_bytes(device="cpu", dataset=None):
    dataset_name = _resolve_dataset_name(dataset)
    path = os.path.join(_tokenizer_dir(dataset_name), "token_bytes.pt")
```

with:

```python
def get_token_bytes(device="cpu", dataset=None, tokenizer=None):
    dataset_name = _resolve_dataset_name(dataset)
    path = os.path.join(_tokenizer_dir(dataset_name, tokenizer), "token_bytes.pt")
```

In `evaluate_bpb`, replace:

```python
    token_bytes = get_token_bytes(device=device, dataset=dataset_name)
```

with:

```python
    token_bytes = get_token_bytes(device=device, dataset=dataset_name, tokenizer=getattr(tokenizer, "name", None))
```

In `capture.py`, replace:

```python
            "tokenizer": {"name": self.tokenizer_name, "vocab_size": int(self.tokenizer.get_vocab_size())},
```

with:

```python
            "tokenizer": {
                "name": getattr(self.tokenizer, "name", self.tokenizer_name),
                "source": getattr(self.tokenizer, "source", None),
                "vocab_size": int(self.tokenizer.get_vocab_size()),
            },
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest -q`
Expected: `51 passed` (or `50 passed, 1 skipped` offline).

- [ ] **Step 5: Commit**

```bash
git add prepare.py capture.py tests/test_prepare_options.py tests/test_capture.py
git commit -m "prepare: tokenizers carry their name; scoring and run files follow it"
```

---

### Task 4: Folktales and the `prepare.py` command line

**Files:**
- Modify: `prepare.py` (`DATASET_CONFIGS`; `download_data`; the `if __name__ == "__main__":` block becomes `main(argv=None)`)
- Modify: `tests/test_prepare_options.py` (append)

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `_pack_paragraphs(raw_text, max_chars=1500) -> list[str]`; `main(argv=None) -> int`; `uv run prepare.py --dataset {tinystories,folktales} --tokenizer {own,phi3,gpt2}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prepare_options.py`:

```python
def test_pack_paragraphs_joins_wrapped_lines_and_caps_length():
    raw = "Lovely Ilonka\n\nThere was once\na king's son.\n\n\nHe wished to marry.\n\n" + ("word " * 400)
    docs = prepare._pack_paragraphs(raw, max_chars=80)
    assert docs[0] == "Lovely Ilonka\n\nThere was once a king's son.\n\nHe wished to marry."
    assert docs[1].startswith("word word") and len(docs) == 2


def test_folktales_splits():
    assert prepare.DATASET_CONFIGS["folktales"]["splits"] == {"test": (0, 0), "val": (0, 300), "train": (300, None)}


def test_main_prepares_and_activates_the_pair(clean_env, monkeypatch):
    calls = []
    monkeypatch.setattr(prepare, "download_data", lambda d: calls.append(("download", d)))
    monkeypatch.setattr(prepare, "train_tokenizer", lambda d, t: calls.append(("tokenizer", d, t)))
    assert prepare.main(["--dataset", "folktales", "--tokenizer", "phi3"]) == 0
    assert calls == [("download", "folktales"), ("tokenizer", "folktales", "phi3")]
    assert (clean_env / "active_dataset.txt").read_text(encoding="utf-8").strip() == "folktales"
    assert (clean_env / "active_tokenizer.txt").read_text(encoding="utf-8").strip() == "phi3"


def test_main_without_flags_keeps_the_active_pair(clean_env, monkeypatch):
    monkeypatch.setattr(prepare, "download_data", lambda d: None)
    monkeypatch.setattr(prepare, "train_tokenizer", lambda d, t: None)
    prepare._set_active_tokenizer("gpt2")
    prepare.main([])
    assert (clean_env / "active_tokenizer.txt").read_text(encoding="utf-8").strip() == "gpt2"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_prepare_options.py -q`
Expected: the 4 new tests fail (`_pack_paragraphs`, `main` missing; `KeyError: 'folktales'`).

- [ ] **Step 3: Implement**

Replace:

```python
DATASET_CONFIGS = {
    "tinystories": {
```

with:

```python
DATASET_CONFIGS = {
    "folktales": {
        # merve/folk-mythology-tales (CC0 1.0 per its card), packed into documents by _pack_paragraphs.
        "filename": "folktales.parquet",
        "splits": {
            "test": (0, 0),
            "val": (0, 300),
            "train": (300, None),
        },
    },
    "tinystories": {
```

Replace:

```python
def download_data(dataset_name):
    dataset = _resolve_dataset_name(dataset_name)
    _download_tinystories_file(dataset)
```

with:

```python
def _pack_paragraphs(raw_text, max_chars=1500):
    """Blank-line paragraphs, wrapped lines joined, packed in order into documents of at most max_chars."""
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
    docs, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) > max_chars:
            docs.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        docs.append(current)
    return docs


def _build_folktales_parquet(dataset_name):
    import pyarrow as pa

    path = _tiny_parquet_path(dataset_name)
    if os.path.exists(path):
        print(f"Data: folktales already built at {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    response = requests.get(FOLKTALES_TXT_URL, timeout=120)
    response.raise_for_status()
    docs = _pack_paragraphs(response.text)
    temp_path = path + ".tmp"
    pq.write_table(pa.table({"text": docs}), temp_path)
    os.replace(temp_path, path)
    print(f"Data: folktales -> {len(docs):,} documents, {sum(map(len, docs)):,} characters at {path}")


def download_data(dataset_name):
    dataset = _resolve_dataset_name(dataset_name)
    if dataset == "folktales":
        _build_folktales_parquet(dataset)
        return
    _download_tinystories_file(dataset)
```

Replace the whole block from `if __name__ == "__main__":` to the end of the file with:

```python
def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=None,
        help="Dataset to prepare and make active. Default: AUTORESEARCH_DATASET, then the active one, then tinystories.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=TOKENIZER_CHOICES,
        default=None,
        help="Tokenizer to build or fetch and make active: own (trained on the dataset), phi3 or gpt2. "
             "Default: AUTORESEARCH_TOKENIZER, then the active one, then own.",
    )
    args = parser.parse_args(argv)

    dataset_name = _resolve_dataset_name(args.dataset)
    tokenizer_name = _resolve_tokenizer_name(args.tokenizer)

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Dataset: {dataset_name}")
    print(f"Tokenizer: {tokenizer_name} ({TOKENIZER_SOURCES[tokenizer_name]})")
    print()

    download_data(dataset_name)
    print()
    train_tokenizer(dataset_name, tokenizer_name)
    _set_active_dataset(dataset_name)
    _set_active_tokenizer(tokenizer_name)
    print()
    if tokenizer_name == "gpt2":
        print("Note: the GPT-2 vocabulary needs about 10 GB of GPU memory; it will not fit on 8 GB cards.")
    print(f"Done! Ready to train. Active: dataset '{dataset_name}', tokenizer '{tokenizer_name}'.")
    print("val_bpb compares across tokenizers on the same dataset, never across datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest -q`
Expected: `55 passed` (or `54 passed, 1 skipped` offline).

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_prepare_options.py
git commit -m "prepare: Folktales dataset and --tokenizer on the command line"
```

---

### Task 5: Docs, then end-to-end on the GPU

**Files:**
- Modify: `program.md` (setup step 4; "What you CANNOT do")
- Modify: `README.md` ("What about using different datasets?" section)

**Interfaces:**
- Consumes: the command line from Task 4.

- [ ] **Step 1: program.md**

Replace:

```markdown
`datasets\<dataset>\tokenizer\` with `tokenizer.pkl` and `token_bytes.pt`, and `active_dataset.txt` at the cache root.
```

with:

```markdown
`datasets\<dataset>\tokenizer\` (or `tokenizer-<name>\` when a standard tokenizer such as `phi3` or `gpt2` is active) with `tokenizer.pkl` and `token_bytes.pt`, and `active_dataset.txt` and `active_tokenizer.txt` at the cache root (no `active_tokenizer.txt` means `own`).
```

Replace:

```markdown
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
```

with:

```markdown
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Change the dataset or the tokenizer: do not run `prepare.py`, set `AUTORESEARCH_DATASET` or `AUTORESEARCH_TOKENIZER`, or edit the `active_*.txt` files. They are the human's decisions for this campaign, and scores only compare within one pair.
```

- [ ] **Step 2: README**

Replace everything from the line `### What about using different datasets?` up to (not including) `### Can I change the time limit?` with:

````markdown
### What about using different datasets or tokenizers?

Stay on TinyStories with its own tokenizer until you have stable, repeatable runs. Then change one decision at a time. Two of each are built in, and choosing them is a human step (the agent never changes them):

```powershell
uv run prepare.py --dataset folktales                       # folk and myth tales instead of TinyStories
uv run prepare.py --dataset tinystories --tokenizer phi3    # Phi-3 / Llama 2 vocabulary (32,011 tokens)
uv run prepare.py --dataset tinystories --tokenizer gpt2    # GPT-2 vocabulary (50,257); needs about 10 GB of GPU memory
uv run prepare.py --dataset tinystories --tokenizer own     # back to the default
```

`prepare.py` downloads what it needs and makes that pair active; `train.py`, `generate.py` and `chat.py` all use the active pair, and each run file records it.

- `val_bpb` compares across tokenizers on the same dataset, never across datasets. Start a fresh `results.tsv` when you switch dataset.
- A model can only be chatted with under the pair it was trained with.
- To add your own dataset, add an entry to `DATASET_CONFIGS` and its name to `DATASET_CHOICES` in `prepare.py`.

````

Run: `grep -n "tokenizer" program.md | head; grep -n "different datasets or tokenizers" README.md`
Expected: the new program.md lines and the README heading appear.

- [ ] **Step 3: Commit the docs**

```bash
git add program.md README.md
git commit -m "docs: choosing a dataset and tokenizer; the agent never changes them"
```

- [ ] **Step 4: Prepare Folktales with the Phi-3 tokenizer, then smoke-test**

Check the GPU first. Then:

```bash
uv run prepare.py --dataset folktales --tokenizer phi3 2>&1 | tail -6
SMOKE_DIR="$(cygpath -w "$(mktemp -d)")"
AUTORESEARCH_RUNS_DIR="$SMOKE_DIR" uv run train.py --smoke-test > smoke.log 2>&1; echo "exit $?"
tr '\r' '\n' < smoke.log | grep -E "^Vocab size|^Dataset|^val_bpb"
uv run python -c "
import json, glob, sys
r = json.load(open(glob.glob(sys.argv[1] + '/*.json')[0], encoding='utf-8'))
print(r['dataset'], r['tokenizer'])
assert r['dataset'] == 'folktales' and r['tokenizer']['name'] == 'phi3' and r['tokenizer']['vocab_size'] == 32015
" "$SMOKE_DIR"
git checkout -- checkpoint_pre_eval.pt
```

Expected: prepare prints `folktales -> 9,195 documents` and `phi3 ready (vocab_size=32015)` and `Active: dataset 'folktales', tokenizer 'phi3'`; smoke `exit 0` with `Vocab size: 32,015` and `Dataset: folktales`; the run file check passes. This also proves Review Focus 2: `train.py` loaded the pickle `prepare.py` wrote.

- [ ] **Step 5: GPT-2 on TinyStories, smoke-test**

```bash
uv run prepare.py --dataset tinystories --tokenizer gpt2 2>&1 | tail -4
uv run train.py --smoke-test > smoke.log 2>&1; echo "exit $?"
tr '\r' '\n' < smoke.log | grep -E "^Vocab size|^val_bpb"
git checkout -- checkpoint_pre_eval.pt
```

Expected: the 10 GB note prints; smoke `exit 0` with `Vocab size: 50,261`.

- [ ] **Step 6: One full run with Phi-3 on TinyStories, then chat with it**

```bash
uv run prepare.py --dataset tinystories --tokenizer phi3 2>&1 | tail -2
RUNS="$(cygpath -w "$(mktemp -d)")"
AUTORESEARCH_RUNS_DIR="$RUNS" uv run train.py > run.log 2>&1; echo "exit $?"
tr '\r' '\n' < run.log | grep -E "^val_bpb|^num_steps|^peak_vram_mb"
uv run generate.py "Once upon a time" --max-tokens 40 2>&1 | tail -3
git checkout -- checkpoint_pre_eval.pt
```

Expected: `exit 0`; `val_bpb` within 0.01 of the check's 0.570 (same recipe and tokenizer; run-to-run noise is about 0.003); `generate.py` prints readable story text, which proves the chat path follows the active tokenizer.

- [ ] **Step 7: Restore the default pair**

```bash
uv run prepare.py --dataset tinystories --tokenizer own 2>&1 | tail -1
cat "$LOCALAPPDATA/autoresearch/active_dataset.txt" "$LOCALAPPDATA/autoresearch/active_tokenizer.txt"
rm -f smoke.log
git status --short
```

Expected: `Active: dataset 'tinystories', tokenizer 'own'`; the two files read `tinystories` and `own`; `git status` shows nothing new (`run.log` is gitignored).

---

### Task 6: Port to the starter kit

**Files (in `C:\Users\tj\repos\autoresearch-starter`, from `main`):**
- Create branch `feat/dataset-tokenizer-options`
- Copy: `prepare.py`, `capture.py`, `tests/test_prepare_options.py`, `tests/test_capture.py`
- Modify: `pyproject.toml`, `uv.lock`, `program.md`, `README.md`

- [ ] **Step 1: Branch and copy**

```bash
unset VIRTUAL_ENV
cd /c/Users/tj/repos/autoresearch-starter
git switch main && git pull --ff-only
git switch -c feat/dataset-tokenizer-options
cp ../autoresearch-win-rtx/prepare.py ../autoresearch-win-rtx/capture.py .
cp ../autoresearch-win-rtx/tests/test_prepare_options.py ../autoresearch-win-rtx/tests/test_capture.py tests/
uv add tokenizers
uv run pytest -q
```

Expected: `55 passed` (or `54 passed, 1 skipped`).

- [ ] **Step 2: program.md**

Make the two replacements of Task 5 Step 1. The starter's lines carry the same original text.

- [ ] **Step 3: README**

Replace everything from `### Swap the dataset` up to (not including) `### The headline swap: rewrite \`program.md\`` with:

````markdown
### Swap the dataset or the tokenizer

Two datasets and three tokenizers are built in. Choosing them is your decision, not the agent's:

```powershell
uv run prepare.py --dataset folktales                       # folk and myth tales instead of TinyStories
uv run prepare.py --dataset tinystories --tokenizer phi3    # Phi-3 / Llama 2 vocabulary (32,011 tokens)
uv run prepare.py --dataset tinystories --tokenizer gpt2    # GPT-2 vocabulary (50,257); needs about 10 GB of GPU memory
uv run prepare.py --dataset tinystories --tokenizer own     # back to the default: a vocabulary built from the data
```

The pair you prepare last is active for training, `generate.py` and `chat.py`, and every run file records it. Scores compare across tokenizers on the same dataset, never across datasets: start a fresh `results.tsv` when you switch dataset.

To add your own text (poetry, code, your own writing), add an entry to `DATASET_CONFIGS` and its name to `DATASET_CHOICES` in `prepare.py`.

````

- [ ] **Step 4: Smoke test with the default pair**

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
SMOKE_DIR="$(cygpath -w "$(mktemp -d)")"
AUTORESEARCH_RUNS_DIR="$SMOKE_DIR" uv run train.py --smoke-test > smoke.log 2>&1; echo "exit $?"
tr '\r' '\n' < smoke.log | grep -E "^Vocab size|^val_bpb"
rm -f smoke.log checkpoint_pre_eval.pt
```

Expected: `exit 0`, `Vocab size: 8,192`.

- [ ] **Step 5: Commit, then stop**

```bash
git add prepare.py capture.py tests/test_prepare_options.py tests/test_capture.py pyproject.toml uv.lock program.md README.md
git commit -m "prepare: choose a dataset (TinyStories, Folktales) and tokenizer (own, Phi-3, GPT-2)"
```

Then STOP and report to TJ: both branches, the Task 5 evidence (Folktales + Phi-3 smoke, GPT-2 smoke, the full Phi-3 run's `val_bpb`, the `generate.py` sample), and ask before pushing or opening pull requests.
