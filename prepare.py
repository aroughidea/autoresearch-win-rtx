"""
One-time data preparation for autoresearch experiments.
Downloads data and trains a BPE tokenizer.

Usage:
    python prepare.py

Data and tokenizer are stored in the cache directory (overridable with
AUTORESEARCH_CACHE_DIR). The active dataset can be pinned with
AUTORESEARCH_DATASET or by running this script with --dataset.
"""

import argparse
import math
import os
import pickle
import re
import shutil
import time

import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048          # context length
TIME_BUDGET = 300           # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288   # number of tokens for validation eval
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"
EOS_TOKEN = "<|reserved_1|>"  # end-of-sequence: appended to every document during data prep

# ---------------------------------------------------------------------------
# Dataset + cache configuration
# ---------------------------------------------------------------------------

DEFAULT_DATASET = "tinystories"
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


def _default_cache_dir():
    env_cache = os.environ.get("AUTORESEARCH_CACHE_DIR")
    if env_cache:
        return os.path.expanduser(env_cache)

    legacy_cache = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
    if os.name != "nt":
        return legacy_cache

    if os.path.exists(legacy_cache):
        return legacy_cache

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return os.path.join(local_app_data, "autoresearch")
    return legacy_cache


CACHE_DIR = _default_cache_dir()
DATASETS_DIR = os.path.join(CACHE_DIR, "datasets")
ACTIVE_DATASET_PATH = os.path.join(CACHE_DIR, "active_dataset.txt")
ACTIVE_TOKENIZER_PATH = os.path.join(CACHE_DIR, "active_tokenizer.txt")

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
        "filename": "tinystories_gpt4_clean.parquet",
        "url": "https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/resolve/main/tinystories_gpt4_clean.parquet",
        "splits": {
            "test": (0, 10_000),
            "val": (10_000, 20_000),
            "train": (20_000, None),
        },
    },
}


def _normalize_dataset_name(dataset_name):
    if dataset_name is None:
        return None
    value = dataset_name.strip().lower()
    if value not in DATASET_CHOICES:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of {DATASET_CHOICES}.")
    return value


def _load_active_dataset_from_file():
    if not os.path.exists(ACTIVE_DATASET_PATH):
        return None
    with open(ACTIVE_DATASET_PATH, "r", encoding="utf-8") as f:
        value = f.read().strip().lower()
    if value in DATASET_CHOICES:
        return value
    return None


def _resolve_dataset_name(dataset_name=None):
    normalized = _normalize_dataset_name(dataset_name)
    if normalized is not None:
        return normalized

    env_value = os.environ.get("AUTORESEARCH_DATASET")
    try:
        env_dataset = _normalize_dataset_name(env_value)
    except ValueError:
        print(
            f"Warning: ignoring unsupported AUTORESEARCH_DATASET={env_value!r}; "
            f"using '{DEFAULT_DATASET}'."
        )
        env_dataset = None
    if env_dataset is not None:
        return env_dataset

    file_dataset = _load_active_dataset_from_file()
    if file_dataset is not None:
        return file_dataset

    return DEFAULT_DATASET


def _set_active_dataset(dataset_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(ACTIVE_DATASET_PATH, "w", encoding="utf-8") as f:
        f.write(dataset_name + "\n")


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


def _dataset_root(dataset_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    return os.path.join(DATASETS_DIR, dataset)


def _data_dir(dataset_name=None):
    return os.path.join(_dataset_root(dataset_name), "data")


def _tokenizer_dir(dataset_name=None, tokenizer_name=None):
    name = _resolve_tokenizer_name(tokenizer_name)
    folder = "tokenizer" if name == DEFAULT_TOKENIZER else f"tokenizer-{name}"
    return os.path.join(_dataset_root(dataset_name), folder)


def _tiny_parquet_path(dataset_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    config = DATASET_CONFIGS[dataset]
    return os.path.join(_data_dir(dataset), config["filename"])


def _tiny_legacy_parquet_paths(dataset_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    data_dir = _data_dir(dataset)
    legacy_flat_data_dir = os.path.join(CACHE_DIR, "data")
    return (
        os.path.join(data_dir, "tinystories_gpt4-clean.parquet"),
        os.path.join(legacy_flat_data_dir, "tinystories_gpt4_clean.parquet"),
        os.path.join(legacy_flat_data_dir, "tinystories_gpt4-clean.parquet"),
    )


def _resolve_tiny_parquet_for_read(dataset_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    data_dir = _data_dir(dataset)
    current_path = _tiny_parquet_path(dataset)
    if os.path.exists(current_path):
        return current_path

    for legacy_path in _tiny_legacy_parquet_paths(dataset):
        if not os.path.exists(legacy_path):
            continue
        os.makedirs(data_dir, exist_ok=True)
        try:
            os.replace(legacy_path, current_path)
            print(f"Data: migrated legacy TinyStories parquet to {current_path}")
            return current_path
        except OSError:
            try:
                shutil.copy2(legacy_path, current_path)
                print(f"Data: copied legacy TinyStories parquet to {current_path}")
                return current_path
            except OSError:
                return legacy_path
    return current_path


# ---------------------------------------------------------------------------
# Data download (TinyStories only)
# ---------------------------------------------------------------------------


def _download_tinystories_file(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    data_dir = _data_dir(dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    filename = config["filename"]
    filepath = os.path.join(data_dir, filename)
    resolved_existing_path = _resolve_tiny_parquet_for_read(dataset_name)
    if os.path.exists(resolved_existing_path):
        print(f"Data: {filename} already downloaded at {resolved_existing_path}")
        return

    url = config["url"]
    print(f"Data: downloading {filename}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    temp_path = filepath + ".tmp"
    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.rename(temp_path, filepath)
    print(f"Data: downloaded {filename} to {filepath}")


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


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files(dataset_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    data_dir = _data_dir(dataset)
    files = []
    if os.path.exists(data_dir):
        files = sorted(
            name for name in os.listdir(data_dir)
            if name.endswith(".parquet") and not name.endswith(".tmp")
        )
    if files:
        return [os.path.join(data_dir, name) for name in files]
    if dataset == "tinystories":
        tiny_path = _resolve_tiny_parquet_for_read(dataset)
        if os.path.exists(tiny_path):
            return [tiny_path]
    return []


def _iter_tinystories_texts(split, dataset_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    config = DATASET_CONFIGS[dataset]
    start_idx, end_idx = config["splits"][split]
    tiny_path = _resolve_tiny_parquet_for_read(dataset)

    if not os.path.exists(tiny_path):
        raise FileNotFoundError(
            f"TinyStories parquet not found at {tiny_path}. Run prepare.py first."
        )

    current_idx = 0
    parquet_file = pq.ParquetFile(tiny_path)
    for row_group_idx in range(parquet_file.num_row_groups):
        row_group = parquet_file.read_row_group(row_group_idx, columns=["text"])
        texts = row_group.column("text").to_pylist()
        for text in texts:
            if current_idx < start_idx:
                current_idx += 1
                continue
            if end_idx is not None and current_idx >= end_idx:
                return
            yield text
            current_idx += 1


def text_iterator(dataset_name=None, max_chars=1_000_000_000, doc_cap=10_000):
    dataset = _resolve_dataset_name(dataset_name)
    chars = 0

    text_iter = _iter_tinystories_texts("train", dataset_name=dataset)
    for text in text_iter:
        doc = text[:doc_cap] if len(text) > doc_cap else text
        chars += len(doc)
        yield doc
        if chars >= max_chars:
            return


def train_tokenizer(dataset_name=None, tokenizer_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    tokenizer = _resolve_tokenizer_name(tokenizer_name)
    tokenizer_dir = _tokenizer_dir(dataset, tokenizer)
    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {tokenizer_dir}")
        return

    os.makedirs(tokenizer_dir, exist_ok=True)
    if tokenizer != DEFAULT_TOKENIZER:
        _build_standard_tokenizer(tokenizer, tokenizer_dir, dataset)
        return

    parquet_files = list_parquet_files(dataset)
    if len(parquet_files) < 1:
        print("Tokenizer: TinyStories parquet is missing. Run prepare.py first.")
        raise RuntimeError("TinyStories parquet is missing.")

    print(f"Tokenizer: training BPE tokenizer ({dataset})...")
    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(
        text_iterator(dataset_name=dataset),
        vocab_size_no_special,
        pattern=SPLIT_PATTERN,
    )

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    token_offset = len(mergeable_ranks)
    special_tokens = {name: token_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    with open(os.path.join(tokenizer_dir, "dataset.txt"), "w", encoding="utf-8") as f:
        f.write(dataset + "\n")

    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")


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
    """Bytes of text a SentencePiece-style token stands for (0 for special tokens).

    Known bias: Phi-3 marks the start of every document with a word-boundary "▁", which
    counts as one byte the text does not contain. Measured: val_bpb about 0.12% lower on
    TinyStories and 0.08% on Folktales than the same model would score with exact byte counts.
    """
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


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc, dataset, name=DEFAULT_TOKENIZER):
        self.enc = enc
        self.dataset = _resolve_dataset_name(dataset)
        self.name = name
        self.source = TOKENIZER_SOURCES[name]
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)
        self.eos_token_id = enc.encode_single_token(EOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=None, dataset=None, tokenizer=None):
        dataset_name = _resolve_dataset_name(dataset)
        tokenizer_name = _resolve_tokenizer_name(tokenizer)
        resolved_dir = tokenizer_dir if tokenizer_dir is not None else _tokenizer_dir(dataset_name, tokenizer_name)
        with open(os.path.join(resolved_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc, dataset=dataset_name, name=tokenizer_name)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def get_eos_token_id(self):
        return self.eos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu", dataset=None, tokenizer=None):
    dataset_name = _resolve_dataset_name(dataset)
    path = os.path.join(_tokenizer_dir(dataset_name, tokenizer), "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, dataset=None, tokenizer_batch_size=128):
    dataset_name = _resolve_dataset_name(dataset)
    assert split in ("train", "val", "test")

    epoch = 1
    while True:
        batch = []
        for text in _iter_tinystories_texts(split, dataset_name=dataset_name):
            batch.append(text)
            if len(batch) >= tokenizer_batch_size:
                yield batch, epoch
                batch = []
        if batch:
            yield batch, epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, device="cuda", dataset=None, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    dataset_name = _resolve_dataset_name(dataset or getattr(tokenizer, "dataset", None))
    if split == "test":
        assert dataset_name == "tinystories", "Test split exists only for TinyStories."
    assert split in ("train", "val", "test")

    row_capacity = T + 1
    batches = _document_batches(split, dataset=dataset_name)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1
    resolved_device = torch.device(device)
    use_cuda = resolved_device.type == "cuda"

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        eos_token = tokenizer.get_eos_token_id()
        for row in token_lists:
            row.append(eos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)

    if use_cuda:
        gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=resolved_device)
        inputs = gpu_buffer[:B * T].view(B, T)
        targets = gpu_buffer[B * T:].view(B, T)
    else:
        gpu_buffer = None
        inputs = cpu_inputs
        targets = cpu_targets

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.as_tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.as_tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        if use_cuda:
            gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE METRIC DEFINITION)
# ---------------------------------------------------------------------------

def _check_active_pair(tokenizer):
    """Refuse to score any dataset/tokenizer pair other than the active one.

    train.py is the agent's file and could pass --dataset or another tokenizer to
    Tokenizer.from_directory; val_bpb only compares within one pair, so this check lives
    here, in the read-only scorer. The pair is chosen with prepare.py.
    """
    active_dataset = _resolve_dataset_name(None)
    active_tokenizer = _resolve_tokenizer_name(None)
    dataset = getattr(tokenizer, "dataset", active_dataset)
    name = getattr(tokenizer, "name", DEFAULT_TOKENIZER)
    if (dataset, name) != (active_dataset, active_tokenizer):
        raise RuntimeError(
            f"Refusing to score: this run used dataset '{dataset}' with tokenizer '{name}', but the "
            f"active pair is '{active_dataset}' / '{active_tokenizer}'. Choose the pair with prepare.py."
        )


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, device="cuda", dataset=None, eval_tokens=EVAL_TOKENS):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    """
    _check_active_pair(tokenizer)
    dataset_name = _resolve_dataset_name(dataset or getattr(tokenizer, "dataset", None))
    token_bytes = get_token_bytes(device=device, dataset=dataset_name, tokenizer=getattr(tokenizer, "name", None))
    val_loader = make_dataloader(
        tokenizer,
        batch_size,
        MAX_SEQ_LEN,
        "val",
        device=device,
        dataset=dataset_name,
    )
    steps = max(1, eval_tokens // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    if total_bytes == 0:
        raise RuntimeError("Evaluation produced zero target bytes; cannot compute BPB.")
    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
