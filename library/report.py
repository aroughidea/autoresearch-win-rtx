"""What the explorer cannot compute in a browser, written next to the library.

Run from the repo root once the library's runs exist:  uv run library/report.py
Writes library/index.json (collections and their files), library/tokens.json (each tokenizer's
split of the four prompts, characters per token, unused share) and library/copies.json (spans
of Folktales samples repeated word for word from the training text). Reads the local cache; never
changes the active dataset or tokenizer, so it is safe while a training session runs.
"""
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capture import PROMPTS  # noqa: E402
from prepare import SPLIT_PATTERN, Tokenizer, _iter_tinystories_texts, text_iterator  # noqa: E402

LIBRARY = ROOT / "library"
DATASETS = ("tinystories", "folktales")
TOKENIZERS = ("own", "phi3", "gpt2")
SAMPLE_DOCS = 100_000
COPY_WORDS = 8
# TinyStories' training text is too large to index here; the check found copying in Folktales.
COPY_DATASETS = ("folktales",)
WORD = re.compile(r"[A-Za-z0-9']+")
PARITY_DOCS = 20_000
# tiktoken's split pattern uses possessive quantifiers (?+ ++), which JavaScript lacks. Here they
# match what the plain forms match: the optional character can never be a letter, and the trailing
# group can always match empty. The parity check proves it on real text before any export is written.
HF_PATTERN = SPLIT_PATTERN.replace("?+", "?").replace("++", "+")
# The completion explorables recognise an end token by name; ours are reserved_0 and reserved_1.
# The export names them by CLIP's convention. Ids are unchanged.
CONTROL_NAMES = {"<|reserved_0|>": "<|startoftext|>", "<|reserved_1|>": "<|endoftext|>"}
PARITY_EDGES = ["the thing", "  two  spaces", "line\n\nbreaks\r\n", "it's 1010 they'll", "\u00e9t\u00e9 caf\u00e9",
                "emoji \U0001F642!", "THE The tHe", "a\tb", "  ", "\u4e2d\u6587 \u0440\u0443\u0441"]
# The four prompts split the same way in every tokenizer (plain English all three cover), so
# the Tokens view adds sentences where they differ: TinyStories words, folk-tale words, French,
# and numbers. Each tokenizer is cheapest on the kind of text it was built from.
CONTRASTS = (
    "Lily and her mommy baked yummy cookies.",
    "The vizier's daughter whispered to the sultan.",
    "Il était une fois une princesse.",
    "In 1812, the brothers Grimm printed 86 tales.",
)
SENTENCES = tuple(PROMPTS) + CONTRASTS


def _prefixes(tokenizer, text):
    """Each growing prefix of the token ids, decoded, and whether it ends inside a character.

    Decoding prefixes keeps each token's leading space (Phi-3 drops it from a lone token). A
    byte-level tokenizer can end a token halfway through a character such as "é"; that prefix
    decodes with a replacement mark at the end, which is cut off so the next token takes the
    whole character."""
    ids = tokenizer.encode(text)
    out = []
    for i in range(1, len(ids) + 1):
        current = tokenizer.decode(ids[:i])
        partial = i < len(ids) and current.endswith("�")
        out.append((current.rstrip("�") if partial else current, partial))
    return out


def token_pieces(tokenizer, text):
    """The text cut into its tokens, each piece as it reads in the text. A token holding only
    part of a character gets an empty piece; the token that completes it carries the character."""
    pieces, done = [], 0
    for stable, _ in _prefixes(tokenizer, text):
        pieces.append(stable[done:])
        done = max(done, len(stable))
    return pieces


def partial_tokens(tokenizer, text):
    """Indexes of tokens that end halfway through a character."""
    return [i for i, (_, partial) in enumerate(_prefixes(tokenizer, text)) if partial]


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
    """Every folder under the library with a collection.json: title, kind, run files, scoreboard, and
    any notes naming particular runs (a study's runs share one commit, so only these tell them apart)."""
    out = []
    for meta_path in sorted(Path(library).rglob("collection.json")):
        folder = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        results = folder / "results.tsv"
        runs = sorted(p.relative_to(library).as_posix() for p in (folder / "runs").glob("*.json"))
        entry = {
            "id": folder.relative_to(library).as_posix(),
            "title": meta["title"],
            "kind": meta["kind"],
            "note": meta.get("note", ""),
            "runs": runs,
            "results": results.relative_to(library).as_posix() if results.exists() else None,
        }
        notes = meta.get("run_notes") or {}
        by_stem = {Path(rel).stem: rel for rel in runs}
        unknown = sorted(set(notes) - set(by_stem))
        if unknown:
            raise ValueError(f"{meta_path}: run_notes names runs that do not exist: {', '.join(unknown)}")
        if notes:
            entry["run_notes"] = {by_stem[stem]: text for stem, text in sorted(notes.items())}
        out.append(entry)
    return out


def load_runs(library=LIBRARY):
    runs = {}
    for c in collections(library):
        for rel in c["runs"]:
            runs[rel] = json.loads((Path(library) / rel).read_text(encoding="utf-8"))
    return runs


def train_token_estimate(chars, docs, chars_per_token):
    """Tokens in one pass over the training split: its text, plus a start and an end marker per document."""
    return round(chars / chars_per_token + 2 * docs)


def train_split_size(dataset):
    """Characters and documents in the training split, read the way the data loader reads it."""
    chars = docs = 0
    for text in _iter_tinystories_texts("train", dataset_name=dataset):
        chars += len(text)
        docs += 1
    return chars, docs


def bytes_to_unicode():
    """GPT-2's byte alphabet: each byte as a printable character."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) + list(range(ord("\u00ae"), ord("\u00ff") + 1))
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
    """The merge list a BPE model needs, recovered from tiktoken's ranks: each multi-byte token is
    the merge of the two parts BPE reaches using only lower-ranked merges."""
    merges = []
    for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(token) > 1:
            parts = _bpe_parts(ranks, token, rank)
            assert len(parts) == 2, token
            merges.append((parts[0], parts[1]))
    return merges


def hf_tokenizer_json(ranks, specials, pattern):
    """A Hugging Face tokenizer.json equivalent to a tiktoken encoding (byte-level BPE)."""
    alphabet = bytes_to_unicode()

    def show(bs):
        return "".join(alphabet[b] for b in bs)

    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": i, "content": CONTROL_NAMES.get(name, name), "single_word": False, "lstrip": False,
                          "rstrip": False, "normalized": False, "special": True}
                         for name, i in sorted(specials.items(), key=lambda kv: kv[1])],
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


def export_own_tokenizers(datasets=DATASETS, out_dir=None):
    """Each dataset's own vocabulary as tokenizer.json, written only after it tokenizes PARITY_DOCS
    real documents and PARITY_EDGES exactly as our tiktoken encoding does."""
    from tokenizers import Tokenizer as HFTokenizer

    out_dir = out_dir or (LIBRARY / "tokenizers")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for dataset in datasets:
        enc = Tokenizer.from_directory(dataset=dataset, tokenizer="own").enc
        doc = hf_tokenizer_json(enc._mergeable_ranks, enc._special_tokens, HF_PATTERN)
        text = json.dumps(doc, ensure_ascii=False)
        hf = HFTokenizer.from_str(text)
        samples = PARITY_EDGES + list(itertools.islice(text_iterator(dataset), PARITY_DOCS))
        bad = [t for t in samples if hf.encode(t, add_special_tokens=False).ids != enc.encode_ordinary(t)]
        if bad:
            raise RuntimeError(f"{dataset}: the export tokenizes {len(bad)} of {len(samples)} texts differently, "
                               f"e.g. {bad[0][:80]!r}; nothing written")
        key = f"{dataset}-own"
        path = out_dir / f"{key}.json"
        path.write_text(text, encoding="utf-8")
        manifest[key] = {"file": f"tokenizers/{key}.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "size": enc.n_vocab}
        print(f"export: {key}: identical on {len(samples)} texts; {path.stat().st_size // 1024} KB")
    return manifest


def tokens_report():
    out = {"prompts": list(PROMPTS), "sentences": list(SENTENCES), "sample_docs": SAMPLE_DOCS, "datasets": {}}
    for dataset in DATASETS:
        docs = list(itertools.islice(text_iterator(dataset), SAMPLE_DOCS))
        train_chars, train_docs = train_split_size(dataset)
        out["datasets"][dataset] = {}
        for name in TOKENIZERS:
            tok = Tokenizer.from_directory(dataset=dataset, tokenizer=name)
            out["datasets"][dataset][name] = {
                "vocab_size": tok.get_vocab_size(),
                "source": tok.source,
                "splits": [token_pieces(tok, p) for p in SENTENCES],
                "partial": [partial_tokens(tok, p) for p in SENTENCES],
                **vocab_stats(tok, docs),
            }
            entry = out["datasets"][dataset][name]
            entry["train_tokens"] = train_token_estimate(train_chars, train_docs, entry["chars_per_token"])
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
    index = {"schema": 1, "collections": collections(), "tokens": "tokens.json", "copies": "copies.json",
             "tokenizers": export_own_tokenizers()}
    _write("index.json", index)
    _write("tokens.json", tokens_report())
    _write("copies.json", copies_report(load_runs()))
    print(f"index: {len(index['collections'])} collections, {sum(len(c['runs']) for c in index['collections'])} runs")


if __name__ == "__main__":
    main()
