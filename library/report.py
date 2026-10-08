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
    out = {"prompts": list(PROMPTS), "sentences": list(SENTENCES), "sample_docs": SAMPLE_DOCS, "datasets": {}}
    for dataset in DATASETS:
        docs = list(itertools.islice(text_iterator(dataset), SAMPLE_DOCS))
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
