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


def test_tokens_sentences_add_contrasts_after_the_prompts():
    """The four prompts split the same way in every tokenizer; the contrasts are where they differ."""
    assert report.SENTENCES[:4] == tuple(report.PROMPTS)
    assert len(report.SENTENCES) == 8
