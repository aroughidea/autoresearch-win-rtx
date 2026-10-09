"""library/report.py: token pieces, vocabulary statistics, copied spans, the collection index."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

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


def test_a_collection_can_name_some_of_its_runs(tmp_path):
    """A study's runs share a commit, so collection.json says which run is which; a note for a run
    that does not exist is a typo, and stops the report."""
    lib = tmp_path / "library"
    (lib / "study" / "runs").mkdir(parents=True)
    for name in ("folktales-own", "folktales-own-agent-best"):
        (lib / "study" / "runs" / f"{name}.json").write_text("{}", encoding="utf-8")
    meta = {"title": "Study", "kind": "study", "run_notes": {"folktales-own-agent-best": "the agent's best recipe"}}
    (lib / "study" / "collection.json").write_text(json.dumps(meta), encoding="utf-8")
    [c] = report.collections(lib)
    assert c["run_notes"] == {"study/runs/folktales-own-agent-best.json": "the agent's best recipe"}
    meta["run_notes"]["folktales-own-agent-bset"] = "typo"
    (lib / "study" / "collection.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="folktales-own-agent-bset"):
        report.collections(lib)


def test_tokens_sentences_add_contrasts_after_the_prompts():
    """The four prompts split the same way in every tokenizer; the contrasts are where they differ."""
    assert report.SENTENCES[:4] == tuple(report.PROMPTS)
    assert len(report.SENTENCES) == 8


class BytesTok:
    """One token per UTF-8 byte, like a byte-level tokenizer splitting an accented letter."""
    source = "bytes"

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def get_vocab_size(self):
        return 256


def test_token_pieces_rebuild_text_when_a_token_splits_a_character():
    pieces = report.token_pieces(BytesTok(), "é!")
    assert "".join(pieces) == "é!"
    assert pieces == ["", "é", "!"]
    assert report.partial_tokens(BytesTok(), "é!") == [0]
    assert report.partial_tokens(FakeTok(), "Once upon a time") == []


def test_train_token_estimate_counts_the_document_markers():
    assert report.train_token_estimate(chars=4000, docs=10, chars_per_token=4.0) == 1020


def _tiny_encoding():
    import tiktoken
    import prepare
    ranks = {bytes([b]): b for b in range(256)}
    for merged in (b"th", b"the", b" t", b" the", b"in", b"ing", b"10"):
        ranks[merged] = len(ranks)
    specials = {"<|reserved_0|>": len(ranks), "<|reserved_1|>": len(ranks) + 1}
    return tiktoken.Encoding(name="tiny", pat_str=prepare.SPLIT_PATTERN, mergeable_ranks=ranks, special_tokens=specials)


def test_hf_export_tokenizes_exactly_like_tiktoken():
    from tokenizers import Tokenizer as HFTokenizer
    enc = _tiny_encoding()
    hf = HFTokenizer.from_str(json.dumps(report.hf_tokenizer_json(enc._mergeable_ranks, enc._special_tokens, report.HF_PATTERN)))
    for text in ["the thing", "  two  spaces", "line\n\nbreaks\r\n", "it's 1010 they'll", "été café",
                 "emoji \U0001F642!", "the.the,the", "THE The tHe", "a\tb", "  "]:
        assert hf.encode(text, add_special_tokens=False).ids == enc.encode_ordinary(text), text


def test_hf_export_names_the_control_tokens_by_clips_convention():
    enc = _tiny_encoding()
    added = report.hf_tokenizer_json(enc._mergeable_ranks, enc._special_tokens, report.HF_PATTERN)["added_tokens"]
    assert [(a["id"], a["content"]) for a in added] == [(263, "<|startoftext|>"), (264, "<|endoftext|>")]
