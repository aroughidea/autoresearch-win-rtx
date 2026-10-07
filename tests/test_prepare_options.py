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
    build = prepare._gpt2_encoding  # a missing function must fail the test, not skip it
    try:
        enc = build()
    except OSError as exc:  # the GPT-2 vocabulary downloads once; skip offline
        pytest.skip(f"GPT-2 vocabulary unavailable: {exc}")
    assert enc.n_vocab == 50257 + len(prepare.SPECIAL_TOKENS)
    assert enc.encode_single_token(prepare.BOS_TOKEN) == 50257
    ids = enc.encode_ordinary("Once upon a time")
    assert len(ids) == 4 and enc.decode(ids) == "Once upon a time"
