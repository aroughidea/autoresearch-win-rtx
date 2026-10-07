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
