"""chat.py rebuilds each saved model with the code that trained it: the version the record kept, else the
recipe in its run file, and only then today's train.py."""
import torch

import chat
import train

FAKE_VERSION = '''
from __future__ import annotations   # as train.py: dataclasses then look the module up by name

from dataclasses import dataclass

WINDOW_PATTERN = "L"


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 3
    n_kv_head: int = 1
    sequence_len: int = 2048
    window_pattern: str = "SSSL"
    use_activation_checkpointing: bool = False


class GPT:
    """Stands in for the version's network: it only remembers how it was built."""
    from_version = True

    def __init__(self, config):
        self.config = config

    def load_state_dict(self, state_dict):
        self.loaded = True

    def to(self, device):
        return self

    def eval(self):
        return self
'''


def _state_dict(vocab=16, n_embd=256, n_layer=2, n_kv_head=2, head_dim=128):
    """The weights _config_from_state_dict reads shapes from."""
    return {
        "transformer.wte.weight": torch.zeros(vocab, n_embd),
        "resid_lambdas": torch.zeros(n_layer),
        "transformer.h.1.attn.ve_gate.weight": torch.zeros(n_kv_head, 32),
        "transformer.h.0.attn.c_k.weight": torch.zeros(n_kv_head * head_dim, n_embd),
    }


def _project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    versions = tmp_path / "sessions" / "s1" / "versions"
    versions.mkdir(parents=True)
    (versions / "abc123def456.py").write_text(FAKE_VERSION, encoding="utf-8")
    path = tmp_path / "checkpoints" / "20261012T182536-0700_abc123def456.pt"
    path.parent.mkdir()
    torch.save(_state_dict(), path)
    return path


def test_a_model_is_rebuilt_with_the_code_that_trained_it(tmp_path, monkeypatch):
    path = _project(tmp_path, monkeypatch)
    model = chat._load_model_from_checkpoint(str(path), "cpu", {"version": "abc123def456"})
    assert getattr(model, "from_version", False) and model.loaded
    assert model.config.window_pattern == "L"                  # the version's own windows
    assert (model.config.n_kv_head, model.config.n_head, model.config.n_layer) == (2, 2, 2)


def test_without_the_version_the_run_files_recipe_gives_the_windows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, config = chat._code_and_config(_state_dict(), {"recipe": {"WINDOW_PATTERN": "SSSL"}})
    assert code is train and config.window_pattern == "SSSL"


def test_with_no_record_at_all_todays_train_py_gives_the_windows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, config = chat._code_and_config(_state_dict(), None)
    assert code is train and config.window_pattern == train.WINDOW_PATTERN


def test_a_version_named_in_the_run_but_missing_falls_back_to_the_recipe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = {"version": "000000000000", "recipe": {"WINDOW_PATTERN": "SSLL"}}
    code, config = chat._code_and_config(_state_dict(), run)
    assert code is train and config.window_pattern == "SSLL"
