from types import SimpleNamespace

import train


def _runtime():
    return SimpleNamespace(gpu_name="NVIDIA RTX 4000 Ada Generation Laptop GPU", gpu_cc=(8, 9),
                           gpu_total_memory_bytes=12 * 1024 ** 3)


def test_autotune_cache_key_depends_on_vocab_size():
    """A batch size tuned for an 8,192-token vocabulary must not be reused for 32,015 tokens:
    on Windows the larger model silently spills past GPU memory (13 GB on a 12 GB card)."""
    assert train._make_autotune_cache_key(_runtime(), 8192) != train._make_autotune_cache_key(_runtime(), 32015)


def test_autotune_cache_key_is_stable_for_the_same_vocab():
    assert train._make_autotune_cache_key(_runtime(), 8192) == train._make_autotune_cache_key(_runtime(), 8192)


def _config(**changes):
    shape = dict(sequence_len=train.MAX_SEQ_LEN, vocab_size=8192, n_layer=6, n_head=3, n_kv_head=1, n_embd=384,
                 window_pattern="SSSS", attention_backend="sdpa", use_activation_checkpointing=False,
                 compute_dtype=train.torch.float32)
    return train.GPTConfig(**{**shape, **changes})


def test_model_fingerprint_changes_when_the_agent_deepens_the_model():
    """The agent raising DEPTH must not reuse a batch size tuned for the smaller model:
    on Windows the bigger one silently spills past GPU memory and scores worse for no visible reason."""
    small = train._model_fingerprint(_config())
    deeper = train._model_fingerprint(_config(n_layer=8, n_embd=512, n_head=4))
    assert small != deeper
    key = lambda fp: train._make_autotune_cache_key(_runtime(), 8192, fp)
    assert key(small) != key(deeper)


def test_model_fingerprint_sees_changes_outside_the_config(monkeypatch):
    """A wider MLP edited into the model code changes no config field, only the parameter count."""
    before = train._model_fingerprint(_config())
    original_init = train.GPT.__init__

    def wider(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        self.extra = train.nn.Linear(config.n_embd, config.n_embd * 4, bias=False)

    monkeypatch.setattr(train.GPT, "__init__", wider)
    assert train._model_fingerprint(_config()) != before


def test_model_fingerprint_ignores_activation_checkpointing():
    """Checkpointing is one of the candidates being tuned, not part of the model."""
    assert train._model_fingerprint(_config()) == train._model_fingerprint(_config(use_activation_checkpointing=True))
