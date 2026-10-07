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
