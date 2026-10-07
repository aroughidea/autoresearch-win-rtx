import torch

from train import LossScaler


def _param_with_grad(values, dtype=torch.float32):
    p = torch.nn.Parameter(torch.zeros(len(values), dtype=dtype))
    p.grad = torch.tensor(values, dtype=dtype)
    return p


def test_disabled_scaler_leaves_backward_and_grads_alone():
    scaler = LossScaler(enabled=False)
    w = torch.nn.Parameter(torch.tensor([2.0]))
    scaler.backward((w * 3.0).sum())
    assert w.grad.item() == 3.0
    assert scaler.unscale_and_check([w]) is True
    assert w.grad.item() == 3.0


def test_enabled_scaler_returns_true_gradients():
    scaler = LossScaler(enabled=True, init_scale=1024.0)
    w = torch.nn.Parameter(torch.tensor([2.0]))
    scaler.backward((w * 3.0).sum())
    assert w.grad.item() == 3.0 * 1024.0
    assert scaler.unscale_and_check([w]) is True
    assert w.grad.item() == 3.0


def test_small_fp16_gradient_survives_with_scaling():
    """The bug: 1e-8 is below fp16's smallest subnormal and becomes 0 without scaling."""
    tiny = 1e-8
    unscaled = torch.tensor([tiny], dtype=torch.float32).to(torch.float16)
    assert unscaled.item() == 0.0
    scaler = LossScaler(enabled=True, init_scale=4096.0)
    p = _param_with_grad([tiny * 4096.0], dtype=torch.float16)  # what a scaled fp16 backward produces
    assert p.grad.item() != 0.0
    assert scaler.unscale_and_check([p]) is True


def test_overflow_skips_step_halves_scale_and_clears_grads():
    scaler = LossScaler(enabled=True, init_scale=4096.0)
    p = _param_with_grad([1.0, float("inf")])
    assert scaler.unscale_and_check([p]) is False
    assert scaler.scale == 2048.0
    assert p.grad is None


def test_nan_gradient_also_skips_step():
    scaler = LossScaler(enabled=True, init_scale=4096.0)
    p = _param_with_grad([float("nan")])
    assert scaler.unscale_and_check([p]) is False


def test_scale_grows_after_a_run_of_good_steps():
    scaler = LossScaler(enabled=True, init_scale=1024.0, growth_interval=3)
    for _ in range(3):
        assert scaler.unscale_and_check([_param_with_grad([1.0])]) is True
    assert scaler.scale == 2048.0


def test_params_without_grads_are_ignored():
    scaler = LossScaler(enabled=True)
    p = torch.nn.Parameter(torch.zeros(2))
    assert scaler.unscale_and_check([p]) is True
