import torch

from lion_pytorch.cautious_lion import Lion


def _expected_step(parameter, gradient, exp_avg, *, lr, beta1, beta2, factor, weight_decay=0.0, cautious_wd=False):
    mixed = exp_avg * beta1 + gradient * (1.0 - beta1)
    update = torch.sgn(mixed)

    alignment = (torch.conj(update) * gradient).real > 0
    scale = alignment.to(dtype=gradient.real.dtype)
    scale = scale + (1.0 - scale) * factor
    scale /= scale.mean().clamp(min=1e-5)
    update = update * scale

    if cautious_wd:
        decay_mask = (torch.conj(update) * parameter).real > 0
        decay_mask = decay_mask.to(dtype=parameter.real.dtype)
    else:
        decay_mask = 1.0

    parameter = parameter * (1.0 - lr * weight_decay * decay_mask)
    parameter = parameter - lr * update
    exp_avg = exp_avg * beta2 + gradient * (1.0 - beta2)
    return parameter, exp_avg


def _run_step(parameter, gradient, exp_avg, **kwargs):
    parameter = torch.nn.Parameter(parameter.clone())
    optimizer = Lion([parameter], **kwargs)
    optimizer.state[parameter]["exp_avg"] = exp_avg.clone()
    parameter.grad = gradient.clone()
    optimizer.step()
    return parameter.detach(), optimizer.state[parameter]["exp_avg"]


def test_complex_cautious_update_uses_hermitian_alignment():
    parameter = torch.tensor([1.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64)
    gradient = torch.tensor([1.0j, -1.0 - 1.0j], dtype=torch.complex64)
    exp_avg = torch.tensor([1.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64)
    options = dict(lr=0.1, betas=(0.9, 0.99), cautious_factor=0.5)

    actual_parameter, actual_exp_avg = _run_step(parameter, gradient, exp_avg, **options)
    expected_parameter, expected_exp_avg = _expected_step(
        parameter,
        gradient,
        exp_avg,
        lr=0.1,
        beta1=0.9,
        beta2=0.99,
        factor=0.5,
    )

    assert torch.isfinite(actual_parameter.real).all()
    assert torch.isfinite(actual_parameter.imag).all()
    assert torch.allclose(actual_parameter, expected_parameter, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_exp_avg, expected_exp_avg, atol=1e-6, rtol=1e-6)


def test_complex_cautious_weight_decay_uses_hermitian_alignment():
    parameter = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j], dtype=torch.complex64)
    gradient = torch.tensor([1.0j, -1.0j], dtype=torch.complex64)
    exp_avg = torch.tensor([1.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64)
    options = dict(
        lr=0.1,
        betas=(0.9, 0.99),
        cautious_factor=0.5,
        weight_decay=0.2,
        cautious_wd=True,
    )

    actual_parameter, actual_exp_avg = _run_step(parameter, gradient, exp_avg, **options)
    expected_parameter, expected_exp_avg = _expected_step(
        parameter,
        gradient,
        exp_avg,
        lr=0.1,
        beta1=0.9,
        beta2=0.99,
        factor=0.5,
        weight_decay=0.2,
        cautious_wd=True,
    )

    assert torch.isfinite(actual_parameter.real).all()
    assert torch.isfinite(actual_parameter.imag).all()
    assert torch.allclose(actual_parameter, expected_parameter, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_exp_avg, expected_exp_avg, atol=1e-6, rtol=1e-6)


def test_real_cautious_update_matches_elementwise_alignment():
    parameter = torch.tensor([1.0, 1.0], dtype=torch.float32)
    gradient = torch.tensor([1.0, -1.0], dtype=torch.float32)
    exp_avg = torch.tensor([1.0, 1.0], dtype=torch.float32)
    options = dict(lr=0.1, betas=(0.9, 0.99), cautious_factor=0.5)

    actual_parameter, actual_exp_avg = _run_step(parameter, gradient, exp_avg, **options)
    expected_parameter, expected_exp_avg = _expected_step(
        parameter,
        gradient,
        exp_avg,
        lr=0.1,
        beta1=0.9,
        beta2=0.99,
        factor=0.5,
    )

    assert torch.allclose(actual_parameter, expected_parameter, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_exp_avg, expected_exp_avg, atol=1e-6, rtol=1e-6)
