import torch

from nthmc.u2.training import FieldTransformation
from nthmc.u2.training_observables import u2_conj, u2_exp, u2_log, u2_mul


def _nontrivial_transform(*, dtype: torch.dtype) -> FieldTransformation:
    transform = FieldTransformation(2, n_subsets=8, model_tag="base")
    generator = torch.Generator().manual_seed(314159)

    for model in transform.models:
        model.to(dtype=dtype)
        with torch.no_grad():
            for parameter in (
                model.conv_input.weight,
                model.conv_input.bias,
                model.conv_output.weight,
                model.conv_output.bias,
            ):
                std = 0.08 if parameter.ndim > 1 else 0.03
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=dtype) * std)
            model.out_scale.scale.fill_(0.5)

    return transform


def _full_field_logdet(transform: FieldTransformation, links: torch.Tensor) -> torch.Tensor:
    base_output = transform.forward(links).detach()
    algebra_shape = (*links.shape[:-1], 4)

    def perturbed_forward(algebra_flat: torch.Tensor) -> torch.Tensor:
        algebra = algebra_flat.reshape(algebra_shape)
        varied_links = u2_mul(u2_exp(algebra), links)
        varied_output = transform.forward(varied_links)
        relative_output = u2_mul(varied_output, u2_conj(base_output))
        return u2_log(relative_output).reshape(-1)

    algebra = torch.zeros(algebra_shape, dtype=links.dtype, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(
        perturbed_forward,
        algebra.reshape(-1),
        vectorize=True,
    )
    sign, logabsdet = torch.linalg.slogdet(jacobian)
    assert sign > 0
    return logabsdet


def test_u2_manual_logdet_matches_full_field_jacobian_in_float64() -> None:
    transform = _nontrivial_transform(dtype=torch.float64)
    generator = torch.Generator().manual_seed(271828)
    links = u2_exp(0.15 * torch.randn((1, 2, 2, 2, 4), generator=generator, dtype=torch.float64))

    manual = transform.compute_jac_logdet_manual(links)[0]
    full_autograd = _full_field_logdet(transform, links)

    assert torch.abs(manual) > 1e-3
    assert torch.allclose(manual, full_autograd, rtol=1e-10, atol=1e-10)
