import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from nthmc.u1 import jax_backend as jb
from nthmc.u1.field_transform import FieldTransformation
from nthmc.u1.u1_hmc import HMCU1
from nthmc.u1.u1_observables import plaq_from_field_batch, topo_from_field


def _make_transform(lattice_size: int = 4, model_tag: str = "addcos") -> FieldTransformation:
    torch.manual_seed(123)
    transform = FieldTransformation(
        lattice_size,
        device="cpu",
        n_subsets=8,
        identity_init=True,
        model_tag=model_tag,
        compile_enabled=False,
    )
    for model in transform.models:
        model.eval()
    return transform


def test_jax_u1_observables_match_torch() -> None:
    torch.manual_seed(1)
    theta = torch.randn(2, 2, 4, 4)
    theta_jax = jnp.asarray(theta.numpy())

    torch_plaq = plaq_from_field_batch(theta)
    jax_plaq = jb.plaq_from_field_batch(theta_jax)
    assert np.allclose(np.asarray(jax_plaq), torch_plaq.numpy(), atol=1e-6)

    torch_topo = topo_from_field(theta[0])
    jax_topo = jb.topo_from_field(theta_jax[0])
    assert np.allclose(np.asarray(jax_topo), torch_topo.numpy(), atol=1e-6)


def test_jax_u1_force_matches_torch_autograd() -> None:
    torch.manual_seed(2)
    beta = 3.0
    theta = torch.randn(2, 4, 4)
    hmc = HMCU1(4, beta, 0, 1, 0.1)
    torch_force = hmc.force(theta)

    jax_force = jb.force(jnp.asarray(theta.numpy()), beta)
    assert np.allclose(np.asarray(jax_force), torch_force.numpy(), atol=2e-5)


@pytest.mark.parametrize("model_tag", ["base", "addcos"])
def test_jax_u1_field_transform_matches_torch_forward_and_jacobian(model_tag: str) -> None:
    torch.manual_seed(3)
    transform = _make_transform(model_tag=model_tag)
    params = jb.torch_field_transform_to_jax_params(transform)
    jax_transform = jb.JaxU1FieldTransformation(params, lattice_size=4, n_subsets=8)
    theta = torch.randn(2, 4, 4)

    with torch.no_grad():
        torch_forward = transform.field_transformation(theta)
        torch_jac = transform.compute_jac_logdet(theta.unsqueeze(0))

    jax_forward = jax_transform.field_transformation(jnp.asarray(theta.numpy()))
    jax_jac = jax_transform.compute_jac_logdet(jnp.asarray(theta.unsqueeze(0).numpy()))

    assert np.allclose(np.asarray(jax_forward), torch_forward.numpy(), atol=3e-5)
    assert np.allclose(np.asarray(jax_jac), torch_jac.numpy(), atol=3e-5)


def test_jax_fthmc_chain_runs_and_reports_acceptance() -> None:
    transform = _make_transform(lattice_size=4, model_tag="base")
    params = jb.torch_field_transform_to_jax_params(transform)
    jax_transform = jb.JaxU1FieldTransformation(params, lattice_size=4, n_subsets=8)
    chain = jb.build_fthmc_chain(
        jax_transform,
        beta=3.0,
        n_thermalization=2,
        n_configs=4,
        n_steps=1,
        step_size=0.05,
    )

    result = jax.jit(chain)(jax.random.PRNGKey(7))
    acceptance = float(np.asarray(result.acceptance_rate))
    therm_acceptance = float(np.asarray(result.therm_acceptance_rate))

    assert result.plaq.shape == (4,)
    assert result.topo.shape == (4,)
    assert np.isfinite(np.asarray(result.plaq)).all()
    assert 0.0 <= acceptance <= 1.0
    assert 0.0 <= therm_acceptance <= 1.0
