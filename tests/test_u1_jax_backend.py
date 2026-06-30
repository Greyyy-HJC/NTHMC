import jax
import jax.numpy as jnp
import numpy as np

from nthmc.u1.field_transform import FieldTransformation
from nthmc.u1.u1_fthmc import build_fthmc_chain
from nthmc.u1.u1_observables import action, force, plaq_from_field_batch, topo_from_field


def test_u1_observables_shapes_and_values() -> None:
    theta = jnp.zeros((2, 2, 4, 4), dtype=jnp.float32)
    plaq = plaq_from_field_batch(theta)
    assert plaq.shape == (2, 4, 4)
    assert np.allclose(np.asarray(plaq), 0.0)
    assert float(topo_from_field(theta[0])) == 0.0
    assert float(action(theta[0], 3.0)) == -48.0


def test_u1_force_finite_difference() -> None:
    theta = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (2, 4, 4))
    direction = jax.random.normal(jax.random.PRNGKey(2), theta.shape)
    direction = direction / jnp.linalg.norm(direction)
    eps = 1e-3
    finite_difference = (action(theta + eps * direction, 2.0) - action(theta - eps * direction, 2.0)) / (2 * eps)
    directional = jnp.sum(force(theta, 2.0) * direction)
    assert np.allclose(float(directional), float(finite_difference), rtol=2e-2, atol=2e-3)


def test_u1_field_transform_jacobian_and_chain_smoke() -> None:
    transform = FieldTransformation(4, model_tag="base", n_subsets=8)
    theta = 0.1 * jax.random.normal(jax.random.PRNGKey(3), (2, 4, 4))
    transformed = transform.field_transformation(theta)
    logdet = transform.compute_jac_logdet(theta[jnp.newaxis, ...])
    assert transformed.shape == theta.shape
    assert logdet.shape == (1,)
    assert jnp.isfinite(logdet).all()

    chain = build_fthmc_chain(transform, beta=2.0, n_thermalization=1, n_configs=2, n_steps=1, step_size=0.05)
    result = chain(jax.random.PRNGKey(4))
    assert result.plaq.shape == (2,)
    assert 0.0 <= float(result.acceptance_rate) <= 1.0
