import jax
import jax.numpy as jnp
import numpy as np

from nthmc.u2.field_transform import FieldTransformation
from nthmc.u2.u2_hmc import HMCU2
from nthmc.u2.u2_observables import (
    action_from_field,
    force_from_field,
    identity_field,
    plaquette_mean_from_field,
    topology_from_field,
    u2_exp,
    u2_to_matrix,
)


def test_u2_identity_and_matrix_unitarity() -> None:
    links = identity_field(4)
    matrices = u2_to_matrix(links)
    eye = jnp.eye(2, dtype=matrices.dtype)
    assert np.allclose(np.asarray(matrices.conj().swapaxes(-1, -2) @ matrices), np.asarray(eye), atol=1e-5)
    assert np.allclose(float(plaquette_mean_from_field(links)), 1.0)
    assert float(topology_from_field(links)) == 0.0
    assert np.allclose(float(action_from_field(links, 2.0)), 0.0)


def test_u2_force_finite_difference() -> None:
    links = u2_exp(0.05 * jax.random.normal(jax.random.PRNGKey(1), (2, 3, 3, 4)))
    direction = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 3, 4))
    direction = direction / jnp.linalg.norm(direction)
    eps = 1e-3

    def varied_action(delta):
        return action_from_field(u2_exp(delta) * 0 + u2_exp(delta), 2.0)

    def potential(delta):
        from nthmc.u2.u2_observables import u2_mul

        return action_from_field(u2_mul(u2_exp(delta), links), 2.0)

    finite_difference = (potential(eps * direction) - potential(-eps * direction)) / (2 * eps)
    directional = jnp.sum(force_from_field(links, 2.0) * direction)
    assert np.allclose(float(directional), float(finite_difference), rtol=5e-2, atol=5e-3)


def test_u2_field_transform_and_hmc_smoke() -> None:
    transform = FieldTransformation(3, model_tag="base")
    links = identity_field(3)
    assert np.allclose(np.asarray(transform.field_transformation(links)), np.asarray(links))
    assert np.allclose(np.asarray(transform.compute_jac_logdet(links[jnp.newaxis, ...])), 0.0)
    hmc = HMCU2(3, beta=2.0, n_thermalization_steps=1, n_steps=1, step_size=0.02, tune_step_size=False)
    therm, _, _ = hmc.thermalize()
    _, plaq, acc, topo, _ = hmc.run(2, therm, save_config=False)
    assert len(plaq) == 2
    assert 0.0 <= acc <= 1.0
