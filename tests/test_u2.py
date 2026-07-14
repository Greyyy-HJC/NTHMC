import jax
import jax.numpy as jnp
import numpy as np

from nthmc.u2.field_transform import FieldTransformation
from nthmc.u2.u2_observables import (
    action_from_field,
    force_from_field,
    identity_field,
    plaquette_mean_from_field,
    topology_from_field,
    u2_exp,
    u2_mul,
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
    links = u2_exp(0.05 * jax.random.normal(jax.random.PRNGKey(1), (2, 2, 2, 4)))
    direction = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 2, 4))
    direction = direction / jnp.linalg.norm(direction)
    eps = 1e-3

    def potential(delta):
        return action_from_field(u2_mul(u2_exp(delta), links), 2.0)

    finite_difference = (potential(eps * direction) - potential(-eps * direction)) / (2 * eps)
    directional = jnp.sum(force_from_field(links, 2.0) * direction)
    assert np.allclose(float(directional), float(finite_difference), rtol=5e-2, atol=5e-3)


def test_u2_field_transform_identity_jac() -> None:
    transform = FieldTransformation(2, model_tag="base", n_subsets=1)
    assert transform.params["subsets"][0]["conv_input"]["weight"].shape[1] == 18
    links = identity_field(2)
    assert np.allclose(np.asarray(transform.field_transformation(links)), np.asarray(links))
    assert np.allclose(np.asarray(transform.compute_jac_logdet(links[jnp.newaxis, ...])), 0.0)


def test_u2_nontrivial_field_transform_analytic_jacobian() -> None:
    transform = FieldTransformation(2, model_tag="base", n_subsets=1)
    subset = dict(transform.params["subsets"][0])
    subset["out_scale"] = jnp.ones_like(subset["out_scale"])
    subset["conv_output"] = dict(subset["conv_output"])
    subset["conv_output"]["bias"] = jnp.ones_like(subset["conv_output"]["bias"]) * 0.05
    transform.params = {"subsets": [subset]}

    links = u2_exp(0.03 * jax.random.normal(jax.random.PRNGKey(3), (1, 2, 2, 2, 4)))
    transformed = transform.forward_with_params(transform.params, links)
    logdet = transform.compute_jac_logdet_with_params(transform.params, links)
    inverted, diagnostics = transform.inverse(transformed, max_iter=8, tol=1e-4, return_diagnostics=True)

    assert not np.allclose(np.asarray(transformed), np.asarray(links))
    assert np.all(np.isfinite(np.asarray(logdet)))
    assert np.allclose(np.asarray(transform.forward_with_params(transform.params, inverted)), np.asarray(transformed), atol=1e-4)
    assert float(diagnostics["max_final_diff"]) < 1e-4
    assert 0 < float(diagnostics["mean_iterations"]) <= 8
    assert 0 < float(diagnostics["max_iterations"]) <= 8
    assert int(diagnostics["n_not_converged"]) == 0


def test_u2_autodiff_jacobian_is_check_only() -> None:
    links = u2_exp(0.03 * jax.random.normal(jax.random.PRNGKey(6), (1, 2, 2, 2, 4)))
    transform = FieldTransformation(2, model_tag="base", n_subsets=1, if_check_jac=False)

    def fail_autodiff(*_args, **_kwargs):
        raise AssertionError("autodiff Jacobian should not run when if_check_jac=False")

    transform.compute_jac_logdet_autodiff_with_params = fail_autodiff
    transform._check_jacobian_if_requested(transform.params, links)

    checked = {"called": False}
    transform.if_check_jac = True

    def fake_autodiff(params, batch):
        checked["called"] = True
        return transform.compute_jac_logdet_with_params(params, batch)

    transform.compute_jac_logdet_autodiff_with_params = fake_autodiff
    transform._check_jacobian_if_requested(transform.params, links)
    assert checked["called"]
