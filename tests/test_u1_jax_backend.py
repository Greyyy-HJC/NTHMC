import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    assert np.allclose(np.asarray(transformed), np.asarray(theta))
    assert logdet.shape == (1,)
    assert np.allclose(np.asarray(logdet), 0.0)
    assert jnp.isfinite(logdet).all()

    chain = build_fthmc_chain(transform, beta=2.0, n_thermalization=1, n_configs=2, n_steps=1, step_size=0.05)
    result = chain(jax.random.PRNGKey(4))
    assert result.plaq.shape == (2,)
    assert 0.0 <= float(result.acceptance_rate) <= 1.0


def test_u1_inverse_uses_tolerance_diagnostics() -> None:
    transform = FieldTransformation(4, model_tag="base", n_subsets=1)
    subset = dict(transform.params["subsets"][0])
    subset["out_scale"] = jnp.ones_like(subset["out_scale"])
    subset["conv_output"] = dict(subset["conv_output"])
    subset["conv_output"]["bias"] = jnp.ones_like(subset["conv_output"]["bias"]) * 0.05
    transform.params = {"subsets": [subset]}

    theta = 0.05 * jax.random.normal(jax.random.PRNGKey(6), (1, 2, 4, 4))
    transformed = transform.forward_with_params(transform.params, theta)
    inverted, diagnostics = transform.inverse(transformed, max_iter=50, tol=1e-6, return_diagnostics=True)

    assert np.allclose(np.asarray(transform.forward_with_params(transform.params, inverted)), np.asarray(transformed), atol=1e-5)
    assert float(diagnostics["max_final_diff"]) < 1e-6
    assert int(diagnostics["n_not_converged"]) == 0


def test_u1_autodiff_jacobian_is_check_only() -> None:
    theta = 0.1 * jax.random.normal(jax.random.PRNGKey(5), (1, 2, 4, 4))
    transform = FieldTransformation(4, model_tag="base", n_subsets=1, if_check_jac=False)

    def fail_autodiff(*_args, **_kwargs):
        raise AssertionError("autodiff Jacobian should not run when if_check_jac=False")

    transform.compute_jac_logdet_autodiff_with_params = fail_autodiff
    transform._check_jacobian_if_requested(transform.params, theta)

    checked = {"called": False}
    transform.if_check_jac = True

    def fake_autodiff(params, batch):
        checked["called"] = True
        return transform.compute_jac_logdet_with_params(params, batch)

    transform.compute_jac_logdet_autodiff_with_params = fake_autodiff
    transform._check_jacobian_if_requested(transform.params, theta)
    assert checked["called"]


@pytest.mark.filterwarnings("ignore:.*backend and device argument.*:UserWarning")
def test_u1_data_parallel_training_one_replica(tmp_path) -> None:
    transform = FieldTransformation(
        2,
        model_tag="base",
        n_subsets=1,
        model_dir=tmp_path / "models",
        plot_dir=tmp_path / "plots",
        dump_dir=tmp_path / "dumps",
        save_tag="dp_smoke",
        hyperparams={"lr": 0.0, "inverse_max_iters": 2, "early_stop_patience": 2},
    )
    data = np.asarray(0.01 * jax.random.normal(jax.random.PRNGKey(7), (4, 2, 2, 2)), dtype=np.float32)
    transform.train(data[:2], data[2:], 2.0, n_epochs=1, batch_size=2, data_parallel=True)

    assert (tmp_path / "dumps" / "train_loss_train_beta2.0_dp_smoke.csv").exists()
