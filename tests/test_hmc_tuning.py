from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nthmc.core.hmc_tuning import tune_step_size
from nthmc.u1.u1_fthmc import HMCU1FT
from nthmc.u1.u1_hmc import HMCU1
from nthmc.u2.u2_fthmc import HMCU2FT
from nthmc.u2.u2_hmc import HMCU2


def make_u1_hmc(*, tune: bool = True, thermalization: int = 0):
    return HMCU1(2, 1.0, thermalization, 1, 0.2, tune_step_size=tune)


def make_u2_hmc(*, tune: bool = True, thermalization: int = 0):
    return HMCU2(2, 1.0, thermalization, 1, 0.2, tune_step_size=tune)


def make_u1_fthmc(*, tune: bool = True, thermalization: int = 0):
    return HMCU1FT(
        2,
        1.0,
        thermalization,
        1,
        0.2,
        field_transformation=lambda field: field,
        compute_jac_logdet=lambda batch: jnp.zeros(batch.shape[0]),
        tune_step_size=tune,
    )


def make_u2_fthmc(*, tune: bool = True, thermalization: int = 0):
    return HMCU2FT(
        2,
        1.0,
        thermalization,
        1,
        0.2,
        field_transformation=lambda field: field,
        compute_jac_logdet=lambda batch: jnp.zeros(batch.shape[0]),
        tune_step_size=tune,
    )


SAMPLER_FACTORIES: tuple[tuple[Callable[..., object], str], ...] = (
    (make_u1_hmc, "theta"),
    (make_u2_hmc, "links"),
    (make_u1_fthmc, "theta"),
    (make_u2_fthmc, "links"),
)


@pytest.mark.parametrize(("factory", "state_name"), SAMPLER_FACTORIES)
def test_all_samplers_tune_step_size(factory, state_name: str) -> None:
    sampler = factory()

    def controlled_step(state, step_size: float):
        index = int(state) % 10
        if step_size < 0.3:
            accepted = True
        elif step_size < 0.5:
            accepted = index < 7
        else:
            accepted = False
        return state + 1, accepted, 0.0

    sampler._metropolis_step_at = controlled_step
    kwargs = {
        "n_tune_steps": 10,
        "target_rate": 0.70,
        "target_tolerance": 0.01,
        "max_attempts": 3,
        state_name: 0,
    }
    sampler.tune_step_size(**kwargs)
    assert sampler.dt == pytest.approx(0.4)


@pytest.mark.parametrize(("factory", "_state_name"), SAMPLER_FACTORIES)
def test_tuning_can_be_disabled(factory, _state_name: str) -> None:
    sampler = factory(tune=False, thermalization=0)

    def fail_tuning(**_kwargs):
        raise AssertionError("step-size tuning should be disabled")

    sampler.tune_step_size = fail_tuning
    sampler.thermalize(n_tune_steps=0)
    assert sampler.dt == pytest.approx(0.2)


def test_tuning_falls_back_to_best_observed_step_size() -> None:
    def step(state: int, step_size: float):
        return state + 1, step_size < 0.3, 0.0

    selected = tune_step_size(
        0,
        0.2,
        step,
        n_tune_steps=2,
        target_rate=0.7,
        target_tolerance=0.01,
        max_attempts=2,
    )
    assert selected == pytest.approx(0.2)


@pytest.mark.parametrize("n_tune_steps", [0, -1])
def test_tuning_rejects_nonpositive_step_count(n_tune_steps: int) -> None:
    with pytest.raises(ValueError, match="n_tune_steps must be positive"):
        tune_step_size(0, 0.1, lambda state, _dt: (state, True, 0.0), n_tune_steps=n_tune_steps)


def test_jitted_hmc_step_uses_dynamic_dt_without_recompiling() -> None:
    sampler = HMCU1(2, 0.0, 0, 1, 0.1, tune_step_size=False)
    step = sampler._make_step()
    state = sampler.initialize()
    key = jax.random.PRNGKey(7)

    small = step(state, key, jnp.asarray(0.05, dtype=state.dtype))[0]
    large = step(state, key, jnp.asarray(0.20, dtype=state.dtype))[0]

    assert not np.allclose(np.asarray(small), np.asarray(large))
    assert step._cache_size() == 1
