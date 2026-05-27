#!/usr/bin/env python3
"""Run L=8 U(1) HMC and print raw topological charge samples."""

from __future__ import annotations

import math

import torch

from nthmc.u1.u1_hmc import HMCU1
from nthmc.u1.u1_observables import plaq_from_field, regularize, set_seed

LATTICE_SIZE = 8
BETA = 3.0
N_THERMALIZATION = 200
N_STEPS = 10
STEP_SIZE = 0.35
N_SAMPLES = 100
SEED = 1029


def raw_topology(theta: torch.Tensor) -> float:
    theta_p = regularize(plaq_from_field(theta))
    topo = torch.sum(theta_p) / (2 * math.pi)
    return float(topo.detach().cpu())


def main() -> None:
    torch.set_default_dtype(torch.float32)
    set_seed(SEED)

    hmc = HMCU1(
        lattice_size=LATTICE_SIZE,
        beta=BETA,
        n_thermalization_steps=N_THERMALIZATION,
        n_steps=N_STEPS,
        step_size=STEP_SIZE,
        tune_step_size=False,
    )

    theta_thermalized, _, therm_acceptance = hmc.thermalize()
    configs, _, acceptance, _, _ = hmc.run(N_SAMPLES, theta_thermalized, save_config=True)

    print(
        f"# system=2du1 L={LATTICE_SIZE} beta={BETA} n_samples={N_SAMPLES} "
        f"therm_acceptance={therm_acceptance:.4f} acceptance={acceptance:.4f}"
    )
    for theta in configs:
        print(raw_topology(theta))


if __name__ == "__main__":
    main()
