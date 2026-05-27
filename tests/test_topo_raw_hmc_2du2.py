#!/usr/bin/env python3
"""Run L=8 U(2) HMC and print raw topological charge samples."""

from __future__ import annotations

import math

import torch

from nthmc.u2.u2_hmc import HMCU2
from nthmc.u2.u2_observables import plaquette_from_field, regularize_phase, set_seed

LATTICE_SIZE = 8
BETA = 3.0
N_THERMALIZATION = 200
N_STEPS = 4
STEP_SIZE = 0.1
N_SAMPLES = 100
SEED = 1331


def raw_topology(links: torch.Tensor) -> float:
    plaquettes = plaquette_from_field(links)
    determinant_phase = regularize_phase(2 * plaquettes[..., 0])
    topo = torch.sum(determinant_phase) / (2 * math.pi)
    return float(topo.detach().cpu())


def main() -> None:
    torch.set_default_dtype(torch.float32)
    set_seed(SEED)

    hmc = HMCU2(
        lattice_size=LATTICE_SIZE,
        beta=BETA,
        n_thermalization_steps=N_THERMALIZATION,
        n_steps=N_STEPS,
        step_size=STEP_SIZE,
        tune_step_size=False,
    )

    links_thermalized, _, therm_acceptance = hmc.thermalize()
    configs, _, acceptance, _, _ = hmc.run(N_SAMPLES, links_thermalized, save_config=True)

    print(
        f"# system=2du2 L={LATTICE_SIZE} beta={BETA} n_samples={N_SAMPLES} "
        f"therm_acceptance={therm_acceptance:.4f} acceptance={acceptance:.4f}"
    )
    for links in configs:
        print(raw_topology(links))


if __name__ == "__main__":
    main()
