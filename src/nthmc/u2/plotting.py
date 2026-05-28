"""Plotting helpers for U(2) HMC diagnostics."""

from __future__ import annotations

import numpy as np

from nthmc.core.plotting import hmc_summary as core_hmc_summary
from nthmc.u2.u2_observables import autocorrelation, plaq_mean_theory


def hmc_summary(
    max_lag: int,
    beta: float,
    volume: int,
    therm_plaq: list[float],
    plaq: list[float],
    topological_charges: list[float],
    hamiltonians: list[float],
    therm_acceptance_rate: float,
    acceptance_rate: float,
):
    """Print U(2) diagnostics and return a matplotlib summary figure."""
    autocorrelations = autocorrelation(np.array(topological_charges), max_lag, beta, volume)
    return core_hmc_summary(
        plaq_mean_theory(beta),
        float("nan"),
        volume,
        therm_plaq,
        plaq,
        topological_charges,
        hamiltonians,
        autocorrelations,
        therm_acceptance_rate,
        acceptance_rate,
    )
