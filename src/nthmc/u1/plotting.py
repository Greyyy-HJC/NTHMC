"""Plotting helpers for U(1) HMC diagnostics."""

from __future__ import annotations

import numpy as np

from nthmc.core.plotting import hmc_summary as core_hmc_summary
from nthmc.u1.u1_observables import autocorrelation_from_chi, chi_infinity, plaq_mean_theory


def hmc_summary(
    beta: float,
    max_lag: int,
    volume: int,
    therm_plaq: list[float],
    plaq: list[float],
    topological_charges: list[float],
    hamiltonians: list[float],
    therm_acceptance_rate: float,
    acceptance_rate: float,
):
    """Print U(1) diagnostics and return a matplotlib summary figure."""
    theoretical_plaq = plaq_mean_theory(beta)
    theoretical_chi = chi_infinity(beta)
    autocorrelations = autocorrelation_from_chi(np.array(topological_charges), max_lag, beta, volume)
    return core_hmc_summary(
        theoretical_plaq,
        theoretical_chi,
        volume,
        therm_plaq,
        plaq,
        topological_charges,
        hamiltonians,
        autocorrelations,
        therm_acceptance_rate,
        acceptance_rate,
    )
