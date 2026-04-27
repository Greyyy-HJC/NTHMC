"""Plotting helpers for U(1) HMC diagnostics."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nthmc.common.u1_observables import autocorrelation_from_chi, chi_infinity, plaq_mean_theory


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
    """Print core diagnostics and return a matplotlib summary figure."""
    autocorrelations = autocorrelation_from_chi(np.array(topological_charges), max_lag, beta, volume)
    fig = plot_results(beta, therm_plaq, plaq, topological_charges, hamiltonians, autocorrelations)

    print(f"Thermalization acceptance rate: {therm_acceptance_rate:.4f}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")
    topo = np.array(topological_charges)
    print(">>> Topological susceptibility <Q^2>/V:", np.mean(topo**2) / volume)
    print(">>> Topological susceptibility theory:", chi_infinity(beta))
    return fig


def plot_results(
    beta: float,
    therm_plaq: list[float],
    plaq: list[float],
    topological_charges: list[float],
    hamiltonians: list[float],
    autocorrelations: np.ndarray,
):
    """Create a four-panel HMC diagnostic plot."""
    fig = plt.figure(figsize=(18, 12))
    fontsize = 18

    plt.subplot(221)
    plt.plot(np.arange(len(therm_plaq)), therm_plaq, label="Thermalization Plaquette", color="blue")
    plt.plot(np.arange(len(plaq)) + len(therm_plaq), plaq, label="Plaquette", color="orange")
    plt.axhline(y=plaq_mean_theory(beta), color="r", linestyle="--", label="Theoretical Plaquette")
    plt.legend(loc="upper right", fontsize=fontsize - 2)
    plt.title("Plaquette vs. Iteration", fontsize=fontsize)
    plt.xlabel("Iteration", fontsize=fontsize)
    plt.ylabel("Plaquette", fontsize=fontsize)
    plt.tick_params(direction="in", top=True, right=True, labelsize=fontsize - 2)
    plt.grid(linestyle=":")

    plt.subplot(222)
    plt.plot(hamiltonians)
    plt.title("Hamiltonian vs. Iteration", fontsize=fontsize)
    plt.xlabel("Iteration", fontsize=fontsize)
    plt.ylabel("Hamiltonian", fontsize=fontsize)
    plt.tick_params(direction="in", top=True, right=True, labelsize=fontsize - 2)
    plt.grid(linestyle=":")
    plt.axhline(y=np.mean(hamiltonians), color="r", linestyle="--", label="Mean Hamiltonian")
    plt.legend(fontsize=fontsize - 2, loc="upper right")

    plt.subplot(223)
    plt.plot(topological_charges, marker="o", markersize=3)
    plt.axhline(
        y=np.mean(topological_charges),
        color="r",
        linestyle="--",
        marker="o",
        markersize=3,
        label="Mean Topological Charge",
    )
    plt.title("Topological Charge vs. Iteration", fontsize=fontsize)
    plt.xlabel("Iteration", fontsize=fontsize)
    plt.ylabel("Topological Charge", fontsize=fontsize)
    plt.tick_params(direction="in", top=True, right=True, labelsize=fontsize - 2)
    plt.grid(linestyle=":")
    plt.legend(fontsize=fontsize - 2, loc="upper right")

    plt.subplot(224)
    plt.plot(range(len(autocorrelations)), autocorrelations, marker="o")
    plt.title("Autocorrelation", fontsize=fontsize)
    plt.xlabel("MDTU", fontsize=fontsize)
    plt.ylabel("Autocorrelation", fontsize=fontsize)
    plt.tick_params(direction="in", top=True, right=True, labelsize=fontsize - 2)
    plt.grid(linestyle=":")

    plt.tight_layout()
    print(">>> Theoretical plaquette:", plaq_mean_theory(beta))
    print(">>> Mean plaq:", np.mean(plaq))
    return fig
