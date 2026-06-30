#!/usr/bin/env python3
"""Compute L=16 base R_gamma(16) and R_deltaQ from topology dumps."""
from __future__ import annotations

import sys
from pathlib import Path

import gvar as gv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nthmc.core.resampling import jackknife, jk_ls_avg
from nthmc.u2.u2_observables import autocorrelation, format_beta

LATTICE_SIZE = 16
TRAIN_BETA = 8.0
MODEL_TAG = "base"
N_STEPS = 10
MAX_LAG = 64
GAMMA_LAG = 16
BETAS = [10.0, 12.0, 14.0, 16.0]
SEEDS = [1029, 1331, 1999]

HMC_DUMP_DIR = REPO_ROOT / "2du2" / "evaluation" / "hmc" / "dumps"
FTHMC_DUMP_DIR = REPO_ROOT / "2du2" / "evaluation" / "l16_base" / "dumps"


def scaling_save_tag(lattice_size: int, seed: int) -> str:
    return f"{MODEL_TAG}_train_b{format_beta(TRAIN_BETA)}_L{lattice_size}_{seed}"


def hmc_topo_path(lattice_size: int, beta: float, seed: int) -> Path:
    return HMC_DUMP_DIR / f"topo_hmc_L{lattice_size}_beta{format_beta(beta)}_nsteps{N_STEPS}_{seed}.csv"


def fthmc_topo_path(lattice_size: int, beta: float, seed: int) -> Path:
    save_tag = scaling_save_tag(lattice_size, seed)
    return FTHMC_DUMP_DIR / f"topo_fthmc_L{lattice_size}_beta{format_beta(beta)}_nsteps{N_STEPS}_{save_tag}.csv"


def load_topology(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing topology CSV: {path}")
    return np.atleast_1d(np.loadtxt(path))


def delta_q(topology: np.ndarray) -> float:
    if len(topology) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(topology))))


def gamma_ratio_from_autocorrelations(hmc_autocorrelation, fthmc_autocorrelation):
    if GAMMA_LAG >= len(hmc_autocorrelation) or GAMMA_LAG >= len(fthmc_autocorrelation):
        return np.nan
    hmc_denom = 1.0 - hmc_autocorrelation[GAMMA_LAG]
    fthmc_denom = 1.0 - fthmc_autocorrelation[GAMMA_LAG]
    if np.isclose(gv.mean(hmc_denom), 0.0):
        return gv.gvar(np.inf, 0.0)
    return fthmc_denom / hmc_denom


def ratio_or_inf(numerator, denominator):
    if np.isclose(gv.mean(denominator), 0.0):
        return gv.gvar(np.inf, 0.0)
    return numerator / denominator


def average_pair_with_jackknife(hmc_values, fthmc_values):
    hmc_values = np.asarray(hmc_values, dtype=float)
    fthmc_values = np.asarray(fthmc_values, dtype=float)
    if len(hmc_values) == 1:
        hmc_avg = gv.gvar(hmc_values[0], np.zeros_like(hmc_values[0]))
        fthmc_avg = gv.gvar(fthmc_values[0], np.zeros_like(fthmc_values[0]))
        return hmc_avg, fthmc_avg
    joined = np.concatenate([hmc_values, fthmc_values], axis=1)
    joined_avg = jk_ls_avg(jackknife(joined))
    split = hmc_values.shape[1]
    return joined_avg[:split], joined_avg[split:]


def collect_one(lattice_size: int, beta: float) -> dict:
    volume = lattice_size**2
    hmc_autos, fthmc_autos, hmc_delta_q, fthmc_delta_q = [], [], [], []
    for seed in SEEDS:
        hmc_topo = load_topology(hmc_topo_path(lattice_size, beta, seed))
        fthmc_topo = load_topology(fthmc_topo_path(lattice_size, beta, seed))
        hmc_autos.append(autocorrelation(hmc_topo, MAX_LAG, beta, volume))
        fthmc_autos.append(autocorrelation(fthmc_topo, MAX_LAG, beta, volume))
        hmc_delta_q.append([delta_q(hmc_topo)])
        fthmc_delta_q.append([delta_q(fthmc_topo)])
    hmc_auto_avg, fthmc_auto_avg = average_pair_with_jackknife(hmc_autos, fthmc_autos)
    hmc_delta_avg, fthmc_delta_avg = average_pair_with_jackknife(hmc_delta_q, fthmc_delta_q)
    return {
        "gamma_ratio": gamma_ratio_from_autocorrelations(hmc_auto_avg, fthmc_auto_avg),
        "delta_q_ratio": ratio_or_inf(fthmc_delta_avg[0], hmc_delta_avg[0]),
    }


def main() -> int:
    print(f"L={LATTICE_SIZE} base reference (T2 seeds={SEEDS})")
    rows = []
    for beta in BETAS:
        try:
            result = collect_one(LATTICE_SIZE, beta)
            line = f"beta={beta:.1f} R_gamma(16)={result['gamma_ratio']} R_deltaQ={result['delta_q_ratio']}"
            print(line)
            rows.append((beta, result["gamma_ratio"], result["delta_q_ratio"]))
        except FileNotFoundError as exc:
            print(f"beta={beta:.1f} SKIP: {exc}", file=sys.stderr)
    if not rows:
        return 1
    print("\n| beta | R_gamma(16) | R_deltaQ |")
    print("| --- | --- | --- |")
    for beta, gamma_ratio, delta_q_ratio in rows:
        print(f"| {beta:.1f} | {gamma_ratio} | {delta_q_ratio} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
