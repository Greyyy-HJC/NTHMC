"""Summarize U(1) edge-link FT-HMC dumps versus HMC and production base."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nthmc.core.resampling import jackknife, jk_ls_avg
from nthmc.u1.u1_observables import autocorrelation_from_chi, format_beta

GAMMA_LAG = 16
MAX_LAG = 64
DEFAULT_SEEDS = (1029, 1107, 1331, 1984, 1999, 2008, 2017, 2025)
HEALTHY_ACCEPT = (0.55, 0.85)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_tag", type=str, required=True)
    parser.add_argument("--lattice_size", type=int, default=16)
    parser.add_argument("--train_beta", type=float, default=3.0)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--betas", type=float, nargs="+", default=[3.0, 4.0, 5.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max_lag", type=int, default=MAX_LAG)
    parser.add_argument("--gamma_lag", type=int, default=GAMMA_LAG)
    return parser.parse_args()


def load_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.atleast_1d(np.loadtxt(path))


def load_scalar(path: Path) -> float:
    values = load_vector(path)
    return float(values.reshape(-1)[0])


def delta_q(topology: np.ndarray) -> float:
    if len(topology) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(topology))))


def ratio_or_inf(numerator, denominator):
    if np.isclose(_gv_mean(denominator), 0.0):
        return np.inf
    return numerator / denominator


def _gv_mean(value) -> float:
    if hasattr(value, "sdev"):
        return float(value.mean)
    return float(value)


def gvar_to_dict(value) -> dict[str, float]:
    if hasattr(value, "sdev"):
        return {"mean": float(value.mean), "sdev": float(value.sdev)}
    return {"mean": float(value), "sdev": 0.0}


def average_pair_with_jackknife(left_values, right_values):
    left_values = np.asarray(left_values, dtype=float)
    right_values = np.asarray(right_values, dtype=float)
    if len(left_values) == 1:
        return left_values[0], right_values[0]
    joined = np.concatenate([left_values, right_values], axis=1)
    joined_avg = jk_ls_avg(jackknife(joined))
    split = left_values.shape[1]
    return joined_avg[:split], joined_avg[split:]


def dump_paths(args: argparse.Namespace, beta: float, seed: int) -> dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    domain_root = script_dir.parents[1]
    beta_tag = format_beta(beta)
    train_tag = format_beta(args.train_beta)
    prefix = f"L{args.lattice_size}_beta{beta_tag}_nsteps{args.n_steps}"
    base_save = f"base_train_b{train_tag}_L{args.lattice_size}_{seed}"
    edge_save = f"{args.run_tag}_train_b{train_tag}_L{args.lattice_size}_{seed}"
    hmc_dir = domain_root / "evaluation" / "hmc" / "dumps"
    base_dir = domain_root / "evaluation" / "base" / "dumps"
    edge_dir = script_dir / "dumps"
    return {
        "hmc_topo": hmc_dir / f"topo_hmc_{prefix}_{seed}.csv",
        "hmc_accept": hmc_dir / f"accept_rate_hmc_{prefix}_{seed}.csv",
        "base_topo": base_dir / f"topo_fthmc_{prefix}_{base_save}.csv",
        "base_accept": base_dir / f"accept_rate_fthmc_{prefix}_{base_save}.csv",
        "edge_topo": edge_dir / f"topo_fthmc_{prefix}_{edge_save}.csv",
        "edge_accept": edge_dir / f"accept_rate_fthmc_{prefix}_{edge_save}.csv",
        "edge_step": edge_dir / f"step_size_fthmc_{prefix}_{edge_save}.csv",
        "base_step": base_dir / f"step_size_fthmc_{prefix}_{base_save}.csv",
    }


def per_seed_metrics(args: argparse.Namespace, beta: float, seed: int) -> dict:
    paths = dump_paths(args, beta, seed)
    volume = args.lattice_size**2
    hmc_topo = load_vector(paths["hmc_topo"])
    base_topo = load_vector(paths["base_topo"])
    edge_topo = load_vector(paths["edge_topo"])
    n_keep = min(len(hmc_topo), len(base_topo), len(edge_topo))
    hmc_topo = hmc_topo[:n_keep]
    base_topo = base_topo[:n_keep]
    edge_topo = edge_topo[:n_keep]
    hmc_auto = autocorrelation_from_chi(hmc_topo, args.max_lag, beta, volume)
    base_auto = autocorrelation_from_chi(base_topo, args.max_lag, beta, volume)
    edge_auto = autocorrelation_from_chi(edge_topo, args.max_lag, beta, volume)
    hmc_delta = delta_q(hmc_topo)
    base_delta = delta_q(base_topo)
    edge_delta = delta_q(edge_topo)
    edge_accept = load_scalar(paths["edge_accept"])
    base_accept = load_scalar(paths["base_accept"])
    hmc_accept = load_scalar(paths["hmc_accept"])
    healthy = HEALTHY_ACCEPT[0] <= edge_accept <= HEALTHY_ACCEPT[1]
    return {
        "seed": seed,
        "n_samples": int(n_keep),
        "healthy_accept": healthy,
        "hmc_accept": hmc_accept,
        "base_accept": base_accept,
        "edge_accept": edge_accept,
        "base_step": load_scalar(paths["base_step"]),
        "edge_step": load_scalar(paths["edge_step"]),
        "r_gamma_edge_vs_hmc": float(ratio_or_inf(1.0 - edge_auto[args.gamma_lag], 1.0 - hmc_auto[args.gamma_lag])),
        "r_deltaq_edge_vs_hmc": float(ratio_or_inf(edge_delta, hmc_delta)),
        "r_gamma_base_vs_hmc": float(ratio_or_inf(1.0 - base_auto[args.gamma_lag], 1.0 - hmc_auto[args.gamma_lag])),
        "r_deltaq_base_vs_hmc": float(ratio_or_inf(base_delta, hmc_delta)),
        "r_gamma_edge_vs_base": float(ratio_or_inf(1.0 - edge_auto[args.gamma_lag], 1.0 - base_auto[args.gamma_lag])),
        "r_deltaq_edge_vs_base": float(ratio_or_inf(edge_delta, base_delta)),
        "_edge_auto": edge_auto,
        "_base_auto": base_auto,
        "_hmc_auto": hmc_auto,
        "_edge_delta": edge_delta,
        "_base_delta": base_delta,
        "_hmc_delta": hmc_delta,
    }


def collect_beta(args: argparse.Namespace, beta: float) -> dict:
    rows = []
    missing = []
    for seed in args.seeds:
        try:
            rows.append(per_seed_metrics(args, beta, seed))
        except FileNotFoundError as exc:
            missing.append({"seed": seed, "path": str(exc)})
    healthy_rows = [row for row in rows if row["healthy_accept"]]
    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    summary = {
        "beta": beta,
        "n_complete": len(rows),
        "n_healthy": len(healthy_rows),
        "missing": missing,
        "per_seed": public_rows,
    }
    if healthy_rows:
        hmc_autos = np.asarray([row["_hmc_auto"] for row in healthy_rows], dtype=float)
        edge_autos = np.asarray([row["_edge_auto"] for row in healthy_rows], dtype=float)
        base_autos = np.asarray([row["_base_auto"] for row in healthy_rows], dtype=float)
        hmc_auto_avg, edge_auto_avg = average_pair_with_jackknife(hmc_autos, edge_autos)
        _, base_auto_avg = average_pair_with_jackknife(hmc_autos, base_autos)
        hmc_deltas = np.asarray([[row["_hmc_delta"]] for row in healthy_rows], dtype=float)
        edge_deltas = np.asarray([[row["_edge_delta"]] for row in healthy_rows], dtype=float)
        base_deltas = np.asarray([[row["_base_delta"]] for row in healthy_rows], dtype=float)
        hmc_delta_avg, edge_delta_avg = average_pair_with_jackknife(hmc_deltas, edge_deltas)
        _, base_delta_avg = average_pair_with_jackknife(hmc_deltas, base_deltas)
        summary["jackknife_healthy"] = {
            "r_gamma_edge_vs_hmc": gvar_to_dict(
                ratio_or_inf(1.0 - edge_auto_avg[args.gamma_lag], 1.0 - hmc_auto_avg[args.gamma_lag])
            ),
            "r_deltaq_edge_vs_hmc": gvar_to_dict(ratio_or_inf(edge_delta_avg[0], hmc_delta_avg[0])),
            "r_gamma_base_vs_hmc": gvar_to_dict(
                ratio_or_inf(1.0 - base_auto_avg[args.gamma_lag], 1.0 - hmc_auto_avg[args.gamma_lag])
            ),
            "r_deltaq_base_vs_hmc": gvar_to_dict(ratio_or_inf(base_delta_avg[0], hmc_delta_avg[0])),
            "r_gamma_edge_vs_base": gvar_to_dict(
                ratio_or_inf(1.0 - edge_auto_avg[args.gamma_lag], 1.0 - base_auto_avg[args.gamma_lag])
            ),
            "r_deltaq_edge_vs_base": gvar_to_dict(ratio_or_inf(edge_delta_avg[0], base_delta_avg[0])),
        }
    else:
        summary["jackknife_healthy"] = {}
    return summary


def main() -> None:
    args = parse_args()
    payload = {
        "run_tag": args.run_tag,
        "lattice_size": args.lattice_size,
        "train_beta": args.train_beta,
        "n_steps": args.n_steps,
        "gamma_lag": args.gamma_lag,
        "max_lag": args.max_lag,
        "seeds": args.seeds,
        "betas": {},
    }
    dump_dir = Path(__file__).resolve().parent / "dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    for beta in args.betas:
        summary = collect_beta(args, beta)
        payload["betas"][format_beta(beta)] = summary
        print(
            f"beta={format_beta(beta)} complete={summary['n_complete']} "
            f"healthy={summary['n_healthy']} missing={len(summary['missing'])}"
        )
        jack = summary.get("jackknife_healthy") or {}
        if jack:
            print(
                f"  R_gamma edge/HMC={jack['r_gamma_edge_vs_hmc']['mean']:.3f}"
                f"({jack['r_gamma_edge_vs_hmc']['sdev']:.3f}) "
                f"R_deltaQ edge/HMC={jack['r_deltaq_edge_vs_hmc']['mean']:.3f}"
                f"({jack['r_deltaq_edge_vs_hmc']['sdev']:.3f})"
            )
            print(
                f"  R_gamma edge/base={jack['r_gamma_edge_vs_base']['mean']:.3f}"
                f"({jack['r_gamma_edge_vs_base']['sdev']:.3f}) "
                f"R_deltaQ edge/base={jack['r_deltaq_edge_vs_base']['mean']:.3f}"
                f"({jack['r_deltaq_edge_vs_base']['sdev']:.3f})"
            )
    out_path = dump_dir / f"edge_summary_{args.run_tag}_L{args.lattice_size}_train_b{format_beta(args.train_beta)}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
