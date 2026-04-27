"""Generate 2D U(2) gauge configurations with standard HMC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nthmc.u2.plotting import hmc_summary
from nthmc.u2.u2_hmc import HMCU2
from nthmc.u2.u2_observables import format_beta, set_seed, u2_to_matrix


def choose_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2D U(2) gauge configurations")
    parser.add_argument("--lattice_size", type=int, default=8)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--n_thermalization", type=int, default=200)
    parser.add_argument("--store_interval", type=int, default=1)
    parser.add_argument("--n_configs", type=int, default=2048)
    parser.add_argument("--n_steps", type=int, default=1)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--n_tune_steps", type=int, default=2000)
    parser.add_argument("--rand_seed", type=int, default=1331)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no_tune_step_size", action="store_true")
    parser.add_argument("--max_lag", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    torch.set_default_dtype(torch.float32)
    set_seed(args.rand_seed)

    script_dir = Path(__file__).resolve().parent
    domain_root = script_dir.parent
    gauge_dir = domain_root / "configs"
    dump_dir = script_dir / "dumps"
    plot_dir = script_dir / "plots"
    gauge_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(">>> U(2) gauge generation")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print(f"resolved_device: {device}")
    print("=" * 60)

    hmc = HMCU2(
        args.lattice_size,
        args.beta,
        args.n_thermalization,
        args.n_steps,
        args.step_size,
        device=device,
        tune_step_size=not args.no_tune_step_size,
    )
    links_thermalized, therm_plaq, therm_acceptance_rate = hmc.thermalize(n_tune_steps=args.n_tune_steps)
    n_iterations = args.store_interval * args.n_configs
    configs, plaq, acceptance_rate, topo, hamiltonians = hmc.run(
        n_iterations,
        links_thermalized,
        store_interval=args.store_interval,
        save_config=True,
    )

    beta_tag = format_beta(args.beta)
    volume = args.lattice_size**2
    fig = hmc_summary(
        args.max_lag,
        args.beta,
        volume,
        therm_plaq,
        plaq,
        topo,
        hamiltonians,
        therm_acceptance_rate,
        acceptance_rate,
    )
    if fig is not None:
        fig.savefig(plot_dir / f"gauge_gen_hmc_L{args.lattice_size}_beta{beta_tag}.pdf", transparent=True)

    np.save(gauge_dir / f"links_L{args.lattice_size}_beta{beta_tag}.npy", u2_to_matrix(torch.stack(configs)).numpy())
    np.savetxt(dump_dir / f"plaq_L{args.lattice_size}_beta{beta_tag}.csv", np.array(plaq), fmt="%.6e")
    np.savetxt(dump_dir / f"topo_L{args.lattice_size}_beta{beta_tag}.csv", np.array(topo), fmt="%.6e")
    np.savetxt(dump_dir / f"accept_rate_L{args.lattice_size}_beta{beta_tag}.csv", [acceptance_rate], fmt="%.6e")
    print(">>> Gauge generation completed")


if __name__ == "__main__":
    main()
