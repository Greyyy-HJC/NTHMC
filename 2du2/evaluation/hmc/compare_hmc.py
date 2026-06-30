"""Evaluate standard HMC for 2D U(2)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nthmc.core.jax_env import bootstrap_cuda_wheel_paths, preconfigure_platform_from_argv, set_platform


bootstrap_cuda_wheel_paths()
preconfigure_platform_from_argv()

from nthmc.u2.plotting import hmc_summary
from nthmc.u2.u2_hmc import HMCU2
from nthmc.u2.u2_observables import format_beta, set_seed


def choose_device(device: str) -> str:
    if device == "cuda":
        device = "gpu"
    if device != "auto":
        set_platform(device)
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standard HMC for 2D U(2)")
    parser.add_argument("--lattice_size", type=int, default=8)
    parser.add_argument("--n_configs", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--n_thermalization", type=int, default=200)
    parser.add_argument("--n_steps", type=int, default=4)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--n_tune_steps", type=int, default=1000)
    parser.add_argument("--max_lag", type=int, default=20)
    parser.add_argument("--rand_seed", type=int, default=1331)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu", "cuda"])
    parser.add_argument("--no_tune_step_size", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    set_seed(args.rand_seed)

    script_dir = Path(__file__).resolve().parent
    plot_dir = script_dir / "plots"
    dump_dir = script_dir / "dumps"
    for directory in (plot_dir, dump_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(">>> U(2) standard HMC evaluation")
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
        seed=args.rand_seed,
    )

    therm_start = time.time()
    links_thermalized, therm_plaq, therm_acceptance_rate = hmc.thermalize(n_tune_steps=args.n_tune_steps)
    therm_time = time.time() - therm_start
    print(f">>> HMC thermalization completed in {therm_time:.2f} seconds")

    run_start = time.time()
    _, plaq, acceptance_rate, topo, hamiltonians = hmc.run(args.n_configs, links_thermalized, save_config=False)
    run_time = time.time() - run_start
    print(f">>> HMC run completed in {run_time:.2f} seconds")

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
        fig.savefig(
            plot_dir / f"comparison_hmc_L{args.lattice_size}_beta{beta_tag}_nsteps{args.n_steps}_{args.rand_seed}.pdf",
            transparent=True,
        )

    np.savetxt(
        dump_dir / f"topo_hmc_L{args.lattice_size}_beta{beta_tag}_nsteps{args.n_steps}_{args.rand_seed}.csv",
        np.array(topo),
        fmt="%.6e",
    )
    np.savetxt(
        dump_dir / f"accept_rate_hmc_L{args.lattice_size}_beta{beta_tag}_nsteps{args.n_steps}_{args.rand_seed}.csv",
        [acceptance_rate],
        fmt="%.6e",
    )
    print(f">>> Total HMC time: {therm_time + run_time:.2f} seconds")


if __name__ == "__main__":
    main()
