"""Evaluate a trained base field transformation with FT-HMC."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nthmc.core.jax_env import bootstrap_cuda_wheel_paths, preconfigure_platform_from_argv, set_platform


bootstrap_cuda_wheel_paths()
preconfigure_platform_from_argv()

from nthmc.u1.field_transform import FieldTransformation
from nthmc.u1.plotting import hmc_summary
from nthmc.u1.u1_observables import format_beta, set_seed
from nthmc.u1.u1_fthmc import HMCU1FT


def choose_device(device: str) -> str:
    if device == "cuda":
        device = "gpu"
    if device != "auto":
        set_platform(device)
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base FT-HMC for 2D U(1)")
    parser.add_argument("--lattice_size", type=int, default=16)
    parser.add_argument("--n_configs", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--train_beta", type=float, default=6.0)
    parser.add_argument("--n_thermalization", type=int, default=2000)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--ft_step_size", type=float, default=0.1)
    parser.add_argument("--n_tune_steps", type=int, default=1000)
    parser.add_argument("--max_lag", type=int, default=200)
    parser.add_argument("--rand_seed", type=int, default=1331)
    parser.add_argument("--model_tag", type=str, default="base")
    parser.add_argument("--save_tag", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu", "cuda"])
    parser.add_argument("--no_tune_step_size", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    set_seed(args.rand_seed)

    script_dir = Path(__file__).resolve().parent
    domain_root = script_dir.parents[1]
    model_dir = domain_root / "artifacts" / "models"
    plot_dir = script_dir / "plots"
    dump_dir = script_dir / "dumps"
    for directory in (plot_dir, dump_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(">>> U(1) JAX base FT-HMC evaluation")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print(f"resolved_device: {device}")
    print("=" * 60)

    field_transform = FieldTransformation(
        args.lattice_size,
        device=device,
        n_subsets=8,
        if_check_jac=False,
        model_tag=args.model_tag,
        save_tag=args.save_tag,
        model_dir=model_dir,
        plot_dir=plot_dir,
        dump_dir=dump_dir,
    )

    model_load_start = time.time()
    field_transform.load_best_model(args.train_beta)
    model_load_time = time.time() - model_load_start
    print(f">>> Model loaded in {model_load_time:.2f} seconds")

    hmc = HMCU1FT(
        args.lattice_size,
        args.beta,
        args.n_thermalization,
        args.n_steps,
        args.ft_step_size,
        field_transformation=field_transform.field_transformation,
        compute_jac_logdet=field_transform.compute_jac_logdet,
        observable_field_transformation=field_transform.field_transformation,
        device=device,
        tune_step_size=not args.no_tune_step_size,
        seed=args.rand_seed,
    )

    therm_start = time.time()
    theta_thermalized, therm_plaq, therm_acceptance_rate = hmc.thermalize(n_tune_steps=args.n_tune_steps)
    therm_time = time.time() - therm_start
    print(f">>> FT thermalization completed in {therm_time:.2f} seconds")

    run_start = time.time()
    _, plaq, acceptance_rate, topo, hamiltonians = hmc.run(args.n_configs, theta_thermalized, save_config=False)
    run_time = time.time() - run_start
    print(f">>> FT-HMC run completed in {run_time:.2f} seconds")

    beta_tag = format_beta(args.beta)
    volume = args.lattice_size**2
    fig = hmc_summary(
        args.beta,
        args.max_lag,
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
            plot_dir / f"comparison_fthmc_L{args.lattice_size}_beta{beta_tag}_nsteps{args.n_steps}_{args.save_tag}.pdf",
            transparent=True,
        )

    np.savetxt(
        dump_dir / f"topo_fthmc_L{args.lattice_size}_beta{beta_tag}_nsteps{args.n_steps}_{args.save_tag}.csv",
        np.array(topo),
        fmt="%.6e",
    )
    np.savetxt(
        dump_dir / f"accept_rate_fthmc_L{args.lattice_size}_beta{beta_tag}_nsteps{args.n_steps}_{args.save_tag}.csv",
        [acceptance_rate],
        fmt="%.6e",
    )
    benchmark = {
        "backend": "jax",
        "lattice_size": args.lattice_size,
        "beta": args.beta,
        "train_beta": args.train_beta,
        "n_thermalization": args.n_thermalization,
        "n_configs": args.n_configs,
        "n_steps": args.n_steps,
        "ft_step_size": args.ft_step_size,
        "rand_seed": args.rand_seed,
        "model_tag": args.model_tag,
        "save_tag": args.save_tag,
        "device": device,
        "model_load_time_sec": model_load_time,
        "thermalization_time_sec": therm_time,
        "run_time_sec": run_time,
        "total_time_sec": model_load_time + therm_time + run_time,
        "therm_acceptance_rate": therm_acceptance_rate,
        "acceptance_rate": acceptance_rate,
        "plaq_mean": float(np.mean(plaq)) if len(plaq) else float("nan"),
        "topo_mean": float(np.mean(topo)) if len(topo) else float("nan"),
        "topo_std": float(np.std(topo)) if len(topo) else float("nan"),
    }
    benchmark_path = dump_dir / f"benchmark_fthmc_L{args.lattice_size}_beta{beta_tag}_nsteps{args.n_steps}_{args.save_tag}.json"
    benchmark_path.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f">>> Benchmark JSON written to {benchmark_path}")
    print(f">>> Total FT-HMC time: {benchmark['total_time_sec']:.2f} seconds")


if __name__ == "__main__":
    main()
