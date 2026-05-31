"""Evaluate a trained base field transformation with FT-HMC for 2D U(2)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nthmc.u2.field_transform import FieldTransformation
from nthmc.u2.plotting import hmc_summary
from nthmc.u2.u2_fthmc import HMCU2FT
from nthmc.u2.u2_observables import format_beta, set_seed


def choose_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base FT-HMC for 2D U(2)")
    parser.add_argument("--lattice_size", type=int, default=8)
    parser.add_argument("--n_configs", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--train_beta", type=float, default=3.0)
    parser.add_argument("--n_thermalization", type=int, default=200)
    parser.add_argument("--n_steps", type=int, default=4)
    parser.add_argument("--ft_step_size", type=float, default=0.1)
    parser.add_argument("--n_tune_steps", type=int, default=1000)
    parser.add_argument("--max_lag", type=int, default=20)
    parser.add_argument("--rand_seed", type=int, default=1331)
    parser.add_argument("--model_tag", type=str, default="base")
    parser.add_argument("--save_tag", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no_tune_step_size", action="store_true")
    parser.add_argument("--if_compile", action="store_true")
    parser.add_argument("--compile_backend", type=str, default="inductor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    torch.set_default_dtype(torch.float32)
    set_seed(args.rand_seed)

    script_dir = Path(__file__).resolve().parent
    domain_root = script_dir.parents[1]
    model_dir = domain_root / "artifacts" / "models"
    plot_dir = script_dir / "plots"
    dump_dir = script_dir / "dumps"
    for directory in (plot_dir, dump_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(">>> U(2) base FT-HMC evaluation")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print(f"resolved_device: {device}")
    print("=" * 60)

    field_transform = FieldTransformation(
        args.lattice_size,
        device=device,
        n_subsets=8,
        if_check_jac=False,
        num_workers=0,
        model_tag=args.model_tag,
        save_tag=args.save_tag,
        model_dir=model_dir,
        plot_dir=plot_dir,
        dump_dir=dump_dir,
        compile_enabled=False, # for lazy compile
    )

    model_load_start = time.time()
    field_transform.load_best_model(args.train_beta)
    field_transform.freeze_models_for_eval()
    model_load_time = time.time() - model_load_start
    print(f">>> Model loaded in {model_load_time:.2f} seconds")
    if args.if_compile:
        field_transform.enable_eval_compile(backend=args.compile_backend)
        force_field_transformation = field_transform.field_transformation_compiled
        force_compute_jac_logdet = field_transform.compute_jac_logdet_compiled
    else:
        force_field_transformation = None
        force_compute_jac_logdet = None

    hmc = HMCU2FT(
        args.lattice_size,
        args.beta,
        args.n_thermalization,
        args.n_steps,
        args.ft_step_size,
        field_transformation=field_transform.field_transformation,
        compute_jac_logdet=field_transform.compute_jac_logdet,
        observable_field_transformation=field_transform.field_transformation,
        force_field_transformation=force_field_transformation,
        force_compute_jac_logdet=force_compute_jac_logdet,
        device=device,
        tune_step_size=not args.no_tune_step_size,
    )

    therm_start = time.time()
    links_thermalized, therm_plaq, therm_acceptance_rate = hmc.thermalize(n_tune_steps=args.n_tune_steps)
    therm_time = time.time() - therm_start
    print(f">>> FT thermalization completed in {therm_time:.2f} seconds")

    run_start = time.time()
    _, plaq, acceptance_rate, topo, hamiltonians = hmc.run(args.n_configs, links_thermalized, save_config=False)
    run_time = time.time() - run_start
    print(f">>> FT-HMC run completed in {run_time:.2f} seconds")

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
    print(f">>> Total FT-HMC time: {model_load_time + therm_time + run_time:.2f} seconds")


if __name__ == "__main__":
    main()
