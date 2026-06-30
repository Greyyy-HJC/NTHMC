"""Evaluate a trained U(1) field transformation with JAX FT-HMC."""

from __future__ import annotations

import argparse
import json
import os
import site
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def bootstrap_cuda_wheel_library_path() -> None:
    """Re-exec with venv NVIDIA wheel library paths visible to JAX's CUDA plugin."""
    if os.environ.get("NTHMC_JAX_CUDA_BOOTSTRAPPED") == "1":
        return

    lib_dirs = []
    for site_dir in site.getsitepackages():
        nvidia_root = Path(site_dir) / "nvidia"
        if nvidia_root.exists():
            lib_dirs.extend(path for path in nvidia_root.glob("*/lib") if path.is_dir())
    if not lib_dirs:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = ":".join(str(path) for path in lib_dirs)
    os.environ["LD_LIBRARY_PATH"] = f"{new_paths}:{existing}" if existing else new_paths
    os.environ["NTHMC_JAX_CUDA_BOOTSTRAPPED"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


bootstrap_cuda_wheel_library_path()

import jax

from nthmc.u1.jax_backend import JaxU1FieldTransformation, build_fthmc_chain, load_checkpoint_params
from nthmc.u1.plotting import hmc_summary
from nthmc.u1.u1_observables import format_beta


def choose_platform(device: str) -> str:
    if device == "auto":
        return "gpu" if any(jax_device.platform == "gpu" for jax_device in jax.devices()) else "cpu"
    if device == "cuda":
        return "gpu"
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate JAX FT-HMC for 2D U(1)")
    parser.add_argument("--lattice_size", type=int, default=16)
    parser.add_argument("--n_configs", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--train_beta", type=float, default=6.0)
    parser.add_argument("--n_thermalization", type=int, default=2000)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--ft_step_size", type=float, default=0.1)
    parser.add_argument("--max_lag", type=int, default=200)
    parser.add_argument("--rand_seed", type=int, default=1331)
    parser.add_argument("--model_tag", type=str, default="base", choices=["base", "addcos"])
    parser.add_argument("--save_tag", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    platform = choose_platform(args.device)
    device = jax.devices(platform)[0]

    script_dir = Path(__file__).resolve().parent
    domain_root = script_dir.parents[1]
    model_dir = domain_root / "artifacts" / "models"
    plot_dir = script_dir / "plots"
    dump_dir = script_dir / "dumps"
    for directory in (plot_dir, dump_dir):
        directory.mkdir(parents=True, exist_ok=True)

    beta_tag = format_beta(args.beta)
    train_beta_tag = format_beta(args.train_beta)
    checkpoint_path = model_dir / f"best_model_train_beta{train_beta_tag}_{args.save_tag}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing trained model checkpoint: {checkpoint_path}")

    print("=" * 60)
    print(">>> U(1) JAX FT-HMC evaluation")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print(f"resolved_jax_platform: {platform}")
    print(f"resolved_jax_device: {device}")
    print("=" * 60)

    model_load_start = time.time()
    params = load_checkpoint_params(checkpoint_path, model_tag=args.model_tag, n_subsets=8)
    transform = JaxU1FieldTransformation(params, lattice_size=args.lattice_size, n_subsets=8)
    model_load_time = time.time() - model_load_start
    print(f">>> Model loaded and converted in {model_load_time:.2f} seconds")

    chain = build_fthmc_chain(
        transform,
        beta=args.beta,
        n_thermalization=args.n_thermalization,
        n_configs=args.n_configs,
        n_steps=args.n_steps,
        step_size=args.ft_step_size,
    )
    chain_jit = jax.jit(chain)
    key = jax.random.PRNGKey(args.rand_seed)

    with jax.default_device(device):
        compile_start = time.time()
        compiled_chain = chain_jit.lower(key).compile()
        compile_time = time.time() - compile_start
        print(f">>> JAX compilation completed in {compile_time:.2f} seconds")

        run_start = time.time()
        result = compiled_chain(key)
        result.plaq.block_until_ready()
        execution_time = time.time() - run_start
        print(f">>> JAX FT-HMC thermalization+run completed in {execution_time:.2f} seconds")

    therm_plaq = np.asarray(result.therm_plaq)
    plaq = np.asarray(result.plaq)
    topo = np.asarray(result.topo)
    hamiltonians = np.asarray(result.hamiltonians)
    therm_acceptance_rate = float(np.asarray(result.therm_acceptance_rate))
    acceptance_rate = float(np.asarray(result.acceptance_rate))

    volume = args.lattice_size**2
    fig = hmc_summary(
        args.beta,
        args.max_lag,
        volume,
        therm_plaq.tolist(),
        plaq.tolist(),
        topo.tolist(),
        hamiltonians.tolist(),
        therm_acceptance_rate,
        acceptance_rate,
    )
    if fig is not None:
        fig.savefig(
            plot_dir / f"comparison_fthmc_jax_L{args.lattice_size}_beta{beta_tag}_nsteps{args.n_steps}_{args.save_tag}.pdf",
            transparent=True,
        )

    np.savetxt(
        dump_dir / f"topo_fthmc_jax_L{args.lattice_size}_beta{beta_tag}_nsteps{args.n_steps}_{args.save_tag}.csv",
        topo,
        fmt="%.6e",
    )
    np.savetxt(
        dump_dir / f"accept_rate_fthmc_jax_L{args.lattice_size}_beta{beta_tag}_nsteps{args.n_steps}_{args.save_tag}.csv",
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
        "device": str(device),
        "model_load_time_sec": model_load_time,
        "compile_time_sec": compile_time,
        "execution_time_sec": execution_time,
        "total_time_sec": model_load_time + compile_time + execution_time,
        "therm_acceptance_rate": therm_acceptance_rate,
        "acceptance_rate": acceptance_rate,
        "plaq_mean": float(np.mean(plaq)) if len(plaq) else float("nan"),
        "topo_mean": float(np.mean(topo)) if len(topo) else float("nan"),
        "topo_std": float(np.std(topo)) if len(topo) else float("nan"),
    }
    benchmark_path = dump_dir / f"benchmark_fthmc_jax_L{args.lattice_size}_beta{beta_tag}_nsteps{args.n_steps}_{args.save_tag}.json"
    benchmark_path.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f">>> Benchmark JSON written to {benchmark_path}")
    print(f">>> Total JAX FT-HMC time: {benchmark['total_time_sec']:.2f} seconds")


if __name__ == "__main__":
    main()
