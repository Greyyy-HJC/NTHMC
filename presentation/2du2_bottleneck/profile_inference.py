"""Profile production 2du2 base FT-HMC inference against the L16 evaluation path.

Constructs HMCU2FT the same way as 2du2/evaluation/base/compare_fthmc.py, times the
real compiled Metropolis step plus the Python run-loop extras, and dumps
jax.stages.Compiled.cost_analysis for that graph and its pieces.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nthmc.core.jax_env import bootstrap_cuda_wheel_paths, preconfigure_platform_from_argv, set_platform


bootstrap_cuda_wheel_paths()
preconfigure_platform_from_argv()

import jax
import jax.numpy as jnp

from nthmc.u2.field_transform import FieldTransformation
from nthmc.u2.u2_fthmc import HMCU2FT
from nthmc.u2.u2_hmc import HMCU2
from nthmc.u2.u2_observables import (
    format_beta,
    identity_field,
    plaquette_mean_from_field,
    set_seed,
    topology_from_field,
    u2_exp,
    u2_mul,
    u2_normalize,
)


def choose_device(device: str) -> str:
    if device == "cuda":
        device = "gpu"
    if device != "auto":
        set_platform(device)
    return device


def dump_device_tag(requested: str) -> str:
    if requested in {"gpu", "cuda"}:
        return "gpu"
    if requested == "cpu":
        return "cpu"
    backend = jax.default_backend()
    return "gpu" if backend in {"gpu", "cuda"} else backend


def _lscpu_value(text: str, key: str) -> str | None:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            value = line.split(":", 1)[1].strip()
            return value or None
    return None


def collect_host_resources() -> dict[str, Any]:
    lscpu_text = ""
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            lscpu_text = result.stdout
    except OSError:
        lscpu_text = ""
    physical_cores: int | None = None
    try:
        result = subprocess.run(
            ["lscpu", "-p=Core,Socket"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            cores = {
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip() and not line.startswith("#")
            }
            physical_cores = len(cores) if cores else None
    except OSError:
        physical_cores = None
    sockets = _lscpu_value(lscpu_text, "Socket(s)")
    cores_per_socket = _lscpu_value(lscpu_text, "Core(s) per socket")
    threads_per_core = _lscpu_value(lscpu_text, "Thread(s) per core")
    if physical_cores is None and sockets and cores_per_socket:
        try:
            physical_cores = int(sockets) * int(cores_per_socket)
        except ValueError:
            physical_cores = None
    mem_total_kb: int | None = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                    break
    except OSError:
        mem_total_kb = None
    hostname = None
    try:
        hostname = os.uname().nodename
    except AttributeError:
        hostname = None
    return {
        "hostname": hostname,
        "cpu_model": _lscpu_value(lscpu_text, "Model name"),
        "sockets": sockets,
        "cores_per_socket": cores_per_socket,
        "threads_per_core": threads_per_core,
        "logical_cpus": _lscpu_value(lscpu_text, "CPU(s)"),
        "nproc": os.cpu_count(),
        "physical_cores": physical_cores,
        "mem_total_kb": mem_total_kb,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "jax_platform_name": os.environ.get("JAX_PLATFORM_NAME"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile 2du2 base FT-HMC inference")
    parser.add_argument("--lattice_size", type=int, default=16)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--train_beta", type=float, default=10.0)
    parser.add_argument("--n_configs", type=int, default=2048)
    parser.add_argument("--n_thermalization", type=int, default=200)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--ft_step_size", type=float, default=0.10)
    parser.add_argument("--hmc_step_size", type=float, default=0.10)
    parser.add_argument("--n_profile_steps", type=int, default=32)
    parser.add_argument("--n_kernel_repeats", type=int, default=8)
    parser.add_argument("--rand_seed", type=int, default=1107)
    parser.add_argument("--model_tag", type=str, default="base")
    parser.add_argument("--save_tag", type=str, default="")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--eval_log", type=Path, default=None)
    parser.add_argument("--hmc_log", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu", "cuda"])
    parser.add_argument("--dump_suffix", type=str, default="")
    return parser.parse_args()


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if hasattr(obj, "_asdict"):
        return to_jsonable(obj._asdict())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return to_jsonable(obj.item())
        except (ValueError, TypeError):
            pass
    return str(obj)


def summarize_cost(cost: Any) -> dict[str, Any]:
    if cost is None:
        return {"flops": None, "bytes_accessed": None, "raw": None}
    items = cost if isinstance(cost, list) else [cost]
    flops = 0.0
    bytes_accessed = 0.0
    found_flops = False
    found_bytes = False
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("flops") is not None:
            flops += float(item["flops"])
            found_flops = True
        raw_bytes = item.get("bytes accessed", item.get("bytes_accessed"))
        if raw_bytes is not None:
            bytes_accessed += float(raw_bytes)
            found_bytes = True
    return {
        "flops": flops if found_flops else None,
        "bytes_accessed": bytes_accessed if found_bytes else None,
        "raw": to_jsonable(cost),
    }


def summarize_memory(memory: Any) -> Any:
    return to_jsonable(memory)


def last_match(pattern: str, text: str) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    return matches[-1] if matches else None


def last_float(pattern: str, text: str) -> float | None:
    match = last_match(pattern, text)
    return float(match.group(1)) if match else None


def tqdm_step_seconds(text: str, desc: str) -> dict[str, float | None]:
    rows = re.findall(
        rf"{re.escape(desc)}:\s+\d+%\|.*?\|\s+(\d+)/(\d+)\s+\[(?P<elapsed>[^,]+)<(?P<remain>[^,]+),\s+(?P<rate>[\d.]+)s/it\]",
        text,
    )
    if not rows:
        return {"first_s_per_it": None, "last_s_per_it": None, "n_logged": 0}
    rates = [float(row[4]) for row in rows]
    return {
        "first_s_per_it": rates[0],
        "last_s_per_it": rates[-1],
        "n_logged": len(rates),
        "projected_remaining_h": float(rows[0][1]) * rates[0] / 3600.0,
    }


def parse_eval_log(path: Path | None, kind: str) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    text = path.read_text(errors="replace").replace("\r", "\n")
    if kind == "fthmc":
        therm = tqdm_step_seconds(text, "Thermalizing FT-HMC")
        run = tqdm_step_seconds(text, "Running FT-HMC")
        return {
            "path": str(path),
            "model_load_s": last_float(r">>> Model loaded in ([0-9.]+) seconds", text),
            "therm_s": last_float(r">>> FT thermalization completed in ([0-9.]+) seconds", text),
            "run_s": last_float(r">>> FT-HMC run completed in ([0-9.]+) seconds", text),
            "total_s": last_float(r">>> Total FT-HMC time: ([0-9.]+) seconds", text),
            "first_compile_s_per_it": therm.get("first_s_per_it"),
            "therm_steady_s_per_it": therm.get("last_s_per_it"),
            "therm_projected_h": therm.get("projected_remaining_h"),
            "run_first_s_per_it": run.get("first_s_per_it"),
            "run_steady_s_per_it": run.get("last_s_per_it"),
            "cuda_graph_oom": "instantiate CUDA graph" in text,
        }
    therm = tqdm_step_seconds(text, "Thermalizing")
    run = tqdm_step_seconds(text, "Running HMC")
    return {
        "path": str(path),
        "therm_s": last_float(r">>> HMC thermalization completed in ([0-9.]+) seconds", text),
        "run_s": last_float(r">>> HMC run completed in ([0-9.]+) seconds", text),
        "total_s": last_float(r">>> Total HMC time: ([0-9.]+) seconds", text),
        "therm_steady_s_per_it": therm.get("last_s_per_it"),
        "run_steady_s_per_it": run.get("last_s_per_it"),
    }


def compile_callable(fn: Callable, args: tuple[Any, ...]) -> Any:
    if hasattr(fn, "trace"):
        return fn.trace(*args).lower().compile()
    if hasattr(fn, "lower"):
        return fn.lower(*args).compile()
    jitted = jax.jit(fn)
    if hasattr(jitted, "trace"):
        return jitted.trace(*args).lower().compile()
    return jitted.lower(*args).compile()


def profile_compiled(compiled: Any, args: tuple[Any, ...], n_repeats: int) -> dict[str, Any]:
    warmup = compiled(*args)
    jax.block_until_ready(warmup)
    times: list[float] = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        times.append(time.perf_counter() - start)
    cost = compiled.cost_analysis() if hasattr(compiled, "cost_analysis") else None
    memory = compiled.memory_analysis() if hasattr(compiled, "memory_analysis") else None
    return {
        "n_repeats": n_repeats,
        "times_s": times,
        "mean_s": float(np.mean(times)) if times else None,
        "std_s": float(np.std(times)) if times else None,
        "cost": summarize_cost(cost),
        "memory": summarize_memory(memory),
    }


def aot_kernel(name: str, fn: Callable, args: tuple[Any, ...], n_repeats: int) -> dict[str, Any]:
    print(f">>> AOT {name}")
    compile_start = time.perf_counter()
    compiled = compile_callable(jax.jit(fn), args)
    compile_s = time.perf_counter() - compile_start
    result = profile_compiled(compiled, args, n_repeats)
    result["name"] = name
    result["compile_s"] = compile_s
    flops = result["cost"]["flops"]
    mean_s = result["mean_s"]
    print(
        f"    compile {compile_s:.2f}s, mean {mean_s:.4f}s, "
        f"flops {flops if flops is not None else 'n/a'}"
    )
    return result


def time_metropolis(hmc: HMCU2FT | HMCU2, links: Any, n_steps: int) -> tuple[Any, dict[str, Any]]:
    compile_start = time.perf_counter()
    links, accepted, hamiltonian = hmc.metropolis_step(links)
    first_s = time.perf_counter() - compile_start
    times: list[float] = []
    for _ in range(n_steps):
        step_start = time.perf_counter()
        links, accepted, hamiltonian = hmc.metropolis_step(links)
        times.append(time.perf_counter() - step_start)
    return links, {
        "first_step_s": first_s,
        "n_steady_steps": n_steps,
        "steady_times_s": times,
        "steady_mean_s": float(np.mean(times)) if times else None,
        "steady_std_s": float(np.std(times)) if times else None,
        "last_accepted": bool(accepted),
        "last_hamiltonian": float(hamiltonian),
    }


def make_jitted_observables(transform: Callable, *, include_topology: bool) -> Callable:
    def obs(links: Any):
        links_ori = u2_normalize(transform(links))
        plaq = plaquette_mean_from_field(links_ori)
        if include_topology:
            return links_ori, plaq, topology_from_field(links_ori)
        return links_ori, plaq

    return jax.jit(obs)


def time_observable_extras(obs_fn: Callable, links: Any, n_repeats: int, *, include_topology: bool) -> dict[str, Any]:
    warmup = obs_fn(links)
    jax.block_until_ready(warmup)
    if include_topology:
        _, plaq, topo = warmup
        float(plaq)
        float(topo)
    else:
        _, plaq = warmup
        float(plaq)
    times: list[float] = []
    plaq = 0.0
    topo = 0.0
    for _ in range(n_repeats):
        start = time.perf_counter()
        result = obs_fn(links)
        if include_topology:
            _, plaq, topo = result
            float(plaq)
            float(topo)
        else:
            _, plaq = result
            float(plaq)
        times.append(time.perf_counter() - start)
    _ = (plaq, topo)
    return {
        "n_repeats": n_repeats,
        "include_topology": include_topology,
        "jitted": True,
        "times_s": times,
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
    }


def force_calls_per_trajectory(n_steps: int) -> int:
    return 2 * n_steps + 1


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    set_seed(args.rand_seed)

    script_dir = Path(__file__).resolve().parent
    save_tag = args.save_tag or f"{args.model_tag}_train_b{format_beta(args.train_beta)}_L{args.lattice_size}_{args.rand_seed}"
    output_dir = args.output_dir.resolve() if args.output_dir is not None else script_dir
    dump_dir = output_dir / "dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    model_dir = REPO_ROOT / "2du2" / "artifacts" / "models"
    eval_log = args.eval_log or (
        REPO_ROOT
        / "2du2"
        / "evaluation"
        / "base"
        / "logs"
        / f"fthmc_{args.model_tag}_L{args.lattice_size}_train_b{format_beta(args.train_beta)}_eval_b{format_beta(args.beta)}_{args.rand_seed}.log"
    )
    hmc_log = args.hmc_log or (
        REPO_ROOT
        / "2du2"
        / "evaluation"
        / "hmc"
        / "logs"
        / f"hmc_L{args.lattice_size}_b{format_beta(args.beta)}_{args.rand_seed}.log"
    )

    device_tag = dump_device_tag(device)
    host_resources = collect_host_resources()

    print("=" * 60)
    print(">>> U(2) JAX base FT-HMC inference profile")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print(f"resolved_device: {device}")
    print(f"dump_device_tag: {device_tag}")
    print(f"save_tag: {save_tag}")
    print(f"XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
    print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")
    print(f"hostname: {host_resources.get('hostname')}")
    print(f"nproc: {host_resources.get('nproc')}")
    print(f"physical_cores: {host_resources.get('physical_cores')}")
    print(f"mem_total_kb: {host_resources.get('mem_total_kb')}")
    print(f"OMP_NUM_THREADS: {host_resources.get('omp_num_threads')}")
    print("=" * 60)

    eval_log_info = parse_eval_log(eval_log, "fthmc")
    hmc_log_info = parse_eval_log(hmc_log, "hmc")
    if eval_log_info:
        print(
            f">>> Eval log total {eval_log_info.get('total_s')} s, "
            f"first compile {eval_log_info.get('first_compile_s_per_it')} s/it, "
            f"run {eval_log_info.get('run_steady_s_per_it')} s/it"
        )

    field_transform = FieldTransformation(
        args.lattice_size,
        device=device,
        n_subsets=8,
        if_check_jac=False,
        model_tag=args.model_tag,
        save_tag=save_tag,
        model_dir=model_dir,
        plot_dir=output_dir / "plots",
        dump_dir=dump_dir,
    )
    load_start = time.perf_counter()
    field_transform.load_best_model(args.train_beta)
    model_load_s = time.perf_counter() - load_start
    print(f">>> Model loaded in {model_load_s:.2f} seconds")

    ft_hmc = HMCU2FT(
        args.lattice_size,
        args.beta,
        args.n_thermalization,
        args.n_steps,
        args.ft_step_size,
        field_transformation=field_transform.field_transformation,
        compute_jac_logdet=field_transform.compute_jac_logdet,
        observable_field_transformation=field_transform.field_transformation,
        device=device,
        tune_step_size=False,
        seed=args.rand_seed,
    )
    hmc = HMCU2(
        args.lattice_size,
        args.beta,
        args.n_thermalization,
        args.n_steps,
        args.hmc_step_size,
        device=device,
        tune_step_size=False,
        seed=args.rand_seed,
    )

    ft_links = ft_hmc.initialize()
    print(">>> Timing production FT-HMC metropolis_step")
    ft_links, ft_wall = time_metropolis(ft_hmc, ft_links, args.n_profile_steps)
    print(
        f"    first {ft_wall['first_step_s']:.2f}s, "
        f"steady mean {ft_wall['steady_mean_s']:.4f}s"
    )
    ft_obs_therm = time_observable_extras(
        ft_hmc.therm_observables, ft_links, args.n_profile_steps, include_topology=False
    )
    ft_obs_run = time_observable_extras(
        ft_hmc.run_observables, ft_links, args.n_profile_steps, include_topology=True
    )
    print(
        f"    therm extras {ft_obs_therm['mean_s']:.4f}s, "
        f"run extras {ft_obs_run['mean_s']:.4f}s"
    )

    hmc_links = hmc.initialize()
    print(">>> Timing production HMC metropolis_step")
    hmc_links, hmc_wall = time_metropolis(hmc, hmc_links, args.n_profile_steps)
    print(
        f"    first {hmc_wall['first_step_s']:.2f}s, "
        f"steady mean {hmc_wall['steady_mean_s']:.4f}s"
    )
    hmc_obs_run = time_observable_extras(
        make_jitted_observables(u2_normalize, include_topology=True),
        hmc_links,
        args.n_profile_steps,
        include_topology=True,
    )

    example = identity_field(args.lattice_size)
    example_batch = example[jnp.newaxis, ...]
    key = jax.random.PRNGKey(args.rand_seed)
    dt = jnp.asarray(args.ft_step_size, dtype=example.dtype)
    hmc_dt = jnp.asarray(args.hmc_step_size, dtype=example.dtype)
    subset_params = field_transform.params["subsets"][0]
    plaq_features, rect_features = field_transform._cnn_features_with_masks(
        example_batch, field_transform._plaq_masks[0], field_transform._rect_masks[0]
    )

    def cnn_apply(plaq: Any, rect: Any) -> Any:
        return field_transform.model.apply(subset_params, plaq, rect)

    def forward(links: Any) -> Any:
        return field_transform.field_transformation(links)

    def jac_logdet(links: Any) -> Any:
        return field_transform.compute_jac_logdet(links[jnp.newaxis, ...])[0]

    def new_action(links: Any) -> Any:
        return ft_hmc.new_action(links)

    def force(links: Any) -> Any:
        def varied_action(algebra: Any) -> Any:
            return ft_hmc.new_action(u2_mul(u2_exp(algebra), links))

        return jax.grad(varied_action)(jnp.zeros((*links.shape[:-1], 4), dtype=links.dtype))

    def hmc_force(links: Any) -> Any:
        return hmc.force(links)

    kernels = {
        "cnn_apply": aot_kernel("cnn_apply", cnn_apply, (plaq_features, rect_features), args.n_kernel_repeats),
        "forward": aot_kernel("forward", forward, (example,), args.n_kernel_repeats),
        "jac_logdet": aot_kernel("jac_logdet", jac_logdet, (example,), args.n_kernel_repeats),
        "new_action": aot_kernel("new_action", new_action, (example,), args.n_kernel_repeats),
        "force": aot_kernel("force", force, (example,), args.n_kernel_repeats),
        "hmc_force": aot_kernel("hmc_force", hmc_force, (example,), args.n_kernel_repeats),
    }

    print(">>> AOT production FT-HMC compiled step")
    if ft_hmc._compiled_step is None:
        ft_hmc._compiled_step = ft_hmc._make_step()
    ft_step_compile_start = time.perf_counter()
    ft_step_compiled = compile_callable(ft_hmc._compiled_step, (example, key, dt))
    ft_step_aot_s = time.perf_counter() - ft_step_compile_start
    kernels["fthmc_step"] = profile_compiled(ft_step_compiled, (example, key, dt), args.n_kernel_repeats)
    kernels["fthmc_step"]["name"] = "fthmc_step"
    kernels["fthmc_step"]["compile_s"] = ft_step_aot_s
    print(
        f"    compile {ft_step_aot_s:.2f}s, mean {kernels['fthmc_step']['mean_s']:.4f}s, "
        f"flops {kernels['fthmc_step']['cost']['flops']}"
    )

    print(">>> AOT production HMC compiled step")
    if hmc._compiled_step is None:
        hmc._compiled_step = hmc._make_step()
    hmc_step_compile_start = time.perf_counter()
    hmc_step_compiled = compile_callable(hmc._compiled_step, (example, key, hmc_dt))
    hmc_step_aot_s = time.perf_counter() - hmc_step_compile_start
    kernels["hmc_step"] = profile_compiled(hmc_step_compiled, (example, key, hmc_dt), args.n_kernel_repeats)
    kernels["hmc_step"]["name"] = "hmc_step"
    kernels["hmc_step"]["compile_s"] = hmc_step_aot_s
    print(
        f"    compile {hmc_step_aot_s:.2f}s, mean {kernels['hmc_step']['mean_s']:.4f}s, "
        f"flops {kernels['hmc_step']['cost']['flops']}"
    )

    n_force = force_calls_per_trajectory(args.n_steps)
    ft_loop_mean = (ft_wall["steady_mean_s"] or 0.0) + ft_obs_run["mean_s"]
    hmc_loop_mean = (hmc_wall["steady_mean_s"] or 0.0) + hmc_obs_run["mean_s"]
    projected = {
        "n_thermalization": args.n_thermalization,
        "n_configs": args.n_configs,
        "force_calls_per_trajectory": n_force,
        "new_action_calls_per_trajectory": 2,
        "projected_therm_s": (ft_wall["first_step_s"] or 0.0)
        + (args.n_thermalization - 1) * ((ft_wall["steady_mean_s"] or 0.0) + ft_obs_therm["mean_s"]),
        "projected_run_s": args.n_configs * ft_loop_mean,
        "projected_total_s": (ft_wall["first_step_s"] or 0.0)
        + (args.n_thermalization - 1) * ((ft_wall["steady_mean_s"] or 0.0) + ft_obs_therm["mean_s"])
        + args.n_configs * ft_loop_mean
        + model_load_s,
        "force_times_calls_s": None if kernels["force"]["mean_s"] is None else n_force * kernels["force"]["mean_s"],
        "hmc_projected_run_s": args.n_configs * hmc_loop_mean,
    }

    payload = {
        "config": {
            "lattice_size": args.lattice_size,
            "beta": args.beta,
            "train_beta": args.train_beta,
            "n_steps": args.n_steps,
            "n_configs": args.n_configs,
            "n_thermalization": args.n_thermalization,
            "ft_step_size": args.ft_step_size,
            "hmc_step_size": args.hmc_step_size,
            "n_profile_steps": args.n_profile_steps,
            "n_kernel_repeats": args.n_kernel_repeats,
            "rand_seed": args.rand_seed,
            "model_tag": args.model_tag,
            "save_tag": save_tag,
            "dump_suffix": args.dump_suffix,
            "device": device,
            "device_tag": device_tag,
            "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(item) for item in jax.devices()],
            "host": host_resources,
        },
        "eval_log": eval_log_info,
        "hmc_log": hmc_log_info,
        "model_load_s": model_load_s,
        "wall_clock": {
            "fthmc": {
                **ft_wall,
                "therm_observable_s": ft_obs_therm,
                "run_observable_s": ft_obs_run,
                "python_loop_mean_s": ft_loop_mean,
            },
            "hmc": {
                **hmc_wall,
                "run_observable_s": hmc_obs_run,
                "python_loop_mean_s": hmc_loop_mean,
            },
        },
        "kernels": kernels,
        "extrapolation": projected,
    }
    dump_stem = (
        f"inference_profile_L{args.lattice_size}_beta{format_beta(args.beta)}_"
        f"nsteps{args.n_steps}_{save_tag}"
    )
    if args.dump_suffix:
        dump_stem = f"{dump_stem}_{args.dump_suffix}"
    dump_path = dump_dir / f"{dump_stem}_{device_tag}.json"
    dump_path.write_text(json.dumps(to_jsonable(payload), indent=2))
    print(f">>> Wrote {dump_path}")
    print(
        f">>> Projected FT-HMC total {projected['projected_total_s']:.1f}s "
        f"({projected['projected_total_s'] / 3600.0:.2f}h) vs log {eval_log_info.get('total_s') if eval_log_info else 'n/a'}"
    )


if __name__ == "__main__":
    main()
