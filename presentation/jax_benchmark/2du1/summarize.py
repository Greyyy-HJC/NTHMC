"""Summarize the archived U(1) L32 historical-vs-current benchmark."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
DUMP_DIR = ROOT / "dumps"
N_CONFIGS = 4096


def read_text(name: str) -> str:
    path = LOG_DIR / name
    return path.read_text(errors="replace") if path.exists() else ""


def seconds_after(text: str, label: str) -> float | None:
    matches = re.findall(rf"{re.escape(label)}\s*([0-9]+(?:\.[0-9]+)?)\s+seconds", text)
    return float(matches[-1]) if matches else None


def pbs_seconds(text: str) -> float | None:
    matches = re.findall(r"Total time:\s*(\d+)h\s+(\d+)m\s+(\d+)s", text)
    if not matches:
        return None
    hours, minutes, seconds = map(int, matches[-1])
    return float(hours * 3600 + minutes * 60 + seconds)


def hms_seconds(text: str, label: str) -> float | None:
    matches = re.findall(rf"{re.escape(label)}:\s*(\d+):(\d\d):(\d\d)", text)
    if not matches:
        return None
    hours, minutes, seconds = map(int, matches[-1])
    return float(hours * 3600 + minutes * 60 + seconds)


def first_training_block(text: str) -> str:
    marker = ">>> U(1) base field-transformation training"
    start = text.find(marker)
    if start < 0:
        return ""
    end = text.find(marker, start + len(marker))
    return text[start : end if end >= 0 else len(text)]


def scalar_after(text: str, label: str) -> float | None:
    matches = re.findall(rf"{re.escape(label)}\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", text)
    return float(matches[-1]) if matches else None


def csv_scalar(name: str) -> float | None:
    path = DUMP_DIR / name
    if not path.exists():
        return None
    values = np.loadtxt(path, ndmin=1)
    return float(values[-1]) if values.size else None


def topo_stats(name: str) -> tuple[float | None, float | None, int | None]:
    path = DUMP_DIR / name
    if not path.exists():
        return None, None, None
    values = np.loadtxt(path, ndmin=1)
    if not values.size:
        return None, None, 0
    return float(np.mean(values)), float(np.std(values)), int(values.size)


def speedup(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or new <= 0:
        return None
    return old / new


def throughput(seconds: float | None) -> float | None:
    return N_CONFIGS / seconds if seconds is not None and seconds > 0 else None


def duration(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "pending"
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes):02d}:{seconds:05.2f}"


def fmt(value: float | None, digits: int = 2) -> str:
    return "pending" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def execution_mode(old_fthmc: str) -> str:
    if ">>> completed_execution_mode: pytorch_compiled_force_path" in old_fthmc:
        return "compiled force path"
    if ">>> fallback_execution_mode: pytorch_eager" in old_fthmc:
        return "eager fallback"
    return "pending"


def main() -> None:
    old_hmc = read_text("old_pytorch_hmc_L32_b4.0_1029.log")
    new_hmc = read_text("new_jax_hmc_L32_b4.0_1029.log")
    old_fthmc = read_text("old_pytorch_fthmc_L32_b4.0_1029.log")
    new_fthmc = read_text("new_jax_fthmc_L32_b4.0_1029.log")
    old_gauge = read_text("old_pytorch_gauge_L32_b4.0_1331.log")
    new_gauge = read_text("new_jax_gauge_L32_b4.0_1331.log")
    old_train = read_text("old_pytorch_training_L32_b4.0.log")
    new_train = read_text("new_pytorch_training_L32_b4.0_1029.log")

    timings = {
        "gauge_pbs": (pbs_seconds(old_gauge), pbs_seconds(new_gauge)),
        "training_seed1029": (hms_seconds(first_training_block(old_train), ">>> Total elapsed"), hms_seconds(new_train, ">>> Total elapsed")),
        "hmc_thermalization": (seconds_after(old_hmc, ">>> HMC thermalization completed in"), seconds_after(new_hmc, ">>> HMC thermalization completed in")),
        "hmc_run": (seconds_after(old_hmc, ">>> HMC run completed in"), seconds_after(new_hmc, ">>> HMC run completed in")),
        "hmc_reported_total": (seconds_after(old_hmc, ">>> Total HMC time:"), seconds_after(new_hmc, ">>> Total HMC time:")),
        "hmc_pbs": (pbs_seconds(old_hmc), pbs_seconds(new_hmc)),
        "fthmc_model_load": (seconds_after(old_fthmc, ">>> Model loaded in"), seconds_after(new_fthmc, ">>> Model loaded in")),
        "fthmc_thermalization_compile": (seconds_after(old_fthmc, ">>> FT thermalization completed in"), seconds_after(new_fthmc, ">>> FT thermalization completed in")),
        "fthmc_run": (seconds_after(old_fthmc, ">>> FT-HMC run completed in"), seconds_after(new_fthmc, ">>> FT-HMC run completed in")),
        "fthmc_reported_total": (seconds_after(old_fthmc, ">>> Total FT-HMC time:"), seconds_after(new_fthmc, ">>> Total FT-HMC time:")),
        "fthmc_pbs": (pbs_seconds(old_fthmc), pbs_seconds(new_fthmc)),
    }

    stage_totals = {
        "gauge generation": (*timings["gauge_pbs"], "single-job PBS walltime"),
        "model training": (*timings["training_seed1029"], "single-seed reported elapsed"),
        "HMC evaluation": (*timings["hmc_reported_total"], "reported sampler total"),
        "FT-HMC evaluation": (*timings["fthmc_reported_total"], "reported sampler total"),
    }
    pipeline_old = sum(old for old, _, _ in stage_totals.values() if old is not None)
    pipeline_new = sum(new for _, new, _ in stage_totals.values() if new is not None)

    with (ROOT / "timings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "metric", "old_pytorch_seconds", "new_seconds", "old_over_new_speedup", "timing_basis"])
        for metric, (old, new) in timings.items():
            writer.writerow(["detail", metric, fmt(old), fmt(new), fmt(speedup(old, new), 3), "raw log metric"])
        for stage, (old, new, basis) in stage_totals.items():
            writer.writerow(["stage_total", stage, fmt(old), fmt(new), fmt(speedup(old, new), 3), basis])
        writer.writerow(
            ["pipeline_total", "sequential one-seed pipeline", fmt(pipeline_old), fmt(pipeline_new), fmt(speedup(pipeline_old, pipeline_new), 3), "sum of stage totals"]
        )

    old_hmc_topo = topo_stats("old_pytorch_topo_hmc_L32_beta4.0_nsteps10_1029.csv")
    new_hmc_topo = topo_stats("new_jax_topo_hmc_L32_beta4.0_nsteps10_1029.csv")
    old_fthmc_topo = topo_stats("old_pytorch_topo_fthmc_L32_beta4.0_nsteps10_base_train_b4.0_L32_1029.csv")
    new_fthmc_topo = topo_stats("new_jax_topo_fthmc_L32_beta4.0_nsteps10_1029.csv")
    old_hmc_run, new_hmc_run = timings["hmc_run"]
    old_fthmc_run, new_fthmc_run = timings["fthmc_run"]
    acceptances = {
        "old_hmc": csv_scalar("old_pytorch_accept_rate_hmc_L32_beta4.0_nsteps10_1029.csv"),
        "new_hmc": csv_scalar("new_jax_accept_rate_hmc_L32_beta4.0_nsteps10_1029.csv"),
        "old_fthmc": csv_scalar("old_pytorch_accept_rate_fthmc_L32_beta4.0_nsteps10_base_train_b4.0_L32_1029.csv"),
        "new_fthmc": csv_scalar("new_jax_accept_rate_fthmc_L32_beta4.0_nsteps10_1029.csv"),
    }

    lines = [
        "# U(1) L32 Full-Pipeline Benchmark",
        "",
        "Historical source: `e3847a90570f2434117177c80872c9306b69a93b`. Current evaluation uses JAX while training remains PyTorch.",
        "FT-HMC uses each pipeline's own trained model; this is not a same-checkpoint backend microbenchmark.",
        f"Historical FT-HMC execution mode: **{execution_mode(old_fthmc)}**.",
        "Each implementation was independently tuned to the same acceptance target before the 4096-configuration run; step sizes therefore need not match.",
        "Stage estimates cover one training seed followed by one HMC and one FT-HMC evaluation; queue waits, probes, and optional tuning are excluded.",
        "",
        "## Stage Total Estimates",
        "",
        "| Stage | Timing basis | Old PyTorch | New | Speedup |",
        "|---|---|---:|---:|---:|",
    ]
    for stage, (old, new, basis) in stage_totals.items():
        lines.append(
            f"| {stage} | {basis} | {fmt(old)} s ({duration(old)}) | {fmt(new)} s ({duration(new)}) | {fmt(speedup(old, new), 3)} |"
        )
    lines.extend(
        [
            f"| **sequential pipeline** | sum of stage totals | **{fmt(pipeline_old)} s ({duration(pipeline_old)})** | **{fmt(pipeline_new)} s ({duration(pipeline_new)})** | **{fmt(speedup(pipeline_old, pipeline_new), 3)}** |",
        "",
        "## Evaluation Parameters",
        "",
        "| Sampler | Old PyTorch step size | New JAX step size |",
        "|---|---:|---:|",
        f"| HMC | {fmt(scalar_after(old_hmc, 'step_size:'), 3)} | {fmt(scalar_after(new_hmc, 'step_size:'), 3)} |",
        f"| FT-HMC | {fmt(scalar_after(old_fthmc, 'ft_step_size:'), 3)} | {fmt(scalar_after(new_fthmc, 'ft_step_size:'), 3)} |",
        "",
        "## Timing Details",
        "",
        "| Metric | Old PyTorch (s) | New (s) | Speedup |",
        "|---|---:|---:|---:|",
        ]
    )
    for metric, (old, new) in timings.items():
        lines.append(f"| {metric} | {fmt(old)} | {fmt(new)} | {fmt(speedup(old, new), 3)} |")
    lines.extend(
        [
            "",
            "## Evaluation Throughput",
            "",
            "| Sampler | Old configs/s | New configs/s |",
            "|---|---:|---:|",
            f"| HMC | {fmt(throughput(old_hmc_run), 3)} | {fmt(throughput(new_hmc_run), 3)} |",
            f"| FT-HMC | {fmt(throughput(old_fthmc_run), 3)} | {fmt(throughput(new_fthmc_run), 3)} |",
            "",
            "## Sanity Checks",
            "",
            "| Sampler | Implementation | Acceptance | Mean plaquette | Topology mean | Topology std | Samples |",
            "|---|---|---:|---:|---:|---:|---:|",
            f"| HMC | old PyTorch | {fmt(acceptances['old_hmc'], 4)} | {fmt(scalar_after(old_hmc, '>>> Mean plaq:'), 6)} | {fmt(old_hmc_topo[0], 4)} | {fmt(old_hmc_topo[1], 4)} | {old_hmc_topo[2] or 'pending'} |",
            f"| HMC | new JAX | {fmt(acceptances['new_hmc'], 4)} | {fmt(scalar_after(new_hmc, '>>> Mean plaq:'), 6)} | {fmt(new_hmc_topo[0], 4)} | {fmt(new_hmc_topo[1], 4)} | {new_hmc_topo[2] or 'pending'} |",
            f"| FT-HMC | old PyTorch | {fmt(acceptances['old_fthmc'], 4)} | {fmt(scalar_after(old_fthmc, '>>> Mean plaq:'), 6)} | {fmt(old_fthmc_topo[0], 4)} | {fmt(old_fthmc_topo[1], 4)} | {old_fthmc_topo[2] or 'pending'} |",
            f"| FT-HMC | new JAX | {fmt(acceptances['new_fthmc'], 4)} | {fmt(scalar_after(new_fthmc, '>>> Mean plaq:'), 6)} | {fmt(new_fthmc_topo[0], 4)} | {fmt(new_fthmc_topo[1], 4)} | {new_fthmc_topo[2] or 'pending'} |",
        ]
    )
    finite_acceptances = [value for value in acceptances.values() if value is not None and math.isfinite(value)]
    if finite_acceptances and min(finite_acceptances) < 0.4:
        lines.extend(["", "Warning: at least one final run has acceptance below 0.4; do not use it as an acceptance-matched performance baseline."])
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {ROOT / 'timings.csv'}")
    print(f"Wrote {ROOT / 'summary.md'}")


if __name__ == "__main__":
    main()
