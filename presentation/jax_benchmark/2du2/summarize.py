"""Summarize the archived U(2) L32 historical-vs-current benchmark."""

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
    return (LOG_DIR / name).read_text(errors="replace")


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


def scalar_after(text: str, label: str) -> float | None:
    matches = re.findall(rf"{re.escape(label)}\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", text)
    return float(matches[-1]) if matches else None


def matching_hmc_block(text: str) -> str:
    marker = ">>> U(2) standard HMC evaluation"
    for block in text.split(marker)[1:]:
        candidate = marker + block
        if "beta: 10.0" in candidate and "rand_seed: 1029" in candidate:
            return candidate
    return ""


def completed_training_epochs(text: str) -> int | None:
    matches = re.findall(r"Epoch\s+(\d+)/(\d+)\s+-", text)
    return max((int(epoch) for epoch, _ in matches), default=None)


def elapsed_per_training_step(text: str, elapsed: float | None, steps_per_epoch: int = 69) -> float | None:
    epochs = completed_training_epochs(text)
    if elapsed is None or epochs is None or epochs <= 0:
        return None
    return elapsed / (epochs * steps_per_epoch)


def csv_scalar(name: str) -> float | None:
    values = np.loadtxt(DUMP_DIR / name, ndmin=1)
    return float(values[-1]) if values.size else None


def topo_stats(name: str) -> tuple[float | None, float | None, int]:
    values = np.loadtxt(DUMP_DIR / name, ndmin=1)
    if not values.size:
        return None, None, 0
    return float(np.mean(values)), float(np.std(values)), int(values.size)


def speedup(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or new <= 0:
        return None
    return old / new


def throughput(seconds: float | None) -> float | None:
    return N_CONFIGS / seconds if seconds is not None and seconds > 0 else None


def fmt(value: float | None, digits: int = 2) -> str:
    return "pending" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def duration(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "pending"
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes):02d}:{seconds:05.2f}"


def main() -> None:
    old_gauge = read_text("old_pytorch_gauge_L32_b10.log")
    new_gauge = read_text("new_jax_gauge_L32_b10.log")
    old_train = read_text("old_pytorch_training_L32_b10.0_1029.log")
    new_train = read_text("new_pytorch_training_L32_b10.0_1029.log")
    old_hmc = matching_hmc_block(read_text("old_pytorch_hmc_L32.log"))
    new_hmc = read_text("new_jax_hmc_L32_b10.0_1029.log")
    old_fthmc = read_text("old_pytorch_fthmc_L32_b10.0_1029.log")
    new_fthmc = read_text("new_jax_fthmc_L32_b10.0_1029.log")

    old_train_elapsed = hms_seconds(old_train, ">>> Total elapsed")
    new_train_elapsed = hms_seconds(new_train, ">>> Total elapsed")
    timings = {
        "gauge_pbs": (pbs_seconds(old_gauge), pbs_seconds(new_gauge)),
        "training_pbs": (pbs_seconds(old_train), pbs_seconds(new_train)),
        "training_reported_total": (old_train_elapsed, new_train_elapsed),
        "training_per_step": (
            elapsed_per_training_step(old_train, old_train_elapsed),
            elapsed_per_training_step(new_train, new_train_elapsed),
        ),
        "hmc_thermalization": (
            seconds_after(old_hmc, ">>> HMC thermalization completed in"),
            seconds_after(new_hmc, ">>> HMC thermalization completed in"),
        ),
        "hmc_run": (
            seconds_after(old_hmc, ">>> HMC run completed in"),
            seconds_after(new_hmc, ">>> HMC run completed in"),
        ),
        "hmc_reported_total": (
            seconds_after(old_hmc, ">>> Total HMC time:"),
            seconds_after(new_hmc, ">>> Total HMC time:"),
        ),
        "hmc_pbs": (None, pbs_seconds(new_hmc)),
        "fthmc_model_load": (
            seconds_after(old_fthmc, ">>> Model loaded in"),
            seconds_after(new_fthmc, ">>> Model loaded in"),
        ),
        "fthmc_thermalization_compile": (
            seconds_after(old_fthmc, ">>> FT thermalization completed in"),
            seconds_after(new_fthmc, ">>> FT thermalization completed in"),
        ),
        "fthmc_run": (
            seconds_after(old_fthmc, ">>> FT-HMC run completed in"),
            seconds_after(new_fthmc, ">>> FT-HMC run completed in"),
        ),
        "fthmc_reported_total": (
            seconds_after(old_fthmc, ">>> Total FT-HMC time:"),
            seconds_after(new_fthmc, ">>> Total FT-HMC time:"),
        ),
        "fthmc_pbs": (pbs_seconds(old_fthmc), pbs_seconds(new_fthmc)),
    }
    stage_totals = {
        "gauge generation": (*timings["gauge_pbs"], "single-job PBS walltime"),
        "model training": (*timings["training_reported_total"], "single-seed reported elapsed"),
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

    old_hmc_topo = topo_stats("old_pytorch_topo_hmc_L32_beta10.0_nsteps10_1029.csv")
    new_hmc_topo = topo_stats("new_jax_topo_hmc_L32_beta10.0_nsteps10_1029.csv")
    old_fthmc_topo = topo_stats("old_pytorch_topo_fthmc_L32_beta10.0_nsteps10_base_train_b10.0_L32_1029.csv")
    new_fthmc_topo = topo_stats("new_jax_topo_fthmc_L32_beta10.0_nsteps10_1029.csv")
    acceptances = {
        "old_hmc": csv_scalar("old_pytorch_accept_rate_hmc_L32_beta10.0_nsteps10_1029.csv"),
        "new_hmc": csv_scalar("new_jax_accept_rate_hmc_L32_beta10.0_nsteps10_1029.csv"),
        "old_fthmc": csv_scalar("old_pytorch_accept_rate_fthmc_L32_beta10.0_nsteps10_base_train_b10.0_L32_1029.csv"),
        "new_fthmc": csv_scalar("new_jax_accept_rate_fthmc_L32_beta10.0_nsteps10_1029.csv"),
    }
    old_hmc_run, new_hmc_run = timings["hmc_run"]
    old_fthmc_run, new_fthmc_run = timings["fthmc_run"]

    lines = [
        "# U(2) L32 Full-Pipeline Benchmark",
        "",
        "This benchmark compares the archived PyTorch L32 beta10 pipeline with the current JAX runtime and PyTorch training pipeline.",
        "FT-HMC uses each pipeline's own trained model, so this is a full-pipeline comparison rather than a same-checkpoint microbenchmark.",
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
            f"| HMC | old PyTorch | {fmt(acceptances['old_hmc'], 4)} | {fmt(scalar_after(old_hmc, '>>> Mean plaq:'), 6)} | {fmt(old_hmc_topo[0], 4)} | {fmt(old_hmc_topo[1], 4)} | {old_hmc_topo[2]} |",
            f"| HMC | new JAX | {fmt(acceptances['new_hmc'], 4)} | {fmt(scalar_after(new_hmc, '>>> Mean plaq:'), 6)} | {fmt(new_hmc_topo[0], 4)} | {fmt(new_hmc_topo[1], 4)} | {new_hmc_topo[2]} |",
            f"| FT-HMC | old PyTorch | {fmt(acceptances['old_fthmc'], 4)} | {fmt(scalar_after(old_fthmc, '>>> Mean plaq:'), 6)} | {fmt(old_fthmc_topo[0], 4)} | {fmt(old_fthmc_topo[1], 4)} | {old_fthmc_topo[2]} |",
            f"| FT-HMC | new JAX | {fmt(acceptances['new_fthmc'], 4)} | {fmt(scalar_after(new_fthmc, '>>> Mean plaq:'), 6)} | {fmt(new_fthmc_topo[0], 4)} | {fmt(new_fthmc_topo[1], 4)} | {new_fthmc_topo[2]} |",
        ]
    )
    finite_acceptances = [value for value in acceptances.values() if value is not None and math.isfinite(value)]
    if finite_acceptances and min(finite_acceptances) < 0.4:
        lines.extend(
            [
                "",
                "**Caveat:** the archived new-JAX HMC run has acceptance below 0.4. Its timing and the aggregate pipeline estimate describe the recorded run, but they are not an acceptance-matched performance baseline.",
            ]
        )
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {ROOT / 'timings.csv'}")
    print(f"Wrote {ROOT / 'summary.md'}")


if __name__ == "__main__":
    main()
