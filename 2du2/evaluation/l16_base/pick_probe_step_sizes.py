#!/usr/bin/env python3
"""Select FT_STEP_SIZE per beta from probe_step_size_log.csv (acceptance in [0.55, 0.95])."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ACCEPT_MIN = 0.55
ACCEPT_MAX = 0.95
DIR = Path(__file__).resolve().parent


def probe_log_paths() -> list[Path]:
    shards = sorted(DIR.glob("probe_step_size_log_*.csv"))
    merged = DIR / "probe_step_size_log.csv"
    if merged.exists() and merged not in shards:
        return [merged, *shards]
    return shards or ([merged] if merged.exists() else [])


def main() -> int:
    paths = probe_log_paths()
    if not paths:
        print("Missing probe_step_size_log_*.csv; run sub_probe.sh first.", file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open() as f:
            rows.extend(csv.DictReader(f))

    by_beta: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        try:
            beta = row["beta"]
            step = float(row["ft_step_size"])
            rate = float(row["accept_rate"])
        except (KeyError, ValueError):
            continue
        by_beta.setdefault(beta, []).append((step, rate))

    print("# Selected FT step sizes for sub_gen.sh")
    exports = []
    for beta in sorted(by_beta, key=float):
        in_window = [(s, r) for s, r in by_beta[beta] if ACCEPT_MIN <= r <= ACCEPT_MAX]
        pool = in_window or by_beta[beta]
        step, rate = min(pool, key=lambda x: abs(x[1] - 0.75))
        key = beta.replace(".0", "")
        exports.append(f"export FT_STEP_SIZE_{key}={step}")
        status = "pass" if ACCEPT_MIN <= rate <= ACCEPT_MAX else "FAIL"
        print(f"beta={beta}: ft_step_size={step}, accept={rate:.4f} [{status}]")

    if not exports:
        print("No valid probe rows in log.", file=sys.stderr)
        return 1

    required = {"10", "12", "14", "16"}
    got = {line.split("=")[0].rsplit("_", 1)[-1] for line in exports}
    if got != required:
        print(f"Expected betas {required}, got {got}.", file=sys.stderr)
        return 1

    print()
    print("\n".join(exports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
