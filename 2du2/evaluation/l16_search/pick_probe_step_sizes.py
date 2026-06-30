#!/usr/bin/env python3
"""Select FT_STEP_SIZE per (candidate, beta) from probe_step_size_log_{cand}_{beta}.csv."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ACCEPT_MIN = 0.55
ACCEPT_MAX = 0.95
DIR = Path(__file__).resolve().parent


def pick(rows: list[tuple[float, float]]) -> tuple[float, float]:
    in_window = [(s, r) for s, r in rows if ACCEPT_MIN <= r <= ACCEPT_MAX]
    pool = in_window or rows
    return min(pool, key=lambda x: abs(x[1] - 0.75))


def main() -> int:
    logs = sorted(DIR.glob("probe_step_size_log_*_*.csv"))
    if not logs:
        print("No probe_step_size_log_{cand}_{beta}.csv files.", file=sys.stderr)
        return 1

    by_key: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for path in logs:
        # probe_step_size_log_lw0011_10.0.csv
        stem = path.stem.replace("probe_step_size_log_", "")
        if "_" not in stem:
            continue
        cand, beta = stem.rsplit("_", 1)
        with path.open() as f:
            for row in csv.DictReader(f):
                try:
                    step = float(row["ft_step_size"])
                    rate = float(row["accept_rate"])
                except (KeyError, ValueError):
                    continue
                by_key.setdefault((cand, beta), []).append((step, rate))

    if not by_key:
        print("No valid probe rows.", file=sys.stderr)
        return 1

    for (cand, beta), rows in sorted(by_key.items(), key=lambda x: (x[0][0], float(x[0][1]))):
        step, rate = pick(rows)
        status = "pass" if ACCEPT_MIN <= rate <= ACCEPT_MAX else "FAIL"
        print(f"{cand} beta={beta}: ft_step_size={step}, accept={rate:.4f} [{status}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
