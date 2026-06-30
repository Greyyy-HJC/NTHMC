#!/usr/bin/env python3
"""Concatenate parallel FT-HMC topology chunk dumps into one CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge chunked topo_fthmc CSV dumps")
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument(
        "--base-name",
        required=True,
        help="Filename stem without chunk suffix, e.g. topo_fthmc_L16_beta10.0_nsteps10_TAG",
    )
    parser.add_argument("--n-chunks", type=int, required=True)
    parser.add_argument("--chunk-prefix", default="c")
    args = parser.parse_args()

    arrays = []
    for i in range(args.n_chunks):
        path = args.dump_dir / f"{args.base_name}_{args.chunk_prefix}{i}.csv"
        if not path.exists():
            print(f"Missing chunk: {path}", file=sys.stderr)
            return 1
        arrays.append(np.atleast_1d(np.loadtxt(path)))

    merged = np.concatenate(arrays)
    out = args.dump_dir / f"{args.base_name}.csv"
    np.savetxt(out, merged, fmt="%.6e")
    print(f"Merged {args.n_chunks} chunks -> {out} ({len(merged)} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
