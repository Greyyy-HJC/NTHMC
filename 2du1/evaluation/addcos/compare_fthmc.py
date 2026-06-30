"""Compatibility entrypoint for the U(1) addcos FT-HMC variant."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / "base"
sys.path.insert(0, str(BASE_DIR))

from compare_fthmc import main


if __name__ == "__main__":
    main()
