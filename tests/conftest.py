"""Test defaults that keep the regular suite portable and CPU-only."""

import os


os.environ.setdefault("JAX_PLATFORMS", "cpu")
