"""Synthetic numerical dataset generation (seeded for reproducibility)."""
from __future__ import annotations

import numpy as np


def generate_synthetic(size: int, seed: int,
                       low: float = 0.0, high: float = 100.0) -> np.ndarray:
    """Return a 1-D float64 array of `size` uniform values in [low, high).

    Seeded via numpy's Generator so the same (size, seed) always yields identical
    data -- a prerequisite for repeatable benchmarking.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=size).astype(np.float64)


def generate_synthetic_int(size: int, seed: int,
                           low: int = 0, high: int = 1000) -> np.ndarray:
    """Integer variant reserved for future exact-scheme (BFV) work.

    Not used by the milestone-1 CKKS operations, which are real-valued.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=size).astype(np.int64)
