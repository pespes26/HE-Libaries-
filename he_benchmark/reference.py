"""Plaintext reference computations (NumPy) -- the correctness ground truth.

Every HE/baseline result is compared against these exact plaintext results.
"""
from __future__ import annotations

import numpy as np


def ref_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def ref_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def ref_dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product / weighted sum -- the ground truth for he_dot."""
    return float(np.dot(a, b))


def ref_sum(a: np.ndarray) -> float:
    return float(np.sum(a))


def ref_mean(a: np.ndarray) -> float:
    return float(np.mean(a))


def ref_count(a: np.ndarray) -> int:
    return int(a.size)
