"""CKKS homomorphic-encryption workflow (TenSEAL).

CKKS is an *approximate* scheme over real numbers: decrypted results carry small
numerical error, which the benchmark measures on every run. With
poly_modulus_degree=8192 there are 4096 usable slots, so a dataset larger than that
is split across multiple ciphertexts ("chunks"). All operations work on a uniform
representation: a list of CKKSVector objects.
"""
from __future__ import annotations

from typing import List

import numpy as np
import tenseal as ts

import config


def make_context() -> "ts.Context":
    """Create a CKKS context with a secret key plus relin and Galois keys.

    - relin keys  : required for ciphertext x ciphertext multiplication.
    - Galois keys : required for vector.sum() (it rotates slots to add them).
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=config.CKKS_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=config.CKKS_COEFF_MOD_BIT_SIZES,
    )
    ctx.global_scale = config.CKKS_GLOBAL_SCALE
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


# --- Encryption (two granularities) ---------------------------------------------

def encrypt_packed(ctx, data: np.ndarray) -> List:
    """Batch the dataset into ceil(N / slots) CKKS vectors (idiomatic, fast)."""
    slots = config.CKKS_SLOTS
    chunks = []
    for i in range(0, len(data), slots):
        chunk = data[i:i + slots]
        chunks.append(ts.ckks_vector(ctx, chunk.tolist()))
    return chunks


def encrypt_elementwise(ctx, data: np.ndarray) -> List:
    """Encrypt one value per ciphertext (worst-case per-record overhead)."""
    return [ts.ckks_vector(ctx, [float(x)]) for x in data]


# --- Homomorphic operations (operate on the list-of-vectors representation) ------

def he_add(a: List, b: List) -> List:
    """Element-wise add of two encrypted datasets (same chunk structure)."""
    return [x + y for x, y in zip(a, b)]


def he_mul(a: List, b: List) -> List:
    """Element-wise multiply (ciphertext x ciphertext, uses relin keys)."""
    return [x * y for x, y in zip(a, b)]


def he_sum(a: List):
    """Sum every element across all chunks -> a single length-1 CKKSVector."""
    partial = None
    for v in a:
        s = v.sum()  # in-vector sum via Galois rotations -> length-1 vector
        partial = s if partial is None else partial + s
    return partial


def he_mean(a: List, n: int):
    """Average = sum * (1/n), with 1/n applied as a plaintext scalar."""
    return he_sum(a) * (1.0 / n)


def he_dot(a: List, b: List):
    """Dot product / weighted sum of two encrypted vectors -> length-1 CKKSVector.

    Element-wise ciphertext x ciphertext multiply (one multiplicative level, depth 1 --
    well within the [60,40,40,60] chain's measured 2-level budget), then sum across all
    slots/chunks. Powers the encrypted-weighting/scoring use case.
    """
    return he_sum(he_mul(a, b))


# --- Decryption ------------------------------------------------------------------

def decrypt_vectors(vectors: List) -> np.ndarray:
    """Decrypt a list of CKKS vectors back into one flat float64 array."""
    out: list[float] = []
    for v in vectors:
        out.extend(v.decrypt())
    return np.asarray(out, dtype=np.float64)


def decrypt_scalar(vec) -> float:
    """Decrypt a length-1 result vector (sum/mean) to a Python float."""
    return float(vec.decrypt()[0])


# --- Size reporting --------------------------------------------------------------

def ciphertext_size_bytes(vectors: List) -> int:
    """Total serialized size of the encrypted dataset (per-record data overhead)."""
    return sum(len(v.serialize()) for v in vectors)


def context_size_bytes(ctx) -> int:
    """Serialized public context (public + Galois + relin keys, no secret key).

    Reported separately from ciphertext size so the one-time key cost is not
    conflated with the per-record data overhead.
    """
    return len(ctx.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=True,
        save_relin_keys=True,
    ))
