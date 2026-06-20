"""AES-256-GCM baseline (cryptography).

IMPORTANT FRAMING: AES provides confidentiality only. It CANNOT compute on
ciphertext. To analyze AES-protected data you must decrypt first, compute in
plaintext, then (optionally) re-encrypt. This module measures only the
data-protection cost (encrypt + decrypt) and the size overhead.
"""
from __future__ import annotations

import os

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def make_key() -> bytes:
    """Generate a fresh 256-bit AES key."""
    return AESGCM.generate_key(bit_length=256)


def encrypt(key: bytes, data: np.ndarray):
    """Encrypt the whole dataset (serialized to raw bytes). Returns (nonce, ct)."""
    aes = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce recommended for GCM
    ct = aes.encrypt(nonce, data.tobytes(), None)
    return nonce, ct


def decrypt(key: bytes, nonce: bytes, ct: bytes, shape, dtype=np.float64) -> np.ndarray:
    """Decrypt and restore the original array shape/dtype."""
    aes = AESGCM(key)
    pt = aes.decrypt(nonce, ct, None)
    return np.frombuffer(pt, dtype=dtype).reshape(shape)


def ciphertext_size_bytes(nonce: bytes, ct: bytes) -> int:
    """Total stored size = nonce (12B) + GCM ciphertext (data + 16B auth tag)."""
    return len(nonce) + len(ct)
