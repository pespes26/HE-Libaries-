"""RSA-2048-OAEP baseline (cryptography).

IMPORTANT FRAMING: RSA is an asymmetric cipher for small payloads or key-wrapping.
With OAEP-SHA256 padding, RSA-2048 can encrypt at most ~190 bytes -- it CANNOT
encrypt a bulk dataset. Real systems use *hybrid* encryption (RSA wraps a random
AES key; AES encrypts the data). This module measures only the per-block
(single-record, 8-byte) encrypt/decrypt cost and reports it as such -- never as a
bulk-dataset figure. Like AES, RSA cannot compute on ciphertext.
"""
from __future__ import annotations

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

_PAD = padding.OAEP(
    mgf=padding.MGF1(algorithm=hashes.SHA256()),
    algorithm=hashes.SHA256(),
    label=None,
)


def make_keypair():
    """Generate a 2048-bit RSA keypair. Returns (private_key, public_key)."""
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return priv, priv.public_key()


def encrypt_block(pub, block: bytes) -> bytes:
    """Encrypt one small block (<=190 bytes). Returns a 256-byte ciphertext."""
    return pub.encrypt(block, _PAD)


def decrypt_block(priv, ct: bytes) -> bytes:
    return priv.decrypt(ct, _PAD)


def value_to_block(x: float) -> bytes:
    """Serialize a single float64 record to its 8 raw bytes."""
    return np.float64(x).tobytes()


def block_to_value(b: bytes) -> float:
    return float(np.frombuffer(b, dtype=np.float64)[0])
