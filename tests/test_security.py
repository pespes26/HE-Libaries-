"""Sanity checks for the qualitative security profiles."""
from __future__ import annotations

from he_benchmark import security


def test_three_schemes_present():
    names = {p["scheme"] for p in security.security_profiles()}
    assert names == {"AES-256-GCM", "RSA-2048-OAEP", "CKKS (TenSEAL)"}


def test_only_he_computes_and_protects_in_use():
    by = {p["scheme"]: p for p in security.security_profiles()}
    # Only CKKS can compute on ciphertext / protect data in use.
    assert by["CKKS (TenSEAL)"]["can_compute_on_ciphertext"] is True
    assert by["CKKS (TenSEAL)"]["protects_in_use"] is True
    for s in ("AES-256-GCM", "RSA-2048-OAEP"):
        assert by[s]["can_compute_on_ciphertext"] is False
        assert by[s]["protects_in_use"] is False


def test_takeaways_nonempty():
    assert len(security.key_takeaways()) >= 3
