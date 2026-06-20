"""Correctness tests: decrypted HE/baseline results must match the plaintext reference.

CKKS is approximate -> assert within tolerance. AES/RSA are exact -> assert equality.
"""
from __future__ import annotations

import numpy as np
import pytest

import config
from he_benchmark import baseline_aes as aes
from he_benchmark import baseline_rsa as rsa
from he_benchmark import data as data_mod
from he_benchmark import he_ckks
from he_benchmark import reference as ref

TOL = dict(rtol=config.CKKS_REL_TOL, atol=config.CKKS_ABS_TOL)


@pytest.fixture(scope="module")
def ctx():
    return he_ckks.make_context()


@pytest.fixture
def data():
    return data_mod.generate_synthetic(50, seed=1)


@pytest.fixture
def data2():
    return data_mod.generate_synthetic(50, seed=2)


# --- CKKS (approximate) ----------------------------------------------------------

@pytest.mark.parametrize("granularity", ["packed", "elementwise"])
def test_ckks_sum(ctx, data, granularity):
    enc = (he_ckks.encrypt_packed if granularity == "packed"
           else he_ckks.encrypt_elementwise)(ctx, data)
    got = he_ckks.decrypt_scalar(he_ckks.he_sum(enc))
    assert np.isclose(got, ref.ref_sum(data), **TOL)


@pytest.mark.parametrize("granularity", ["packed", "elementwise"])
def test_ckks_mean(ctx, data, granularity):
    enc = (he_ckks.encrypt_packed if granularity == "packed"
           else he_ckks.encrypt_elementwise)(ctx, data)
    got = he_ckks.decrypt_scalar(he_ckks.he_mean(enc, len(data)))
    assert np.isclose(got, ref.ref_mean(data), **TOL)


def test_ckks_add(ctx, data, data2):
    ea = he_ckks.encrypt_packed(ctx, data)
    eb = he_ckks.encrypt_packed(ctx, data2)
    got = he_ckks.decrypt_vectors(he_ckks.he_add(ea, eb))
    assert np.allclose(got, ref.ref_add(data, data2), **TOL)


def test_ckks_mul(ctx, data, data2):
    ea = he_ckks.encrypt_packed(ctx, data)
    eb = he_ckks.encrypt_packed(ctx, data2)
    got = he_ckks.decrypt_vectors(he_ckks.he_mul(ea, eb))
    assert np.allclose(got, ref.ref_mul(data, data2), **TOL)


def test_ckks_chunking_above_one_ciphertext(ctx):
    """A dataset larger than one ciphertext's slots must still sum correctly."""
    big = data_mod.generate_synthetic(config.CKKS_SLOTS + 100, seed=7)
    enc = he_ckks.encrypt_packed(ctx, big)
    assert len(enc) >= 2  # confirms chunking actually happened
    expected = ref.ref_sum(big)
    got = he_ckks.decrypt_scalar(he_ckks.he_sum(enc))
    # Assert the *relative* error directly (the sum is ~10^5, so a relative bound is the
    # meaningful check here; a tight 1e-3 also guards against a systematic chunking bias).
    assert abs(got - expected) / abs(expected) < 1e-3


# --- AES / RSA (exact) -----------------------------------------------------------

def test_aes_roundtrip(data):
    key = aes.make_key()
    nonce, ct = aes.encrypt(key, data)
    out = aes.decrypt(key, nonce, ct, data.shape)
    assert np.array_equal(out, data)


def test_rsa_block_roundtrip(data):
    priv, pub = rsa.make_keypair()
    block = rsa.value_to_block(float(data[0]))
    out = rsa.block_to_value(rsa.decrypt_block(priv, rsa.encrypt_block(pub, block)))
    assert out == float(data[0])
