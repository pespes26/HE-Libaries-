"""Benchmark orchestrator.

For every (dataset_size x operation x scheme) combination it:
  1. generates seeded synthetic data and computes the plaintext reference result,
  2. times encryption / computation / decryption separately (mean +/- std over repeats),
  3. measures peak process memory and serialized ciphertext size,
  4. verifies the decrypted result against the plaintext reference,
and writes one CSV row per combination plus a run-config JSON for reproducibility.

Schemes covered:
  - plaintext (NumPy)  : the computation-cost reference (no encryption).
  - CKKS (TenSEAL)     : computes ON ciphertext; packed and element-wise granularities.
  - AES-256-GCM        : data-protection cost only (cannot compute on ciphertext).
  - RSA-2048-OAEP      : per-block data-protection cost only (cannot compute; bulk N/A).
"""
from __future__ import annotations

import csv
import gc
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

# Make the project root importable when run directly as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from he_benchmark import baseline_aes as aes  # noqa: E402
from he_benchmark import baseline_rsa as rsa  # noqa: E402
from he_benchmark import data as data_mod  # noqa: E402
from he_benchmark import environment  # noqa: E402
from he_benchmark import he_ckks  # noqa: E402
from he_benchmark import metrics  # noqa: E402
from he_benchmark import reference as ref  # noqa: E402

CSV_COLUMNS = [
    "run_id", "timestamp", "scheme", "library", "operation", "dataset_size",
    "granularity", "n_repeats",
    "encrypt_time_mean", "encrypt_time_std",
    "compute_time_mean", "compute_time_std",
    "decrypt_time_mean", "decrypt_time_std",
    "total_time_mean", "peak_memory_mb",
    "ciphertext_size_bytes", "context_size_bytes",
    "plaintext_result", "decrypted_result",
    "max_abs_error", "mean_rel_error", "correct",
    "poly_modulus_degree", "scale", "notes",
]

# HE/plaintext operation specs. two=needs a second operand; scalar=returns one value.
HE_OPS = [
    {"name": "add", "two": True, "scalar": False},
    {"name": "mul", "two": True, "scalar": False},
    {"name": "sum", "two": False, "scalar": True},
    {"name": "mean", "two": False, "scalar": True},
]


# --- small helpers ---------------------------------------------------------------

def blank_row() -> dict:
    return {c: None for c in CSV_COLUMNS}


def errors(ref_val, got_val):
    """Return (max_abs_error, mean_rel_error) over the full result (vector or scalar)."""
    r = np.atleast_1d(np.asarray(ref_val, dtype=np.float64))
    g = np.atleast_1d(np.asarray(got_val, dtype=np.float64))
    abs_err = np.abs(g - r)
    denom = np.maximum(np.abs(r), 1e-12)
    return float(abs_err.max()), float(np.mean(abs_err / denom))


def ref_result(op: str, a: np.ndarray, b: np.ndarray):
    return {
        "add": lambda: ref.ref_add(a, b),
        "mul": lambda: ref.ref_mul(a, b),
        "sum": lambda: ref.ref_sum(a),
        "mean": lambda: ref.ref_mean(a),
    }[op]()


def make_compute(op: str, enc_a, enc_b, n: int):
    return {
        "add": lambda: he_ckks.he_add(enc_a, enc_b),
        "mul": lambda: he_ckks.he_mul(enc_a, enc_b),
        "sum": lambda: he_ckks.he_sum(enc_a),
        "mean": lambda: he_ckks.he_mean(enc_a, n),
    }[op]


def encrypt_fn(ctx, arr, granularity):
    if granularity == "packed":
        return he_ckks.encrypt_packed(ctx, arr)
    return he_ckks.encrypt_elementwise(ctx, arr)


def decrypt_result(result, scalar: bool):
    return he_ckks.decrypt_scalar(result) if scalar else he_ckks.decrypt_vectors(result)


# --- per-scheme benchmarking -----------------------------------------------------

def bench_plaintext(rows, next_id, ts_now, a, b, size, n):
    """Plaintext compute-cost reference (no encryption)."""
    for op in HE_OPS + [{"name": "count", "two": False, "scalar": True}]:
        name = op["name"]
        fn = (lambda nm=name: ref.ref_count(a)) if name == "count" else \
             (lambda nm=name: ref_result(nm, a, b))
        with metrics.MemorySampler() as ms:
            _ = fn()
        t, _ = metrics.repeat(fn, n)
        row = blank_row()
        row.update(
            run_id=next_id(), timestamp=ts_now, scheme="plaintext", library="numpy",
            operation=name, dataset_size=size, granularity="plaintext", n_repeats=n,
            encrypt_time_mean=0.0, encrypt_time_std=0.0,
            compute_time_mean=t.mean, compute_time_std=t.std,
            decrypt_time_mean=0.0, decrypt_time_std=0.0,
            total_time_mean=t.mean, peak_memory_mb=ms.delta_mb,
            ciphertext_size_bytes=int(a.nbytes),
            plaintext_result=None, decrypted_result=None,
            max_abs_error=0.0, mean_rel_error=0.0, correct=True,
            notes="plaintext reference (compute cost only)",
        )
        rows.append(row)


def bench_ckks(rows, next_id, ts_now, ctx, ctx_size, a, b, size, granularity, n):
    """CKKS HE path for all ops at one (size, granularity)."""
    for spec in HE_OPS:
        op, two, scalar = spec["name"], spec["two"], spec["scalar"]

        # Peak memory over a single full workflow (encrypt -> compute -> decrypt).
        with metrics.MemorySampler() as ms:
            m_ea = encrypt_fn(ctx, a, granularity)
            m_eb = encrypt_fn(ctx, b, granularity) if two else None
            m_res = make_compute(op, m_ea, m_eb, size)()
            _ = decrypt_result(m_res, scalar)
        peak_mb = ms.delta_mb
        # Free the memory-probe ciphertexts before the timed phase so element-wise
        # (O(N) large ciphertexts) does not hold two full copies at once.
        del m_ea, m_eb, m_res
        gc.collect()

        # Timed phases. Encrypt 'a' is the representative per-dataset encryption cost;
        # the second operand is encrypted once outside timing for two-operand ops.
        enc_t, enc_a = metrics.repeat(lambda: encrypt_fn(ctx, a, granularity), n)
        enc_b = encrypt_fn(ctx, b, granularity) if two else None
        compute = make_compute(op, enc_a, enc_b, size)
        comp_t, result = metrics.repeat(compute, n)
        dec_t, decrypted = metrics.repeat(lambda: decrypt_result(result, scalar), n)

        ct_size = he_ckks.ciphertext_size_bytes(enc_a)
        r = ref_result(op, a, b)
        max_abs, mean_rel = errors(r, decrypted)
        correct = bool(np.allclose(np.atleast_1d(decrypted), np.atleast_1d(r),
                                   rtol=config.CKKS_REL_TOL, atol=config.CKKS_ABS_TOL))

        if scalar:
            pres, dres = float(r), float(decrypted)
            note = ""
        else:
            pres, dres = float(np.atleast_1d(r)[0]), float(np.atleast_1d(decrypted)[0])
            note = "vector op: result[0] shown; errors computed over full vector"

        n_ct = len(enc_a)
        note = (note + f"; {n_ct} ciphertext(s), {config.CKKS_SLOTS} slots each").strip("; ")

        row = blank_row()
        row.update(
            run_id=next_id(), timestamp=ts_now, scheme="CKKS", library="TenSEAL",
            operation=op, dataset_size=size, granularity=granularity, n_repeats=n,
            encrypt_time_mean=enc_t.mean, encrypt_time_std=enc_t.std,
            compute_time_mean=comp_t.mean, compute_time_std=comp_t.std,
            decrypt_time_mean=dec_t.mean, decrypt_time_std=dec_t.std,
            total_time_mean=enc_t.mean + comp_t.mean + dec_t.mean,
            peak_memory_mb=peak_mb,
            ciphertext_size_bytes=ct_size, context_size_bytes=ctx_size,
            plaintext_result=pres, decrypted_result=dres,
            max_abs_error=max_abs, mean_rel_error=mean_rel, correct=correct,
            poly_modulus_degree=config.CKKS_POLY_MODULUS_DEGREE,
            scale=float(config.CKKS_GLOBAL_SCALE), notes=note,
        )
        rows.append(row)
        print(f"    CKKS/{granularity:<11} {op:<5} "
              f"enc={enc_t.mean*1e3:8.2f}ms comp={comp_t.mean*1e3:8.2f}ms "
              f"dec={dec_t.mean*1e3:8.2f}ms relerr={mean_rel:.2e} ok={correct}")
        # Release this op's ciphertexts before the next op iteration.
        del enc_a, enc_b, result, decrypted
        gc.collect()


def bench_aes(rows, next_id, ts_now, a, size, n):
    """AES-256-GCM data-protection cost (encrypt + decrypt of the whole dataset)."""
    key = aes.make_key()
    with metrics.MemorySampler() as ms:
        _n, _c = aes.encrypt(key, a)
        _ = aes.decrypt(key, _n, _c, a.shape)
    enc_t, (nonce, ct) = metrics.repeat(lambda: aes.encrypt(key, a), n)
    dec_t, decrypted = metrics.repeat(lambda: aes.decrypt(key, nonce, ct, a.shape), n)
    roundtrip = bool(np.array_equal(decrypted, a))  # reuse the timed-phase result
    row = blank_row()
    row.update(
        run_id=next_id(), timestamp=ts_now, scheme="AES-256-GCM", library="cryptography",
        operation="protect", dataset_size=size, granularity="bulk", n_repeats=n,
        encrypt_time_mean=enc_t.mean, encrypt_time_std=enc_t.std,
        compute_time_mean=None, compute_time_std=None,
        decrypt_time_mean=dec_t.mean, decrypt_time_std=dec_t.std,
        total_time_mean=enc_t.mean + dec_t.mean, peak_memory_mb=ms.delta_mb,
        ciphertext_size_bytes=aes.ciphertext_size_bytes(nonce, ct),
        max_abs_error=0.0, mean_rel_error=0.0, correct=roundtrip,
        notes="confidentiality only; cannot compute on ciphertext",
    )
    rows.append(row)
    print(f"    AES         protect enc={enc_t.mean*1e3:8.2f}ms dec={dec_t.mean*1e3:8.2f}ms "
          f"size={row['ciphertext_size_bytes']}B roundtrip={roundtrip}")


def bench_rsa(rows, next_id, ts_now, priv, pub, a, size, n):
    """RSA-2048-OAEP per-block (single-record) data-protection cost."""
    block = rsa.value_to_block(float(a[0]))
    with metrics.MemorySampler() as ms:
        _cb = rsa.encrypt_block(pub, block)
        _ = rsa.decrypt_block(priv, _cb)
    enc_t, ctb = metrics.repeat(lambda: rsa.encrypt_block(pub, block), n)
    dec_t, decrypted_block = metrics.repeat(lambda: rsa.decrypt_block(priv, ctb), n)
    roundtrip = bool(rsa.block_to_value(decrypted_block) == float(a[0]))  # reuse timed result
    row = blank_row()
    row.update(
        run_id=next_id(), timestamp=ts_now, scheme="RSA-2048-OAEP", library="cryptography",
        operation="protect_perblock", dataset_size=size, granularity="per_record", n_repeats=n,
        encrypt_time_mean=enc_t.mean, encrypt_time_std=enc_t.std,
        compute_time_mean=None, compute_time_std=None,
        decrypt_time_mean=dec_t.mean, decrypt_time_std=dec_t.std,
        total_time_mean=enc_t.mean + dec_t.mean, peak_memory_mb=ms.delta_mb,
        ciphertext_size_bytes=len(ctb),
        max_abs_error=0.0, mean_rel_error=0.0, correct=roundtrip,
        notes=("per-8-byte-block cost; RSA-2048 max payload ~190B so bulk data uses "
               "hybrid (RSA wraps an AES key); cannot compute on ciphertext"),
    )
    rows.append(row)
    print(f"    RSA         block   enc={enc_t.mean*1e3:8.2f}ms dec={dec_t.mean*1e3:8.2f}ms "
          f"size={len(ctb)}B/record roundtrip={roundtrip}")


# --- main ------------------------------------------------------------------------

def main() -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    n = config.N_REPEATS
    ts_now = datetime.now(timezone.utc).isoformat()

    _counter = {"i": 0}
    def next_id():
        _counter["i"] += 1
        return _counter["i"]

    rows: list[dict] = []

    # One-time setup shared across sizes.
    print("Building CKKS context + keys ...")
    ctx = he_ckks.make_context()
    ctx_size = he_ckks.context_size_bytes(ctx)
    print(f"  context (public+keys) size = {ctx_size/1024:.1f} KB")
    print("Generating RSA-2048 keypair ...")
    priv, pub = rsa.make_keypair()

    for size in config.DATASET_SIZES:
        print(f"\n=== dataset_size = {size} ===")
        a = data_mod.generate_synthetic(size, config.RANDOM_SEED,
                                        config.DATA_LOW, config.DATA_HIGH)
        b = data_mod.generate_synthetic(size, config.RANDOM_SEED + 1,
                                        config.DATA_LOW, config.DATA_HIGH)

        bench_plaintext(rows, next_id, ts_now, a, b, size, n)
        bench_aes(rows, next_id, ts_now, a, size, n)
        bench_rsa(rows, next_id, ts_now, priv, pub, a, size, n)

        for gran in config.GRANULARITIES:
            if gran == "elementwise" and size > config.ELEMENTWISE_MAX_SIZE:
                print(f"    (skipping element-wise at size {size} > "
                      f"{config.ELEMENTWISE_MAX_SIZE} cap)")
                continue
            bench_ckks(rows, next_id, ts_now, ctx, ctx_size, a, b, size, gran, n)

    # Write CSV.
    with open(config.RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Write run-config / environment JSON for reproducibility.
    run_config = {
        "timestamp": ts_now,
        "seed": config.RANDOM_SEED,
        "dataset_sizes": config.DATASET_SIZES,
        "n_repeats": n,
        "granularities": config.GRANULARITIES,
        "elementwise_max_size": config.ELEMENTWISE_MAX_SIZE,
        "ckks": {
            "poly_modulus_degree": config.CKKS_POLY_MODULUS_DEGREE,
            "coeff_mod_bit_sizes": config.CKKS_COEFF_MOD_BIT_SIZES,
            "global_scale": float(config.CKKS_GLOBAL_SCALE),
            "slots": config.CKKS_SLOTS,
        },
        "ckks_context_size_bytes": ctx_size,
        "environment": environment.capture(),
    }
    with open(config.RUN_CONFIG_JSON, "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nWrote {len(rows)} rows -> {config.RESULTS_CSV}")
    print(f"Wrote run config       -> {config.RUN_CONFIG_JSON}")


if __name__ == "__main__":
    main()
