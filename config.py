"""Central configuration for the HE benchmark.

Every tunable parameter lives here so experiments are reproducible: change a value,
re-run, and the run-config JSON written next to the CSV records exactly what was used.
"""
from __future__ import annotations

# --- Reproducibility ---
RANDOM_SEED = 42

# Dataset sizes (records) to benchmark. 100_000 is heavy; add it only if hardware allows.
# The small 200 size exists so the element-wise granularity (one ciphertext per record,
# O(N) memory) can run and be compared against packed without exhausting RAM.
DATASET_SIZES = [200, 1_000, 10_000, 100_000]

# Timed repetitions per measurement (one extra warm-up run is discarded).
N_REPEATS = 5

# Synthetic data range (uniform floats).
DATA_LOW = 0.0
DATA_HIGH = 100.0

# --- CKKS parameters (TenSEAL) ---
# poly_modulus_degree = 8192  ->  4096 CKKS slots per ciphertext (n / 2).
# coeff bit sizes sum = 200  <=  ~218-bit limit for 128-bit security at n=8192.
# The [60, 40, 40, 60] chain supports add/sum plus 1-2 multiplicative levels.
CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
CKKS_GLOBAL_SCALE = 2 ** 40
CKKS_SLOTS = CKKS_POLY_MODULUS_DEGREE // 2  # packing capacity per ciphertext

# Correctness tolerances for the *approximate* CKKS scheme. The committed run measures
# at most ~1.3e-3 absolute error (ct x ct multiply on values up to ~1e4) and ~1.4e-7
# relative error; these bounds sit 2-3 orders of magnitude above that, so they catch a
# real precision regression without ever flaking on normal CKKS noise.
CKKS_REL_TOL = 1e-4
CKKS_ABS_TOL = 1e-2

# --- Encryption granularities for HE ---
# "packed"      : whole dataset batched into ceil(N / slots) ciphertexts (idiomatic, fast).
# "elementwise" : one ciphertext per value (shows worst-case per-record overhead, very slow).
GRANULARITIES = ["packed", "elementwise"]

# Element-wise is O(N) large ciphertexts; cap its dataset size so the run stays within
# container memory and finishes quickly. Each CKKS ciphertext at n=8192 is ~0.5 MB, and a
# single op transiently holds several full copies (operands + result + memory probe).
ELEMENTWISE_MAX_SIZE = 200

# --- RSA baseline ---
# One float64 record = 8 bytes. RSA-2048-OAEP can encrypt at most ~190 bytes, so we
# measure the per-block (single-record) cost only -- never a bulk-dataset figure.
RSA_BLOCK_BYTES = 8

# --- Output paths ---
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
RESULTS_CSV = "results/results.csv"
RUN_CONFIG_JSON = "results/run_config.json"
