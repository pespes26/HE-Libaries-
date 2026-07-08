# Benchmarking Homomorphic Encryption

A reproducible benchmark measuring the practical overhead of **Homomorphic Encryption
(HE)** for simple numerical analytics — runtime, memory, ciphertext size, and
correctness — compared against **AES-256** and **RSA-2048** baselines.

**Team:** Bar Pesso, Shay Harush, Shon Platok
**Course:** Confidential Computing

## Research question

For basic numerical operations, what overhead does HE add in runtime, memory, ciphertext
size, and correctness — and when is that overhead still acceptable?

## How the comparison is framed (read this first)

HE and AES/RSA solve **different** problems, so the benchmark never claims one is simply
"better" than the other:

| Scheme | What it does | Can compute on ciphertext? |
| --- | --- | --- |
| **CKKS** (TenSEAL) | Encrypt → compute on ciphertext → decrypt result | **Yes** (approximate, real numbers) |
| **AES-256-GCM** | Confidentiality of data at rest / in transit | No |
| **RSA-2048-OAEP** | Small-payload / key-wrapping confidentiality | No |

Costs are reported in three separated buckets so the comparison is honest:
1. **Data-protection cost** — encryption / decryption (all schemes).
2. **Computation cost** — HE computes directly on ciphertext; AES/RSA must decrypt →
   compute in plaintext → re-encrypt (plaintext compute is measured on its own).
3. **Size overhead** — encrypted bytes vs raw bytes.

Important caveats baked into the results:
- **CKKS is approximate** — every run reports `max_abs_error` and `mean_rel_error` vs the
  plaintext reference.
- **RSA-2048 cannot encrypt bulk data** (OAEP max payload ~190 bytes). It is measured
  **per single record** only; real systems use hybrid encryption (RSA wraps an AES key).
  The RSA rows repeat that one per-record measurement at each dataset size for context —
  the cost does not scale with `dataset_size`.
- **`count` is not a homomorphic computation** — on this layout it is plaintext metadata
  (= N); it is recorded as such, not as an HE win.
- **`total_time_mean` counts one dataset encryption.** For the two-operand ops
  (add, multiply) the second operand is encrypted once outside the timed phase, so an
  end-to-end two-operand workflow would add roughly one more encryption.

## What it measures

- **Operations:** addition, multiplication, sum, average (`count` noted as metadata).
- **Schemes:** CKKS (TenSEAL), AES-256-GCM, RSA-2048-OAEP, plaintext (NumPy) reference.
- **CKKS granularity:** `packed` (dataset batched into ciphertexts) vs `elementwise`
  (one ciphertext per value) — the core "when is HE practical" variable.
- **Metrics:** wall-clock time (encrypt / compute / decrypt, mean ± std over repeats),
  peak process memory (RSS), ciphertext size, key/context size, and CKKS approximation
  error.
- **Depth limit (measured):** `experiments/depth_sweep.py` squares a ciphertext
  repeatedly and records the error per multiplicative depth — with the
  `[60, 40, 40, 60]` chain the error grows from ~1.3e-7 (depth 1) to ~9.4e-7 (depth 2),
  and depth 3 fails outright ("scale out of bounds": modulus chain exhausted). See
  `results/depth_sweep.json` and `figures/ckks_error_vs_depth.png`.

## Requirements & reproducibility notes

- Runs entirely in **Docker** (`python:3.11-slim`). TenSEAL 0.3.16 has no Python 3.13
  wheel, so the image pins **Python 3.11**; `numpy` is pinned `<2.0` to match TenSEAL's ABI.
- All randomness is seeded (`config.RANDOM_SEED`). Each measurement runs `N_REPEATS`
  times after one discarded warm-up; means and standard deviations are reported.
- **Memory is measured via process RSS (psutil), not `tracemalloc`** — TenSEAL
  ciphertexts live in C++ memory that `tracemalloc` cannot see.
  - *Caveat:* the per-operation **peak-RSS delta is noisy** for short operations — once
    pages are resident, later ops reuse them and the delta reads near zero. Treat
    `peak_memory_mb` as indicative only (no RSS figure is generated). The **authoritative
    size/memory-overhead metric is `ciphertext_size_bytes`** (deterministic): e.g. at
    N=100,000 the CKKS encrypted dataset is ~8.36 MB vs 0.8 MB plaintext (~10.4×), while
    AES adds only 28 bytes.
- Sub-millisecond calls (AES, RSA, plaintext NumPy ops) sit at the timer's noise floor,
  so the harness times a **batch of calls per sample** and reports the per-call mean
  (`metrics.repeat`); HE operations are milliseconds-to-seconds and are timed singly.
- The exact configuration + library versions + CPU model are written to
  `results/run_config.json`.

## Setup & run (Docker)

```bash
# Build the image
docker compose build

# Run the correctness tests
docker compose run --rm benchmark pytest -v

# Run the full benchmark -> results/results.csv + results/run_config.json
docker compose run --rm benchmark

# Generate figures -> figures/*.png
docker compose run --rm plots

# Measure CKKS error vs multiplicative depth -> results/depth_sweep.json
docker compose run --rm benchmark python experiments/depth_sweep.py

# Launch the interactive Streamlit app at http://localhost:8501
docker compose up dashboard
```

## Interactive app (Streamlit)

`app/streamlit_app.py` provides a browser UI (`docker compose up dashboard`, then open
http://localhost:8501) with four tabs:

1. **Dashboard** — interactive charts and a filterable table built from `results.csv`
   (runtime vs size, packed vs element-wise, approximation error, protection vs compute
   cost), plus the static figures.
2. **Live HE demo** — choose an operation, dataset size, and granularity; the app
   encrypts, computes on the ciphertext, decrypts only the result, and verifies it
   against plaintext, showing per-phase timing and ciphertext size.
3. **Confidential use case** — a salary-average scenario showing the server computing an
   average on encrypted values it cannot read — the capability AES/RSA lack.
4. **Security & HE vs traditional** — a security comparison of the schemes (security
   levels, what each protects, integrity, quantum resistance) plus the explicit
   traditional-vs-HE analytics contrast: AES must decrypt before computing (plaintext
   exposed); HE computes on ciphertext (never exposed).

The app imports the same `he_benchmark` modules as the benchmark, so the live results
match the measured pipeline.

`results/` and `figures/` are bind-mounted, so outputs appear on the host after the
container exits.

## Configuration

All knobs live in [`config.py`](config.py): dataset sizes, repeat count, CKKS parameters
(`poly_modulus_degree`, coefficient modulus chain, scale, tolerances), granularities, and
the element-wise size cap. Defaults: sizes `[200, 1_000, 10_000, 100_000]`, `N_REPEATS=5`
(+1 discarded warm-up), CKKS `poly_modulus_degree=8192` (4096 slots, ~128-bit security),
element-wise capped at **200** records (it is O(N) large ciphertexts — ~0.33 MB per
record — and exhausts container memory above that; the packed path runs to 100,000,
chunked automatically into ⌈N/4096⌉ ciphertexts).

## Project layout

```
config.py                 # all tunable parameters
he_benchmark/
  data.py                 # seeded synthetic data
  metrics.py              # timing (auto-batched for sub-ms ops), peak-RSS sampler
  reference.py            # plaintext NumPy ground truth
  he_ckks.py              # CKKS context, chunked encrypt, ops, decrypt, sizes
  baseline_aes.py         # AES-256-GCM (protection cost only)
  baseline_rsa.py         # RSA-2048-OAEP per-block (protection cost only)
  security.py             # qualitative security profiles + key takeaways
  environment.py          # version / CPU-model / RAM capture
experiments/run_benchmark.py   # orchestrates everything -> results/results.csv
experiments/depth_sweep.py     # CKKS error vs multiplicative depth -> depth_sweep.json
analysis/make_plots.py         # CSV + depth JSON -> figures/*.png
app/streamlit_app.py           # interactive dashboard + live HE demo + use case
tests/                         # correctness, metrics-layer, security, app smoke (21 tests)
report/FINAL_REPORT.md         # the written report (build_docx.js renders the .docx)
```

## Output

- `results/results.csv` — one row per `scheme × operation × dataset_size (× granularity)`
  with all timing, memory, size, and error columns.
- `results/run_config.json` — full configuration + environment for reproducibility.
- `results/depth_sweep.json` — CKKS error per multiplicative depth (+ the failure point).
- `figures/*.png` — runtime (±std), cost breakdown, ciphertext size, CKKS error vs size,
  CKKS error vs depth, protection-cost, packed-vs-elementwise, and workflow-comparison
  plots.

## Status

Done: end-to-end CKKS benchmark (200 → 100,000 records) + AES/RSA baselines, depth-sweep
experiment, figures, interactive dashboard, and the written report
(`report/FINAL_REPORT.md` / `.docx`). Planned next: BFV (exact integer sum/count), a real
public dataset, and the final presentation deck.
