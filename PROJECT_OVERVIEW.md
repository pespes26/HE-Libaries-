# Benchmarking Homomorphic Encryption — Project Overview (for review)

> This document is a self-contained description of the project for an external reviewer
> (human or LLM). It states what was built, the tech stack, the methodology, the actual
> measured results, and — honestly — what is and isn't covered.

**Team:** Bar Pesso, Shay Harush, Shon Platok
**Course:** Confidential Computing

---

## 1. Goal & research question

A reproducible benchmark measuring the real overhead of **Homomorphic Encryption (HE)** for
simple numerical analytics — runtime, memory, ciphertext size, and correctness — compared
against **AES-256** and **RSA-2048** baselines, with an explicit, honest framing of what HE
buys (computation on encrypted data) versus what it costs.

**Research question:** For basic numerical operations, what overhead does HE add in runtime,
memory, ciphertext size, and correctness — and when is that overhead acceptable?

---

## 2. Tech stack

| Layer | Choice | Version | Rationale |
|---|---|---|---|
| Language | Python | 3.11 | TenSEAL ships no 3.13 wheel; 3.11 pinned for the wheel |
| HE library | **TenSEAL** (CKKS scheme) | 0.3.16 | Easiest HE library for a Python prototype; NumPy-friendly |
| Classical crypto | **cryptography** (AES-256-GCM, RSA-2048-OAEP) | 42.0.8 | Standard, audited primitives |
| Numerics | NumPy | 1.26.4 (pinned <2.0 for TenSEAL ABI) | Plaintext reference + data generation |
| Data handling | pandas | 2.2.2 | CSV analysis + dashboard tables |
| Plotting | matplotlib (Agg, headless) | 3.8.4 | Figure generation |
| System metrics | psutil | 5.9.8 | Process RSS (TenSEAL ciphertexts live in C++ memory, invisible to `tracemalloc`) |
| Tests | pytest | 8.2.2 | Correctness, security, app smoke tests |
| UI | **Streamlit** | 1.37.1 | Interactive dashboard, live demo, use case |
| Runtime | **Docker** (`python:3.11-slim`) + docker-compose | — | Full reproducibility; `results/` and `figures/` bind-mounted to host |
| Design | Ubuntu / Ubuntu Mono fonts; injected CSS | — | Dark "design signature" + crisp motion |

---

## 3. HE scheme & parameters

- **CKKS** — approximate, real-valued; covers addition, multiplication, sum, average
  (division by a constant).
- Parameters: `poly_modulus_degree = 8192` (→ **4096 slots**),
  `coeff_mod_bit_sizes = [60, 40, 40, 60]` (200 bits ≤ ~218 limit), `global_scale = 2^40`
  → **~128-bit security**, supporting add/sum plus 1–2 multiplicative levels.
- **Chunking (handled explicitly):** datasets larger than 4096 slots are split into
  `ceil(N / 4096)` ciphertexts; `sum()` runs per-chunk via Galois rotations, then chunk
  results combine.
- Two **granularities** benchmarked: **packed** (batched, idiomatic) vs **element-wise**
  (one ciphertext per value) — the central "when is HE practical" variable.

---

## 4. Architecture

```
config.py                       # all parameters (sizes, repeats, CKKS params, seed)
he_benchmark/
  data.py                       # seeded synthetic data (uniform float64 in [0,100))
  metrics.py                    # perf_counter timing, peak-RSS sampler thread, repeat-runner
  reference.py                  # plaintext NumPy ground truth
  he_ckks.py                    # CKKS context/keys, chunked encrypt, add/mul/sum/mean, decrypt, sizes
  baseline_aes.py               # AES-256-GCM (data protection only)
  baseline_rsa.py               # RSA-2048-OAEP per-block (data protection only)
  security.py                   # qualitative security profiles + takeaways
  environment.py                # versions / CPU capture
experiments/run_benchmark.py    # orchestrates size × op × scheme; writes results CSV + run-config JSON
analysis/make_plots.py          # 8 figures + workflow_comparison.csv
app/streamlit_app.py            # 4-tab dashboard (design-styled)
tests/                          # 15 tests (correctness, security, Streamlit AppTest smoke)
Dockerfile · docker-compose.yml · .streamlit/config.toml · README.md
results/ · figures/             # generated outputs (bind-mounted)
```

---

## 5. Datasets

- **Synthetic numerical** float64 arrays, uniform in **[0, 100)**, NumPy-seeded
  (`default_rng`) so every run reproduces exactly.
- Two arrays per size: `a` (seed 42) and `b` (seed 43, the second operand for add/multiply).
- **Sizes:** 200, 1,000, 10,000, 100,000 records.
- The **same data is encrypted under all schemes** (AES, RSA, CKKS) — a controlled
  comparison, methodologically stronger than two unrelated datasets.
- Data is synthetic and non-sensitive (chosen for reproducibility; CKKS is real-valued).
- The use-case tab uses a tiny user-entered dataset (default salaries) to make the
  confidential-computing story concrete.

---

## 6. Methodology (fair-comparison rules, strictly followed)

- **HE computes on ciphertext; AES/RSA cannot** — stated everywhere, never blurred.
- Costs are separated into three buckets:
  1. **Data protection** — encrypt/decrypt (all schemes).
  2. **Computation** — HE on ciphertext vs plaintext compute (measured separately).
  3. **Size overhead** — ciphertext vs plaintext bytes.
- **CKKS is approximate** → every run reports `max_abs_error` and `mean_rel_error` vs the
  plaintext NumPy reference.
- **RSA-2048 cannot encrypt bulk data** (~190-byte OAEP limit) → measured **per single
  record** (256 bytes/record), with the hybrid-usage caveat stated. Never presented as a
  bulk figure.
- **`count` is plaintext metadata**, not an HE computation — labeled as such.
- **Reproducibility:** fixed seed; each measurement runs **5 times after a discarded
  warm-up**; mean ± std reported; full config + library versions + CPU saved to
  `results/run_config.json`.
- **Memory** measured via **process RSS (psutil)**, not `tracemalloc`, because TenSEAL
  ciphertexts live in C++ memory.

---

## 7. Operations benchmarked

Addition, multiplication (element-wise, ciphertext × ciphertext with relinearization),
sum (Galois-rotation reduction across chunks), average (sum × 1/N), and count (noted as
metadata). Each is verified against the plaintext result.

---

## 8. Actual measured results (real runs, not invented)

Hardware: the developer's machine inside Docker (`python:3.11-slim`). Numbers are means over
5 repeats. **48 result rows** total in `results/results.csv`.

**CKKS, packed, by dataset size:**

| N | encrypt | sum compute | decrypt | ciphertext size | mean rel. error (sum) |
|---|---|---|---|---|---|
| 200 | ~3.1 ms | ~30 ms | ~0.7 ms | — | ~6e-10 |
| 1,000 | ~2.8 ms | ~65 ms | ~0.7 ms | — | ~3e-10 |
| 10,000 | ~8.7 ms | ~89 ms | ~0.7 ms | — | ~4e-11 |
| 100,000 | ~74 ms | ~503 ms | ~0.7 ms | **8.36 MB** | ~2e-11 |

**Key findings:**
- **Packing dominates practicality.** Element-wise encryption is ~**200× slower** than
  packed (≈580 ms vs ≈2.8 ms to encrypt at N=200).
- **Size overhead:** at N=100,000 the CKKS encrypted dataset is **~8.36 MB vs ~0.8 MB
  plaintext (~10.5×)**; AES adds only **28 bytes** (12-byte nonce + 16-byte GCM tag).
- **Operation shape:** add/multiply are cheap (slot-wise); sum/average cost more because
  `.sum()` needs ~log₂(slots) Galois rotations.
- **Accuracy:** CKKS relative error ranges ~**1.3e-7** (multiply/mean) down to **1e-11**
  (add) — small but non-zero, as CKKS requires.
- **Baselines:** AES-256-GCM protects the whole dataset in ~0.02–0.19 ms (no compute);
  RSA-2048-OAEP ~0.02 ms encrypt / ~0.2 ms decrypt **per record**, 256 bytes/record.
- **Key/context overhead:** ~**34.6 MB** of public + Galois + relin keys, reported
  separately from per-record ciphertext size.

**Traditional-vs-HE analytics contrast** (the assignment's core comparison): computing
`sum` over encrypted data costs ~0.16 ms the AES way (decrypt → compute, **plaintext
exposed**) vs ~500 ms the HE way at N=100,000 (**never decrypted**) — HE is ~3000× slower,
which is the measured price of never exposing the data during computation.

---

## 9. Security analysis (qualitative, in `he_benchmark/security.py`)

| Scheme | Type | Security level | Computes on ciphertext? | Protects data **in use**? | Quantum |
|---|---|---|---|---|---|
| AES-256-GCM | Symmetric (AEAD) | ~256-bit (~128 vs Grover) | No | No | Resists Grover |
| RSA-2048-OAEP | Asymmetric | ~112-bit (NIST) | No | No | Broken by Shor |
| CKKS (TenSEAL) | Homomorphic (RLWE) | ~128-bit | **Yes** | **Yes** | Believed quantum-resistant |

**Core security argument:** HE uniquely protects data **in use** — the compute party never
sees plaintext. AES/RSA require decryption before any computation. Honest caveats included:
HE's benefit costs large runtime/size overhead; AES-GCM also gives integrity/authenticity
while CKKS gives confidentiality only and is approximate; CKKS has a known
approximate-decryption leakage caveat (Li–Micciancio 2021), mitigated by noise flooding.

---

## 10. Interactive app (Streamlit, 4 tabs)

1. **Dashboard** — interactive charts + filterable table from `results.csv` (runtime,
   packed vs element-wise, error, protection-vs-compute, memory), plus static figures.
2. **Live HE demo** — choose operation / size / granularity / seed → encrypt → compute on
   ciphertext → decrypt → verify vs plaintext, showing per-phase timing and ciphertext size.
3. **Confidential use case** — a salary-average scenario: values are encrypted, the server
   computes the average on ciphertext (shows the raw bytes it sees), only the aggregate is
   decrypted.
4. **Security & HE vs traditional** — the security table + the explicit "AES must decrypt,
   HE doesn't" workflow comparison.

The app reuses the exact same `he_benchmark` modules as the benchmark, so live results match
the measured pipeline. It is styled with a dark "design signature" (Ubuntu/Ubuntu Mono, pill
buttons, border-depth) and crisp, accessibility-aware CSS motion (press feedback, focus
rings, reduced-motion support).

---

## 11. Figures generated (`figures/`)

`ckks_runtime_vs_size`, `ckks_cost_breakdown`, `ciphertext_size_vs_size`,
`ckks_error_vs_size`, `protection_cost`, `packed_vs_elementwise`, `memory_vs_size`,
`workflow_comparison` (+ `results/workflow_comparison.csv`).

---

## 12. Testing & verification

- **15 tests pass** (`pytest`): CKKS correctness (packed + element-wise; sum/mean/add/mul;
  chunking above one ciphertext), AES/RSA round-trips, security-profile invariants, and
  **headless Streamlit `AppTest`** smoke tests that run the app and click the live-demo /
  use-case buttons.
- Reproducible run: `docker compose run --rm benchmark` → CSV + run-config; verified across
  all four sizes with all operations correct.
- Smoke verification caught a real version bug (`st.image(use_container_width=…)` invalid in
  Streamlit 1.37 → fixed to `use_column_width`).

---

## 13. How to run

```bash
docker compose build
docker compose run --rm benchmark pytest -v     # 15 tests
docker compose run --rm benchmark               # -> results/results.csv + run_config.json
docker compose run --rm plots                   # -> figures/*.png
docker compose up dashboard                      # http://localhost:8501
```

---

## 14. Honest coverage assessment (what's done vs not)

**Fully covered**
- HE workflow (encrypt → compute on ciphertext → decrypt) with TenSEAL CKKS.
- AES-256 + RSA-2048 baselines with fair framing.
- Operations: add, multiply, sum, average (count as metadata).
- Metrics: runtime (split into encrypt/compute/decrypt), ciphertext size, correctness +
  approximation error, scalability to 100k, peak memory for all schemes.
- Reproducibility (seed, repeats, config + environment captured), CSV output, 8 figures.
- Security analysis + explicit traditional-vs-HE comparison.
- Interactive demo connecting to a confidential-computing use case.

**Partial / limitations (stated honestly)**
- **Peak-RSS memory is noisy** for short operations (page reuse makes per-op deltas
  under-report); the **authoritative size/memory signal is `ciphertext_size_bytes`**. This
  is documented in the README.
- **Data is synthetic** uniform floats, not a named real-world dataset.
- **Element-wise granularity capped at 200 records** (it is O(N) large ciphertexts and OOMs
  the container at higher sizes) — documented; packed runs to 100k.
- **CKKS only** — BFV (exact integer sum/count) is a documented planned follow-up, not built.

**Not built yet (deliverables)**
- A **written final report** (the measurements/figures/security content all exist; the
  report document does not).
- A **final presentation deck**.

---

## 15. Suggested evaluation criteria for the reviewer

1. Is the HE vs AES/RSA comparison **fair and not misleading** (no "HE is better" claims;
   computation-on-ciphertext capability vs cost clearly separated)?
2. Is **correctness verified** against plaintext, with CKKS approximation error reported?
3. Is the work **reproducible** (seeds, repeats, pinned environment, Docker)?
4. Are **runtime, memory, ciphertext size, scalability, and security** all addressed?
5. Are **limitations stated honestly** rather than hidden?
6. Gaps to weigh: synthetic (not real) data; RSS-memory noise; CKKS-only; report/deck not
   yet written.
