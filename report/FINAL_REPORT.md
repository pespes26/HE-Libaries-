# Benchmarking Homomorphic Encryption

**A reproducible study of the overhead of computing on encrypted data**

Bar Pesso · Shay Harush · Shon Platok
Confidential Computing

---

## Abstract

Homomorphic Encryption (HE) allows computation directly on encrypted data, a capability
that conventional ciphers such as AES and RSA do not provide. This project builds a
reproducible benchmark that measures the practical overhead of HE for simple numerical
analytics — runtime, ciphertext size, and correctness — against AES-256 and RSA-2048
baselines, and frames the comparison honestly. The central result, measured on a dataset
of 100,000 records, is one line:

> **Computing a sum the traditional way costs ~0.16 ms but exposes the plaintext to the
> compute party. Computing the same sum homomorphically costs ~500 ms and never exposes
> the data.**

HE is therefore roughly three orders of magnitude slower for this task, carries a
~10× ciphertext-size overhead, and requires a fixed ~34 MB of keys before any computation
begins — yet it is the only one of the three schemes that keeps data encrypted *during*
processing. The benchmark quantifies exactly when that trade is worth making.

---

## 1. Introduction and research question

Confidential computing addresses a gap left by conventional cryptography: AES and RSA
protect data **at rest** and **in transit**, but to *compute* on that data it must first be
decrypted, exposing it to whoever runs the computation. Homomorphic Encryption removes that
exposure by computing on ciphertext directly.

**Research question.** For basic numerical operations, what overhead does HE add in
runtime, memory, ciphertext size, and correctness — and when is that overhead acceptable?

We answer it empirically with a reproducible prototype that runs a fixed set of operations
(addition, multiplication, sum, average) under HE and under classical baselines, verifies
every HE result against a plaintext reference, and records the cost.

---

## 2. The fundamental asymmetry (why this comparison is "unfair" by design)

The most important framing of this project is that **comparing HE to AES/RSA is inherently
unfair, because they solve different problems.** Stating runtime numbers without this frame
would be misleading.

| Property | AES-256 / RSA-2048 | CKKS (HE) |
| --- | --- | --- |
| Protects data **at rest** | Yes | Yes |
| Protects data **in transit** | Yes | Yes |
| Protects data **in use** (during computation) | **No** | **Yes** |
| Can compute on ciphertext | **No** | **Yes** |

To analyze AES- or RSA-protected data you must **decrypt it first**, perform the computation
on plaintext, then optionally re-encrypt. During that computation the data is fully exposed
to the compute party. HE computes on the ciphertext and only the final result is decrypted,
by the key holder.

Consequently this report does **not** claim HE is "better" than AES or RSA. It measures the
**cost of a capability that AES and RSA simply do not have.** Every result below should be
read as the price of keeping data encrypted while it is being used.

---

## 3. Methodology

### 3.1 Schemes and tools

| Layer | Choice | Version |
| --- | --- | --- |
| HE scheme / library | CKKS via **TenSEAL** | 0.3.16 |
| Classical baselines | AES-256-GCM, RSA-2048-OAEP via **cryptography** | 42.0.8 |
| Numerics / reference | NumPy | 1.26.4 |
| Data / analysis | pandas | 2.2.2 |
| Plotting | matplotlib | 3.8.4 |
| Tests | pytest | 8.2.2 |
| Runtime | **Docker** (`python:3.11-slim`) | — |

CKKS was chosen because it operates on real-valued vectors and supports division by a
constant, which covers all four target operations including the average. CKKS is an
**approximate** scheme, so every decrypted result is compared against an exact plaintext
reference and its error is reported.

### 3.2 CKKS parameters

`poly_modulus_degree = 8192` (4096 usable slots per ciphertext),
`coeff_mod_bit_sizes = [60, 40, 40, 60]`, `global_scale = 2^40`. The coefficient-modulus
chain totals 200 bits, within the limit for **~128-bit security** at this ring dimension.
Datasets larger than 4096 values are split into `ceil(N / 4096)` ciphertexts ("chunking");
`sum` is computed per chunk via Galois-key rotations and then combined.

### 3.3 Datasets

Synthetic numerical data: 1-D arrays of `float64` values drawn uniformly from `[0, 100)`,
generated with a fixed NumPy seed so every run is reproducible. The **same data is encrypted
under all schemes** (a controlled comparison, stronger than two unrelated datasets). Sizes
benchmarked: **200, 1,000, 10,000, and 100,000 records.** The data is synthetic and
non-sensitive, chosen for reproducibility and because CKKS is real-valued.

### 3.4 Fair-comparison rules and measurement

Costs are separated into three buckets so no scheme is judged unfairly:
1. **Data protection** — encryption / decryption (all schemes).
2. **Computation** — HE on ciphertext vs. plaintext computation (measured separately).
3. **Size overhead** — ciphertext vs. plaintext bytes.

Each measurement runs **five times after a discarded warm-up**; we report the mean (the
standard deviations are recorded in the CSV and shown as error bars in the figures). Timing
uses `perf_counter`; sub-millisecond calls (AES, RSA, plaintext NumPy) sit near the timer's
practical noise floor, so the harness times a batch of calls per sample and reports the
per-call mean — small non-monotonicities across sizes in such microsecond figures should be
read as jitter. Correctness is checked against the NumPy reference (exact equality for
AES/RSA round-trips; tolerance for approximate CKKS). The full configuration, library
versions, and machine details are written to a run-configuration JSON for reproducibility
(current runs also record the exact CPU model and RAM). RSA-2048-OAEP can encrypt at most
~190 bytes, so it cannot encrypt bulk data; it is measured **per single record** and
reported as such, never as a bulk figure. `count` is plaintext metadata, not an HE
computation, and is labelled accordingly.

---

## 4. Results

### 4.1 Correctness and approximation error

Every operation decrypted correctly under all schemes. AES and RSA round-trips are exact.
CKKS is approximate; its error is small and **stable across dataset sizes**, but differs by
operation. We report **both** the mean relative error and the maximum absolute error,
because the two tell different stories (see the note below the table).

| Operation | Mean relative error | Max absolute error (N=100k) | Note |
| --- | --- | --- | --- |
| Addition | ~1.7 × 10⁻¹¹ | ~1.4 × 10⁻⁸ | Pure addition; relatively very precise |
| Sum | ~2 × 10⁻¹¹ – 6 × 10⁻¹⁰ | ~1.1 × 10⁻⁴ | Additions only; tiny *relative* error, but absolute error grows with the large aggregate |
| Multiplication | ~1.3 × 10⁻⁷ | ~1.3 × 10⁻³ | Ciphertext × ciphertext + rescale sets a ~10⁻⁷ relative floor |
| Average (mean) | ~1.1 – 1.4 × 10⁻⁷ | ~5.4 × 10⁻⁶ | = sum × (1/N); the scalar multiply sets the same ~10⁻⁷ floor |

**Reading the table — a normalization caveat.** Relative error is normalized by the result's
magnitude, and that magnitude differs by ~5 orders of magnitude across these operations
(sum ≈ 5 × 10⁶ vs mean ≈ 50). So the apparent gap — sum at ~10⁻¹¹ versus mean at ~10⁻⁷ — is
**largely a normalization artifact, not a real precision difference**: mean is simply
sum × (1/N), so it cannot be "more wrong" than the sum it is built from. **Absolute error is
the more honest signal for aggregates.** The genuine pattern is by operation *type*: pure
additions (add, sum) are relatively very precise (~10⁻¹¹), while operations that include a
ciphertext rescale (multiply, mean) share a ~10⁻⁷ relative floor. Either way the errors are
negligible for analytics but non-zero, as CKKS requires.

*[Figure: ckks_error_vs_size.png — CKKS approximation error vs dataset size, per operation.]*

### 4.2 Runtime and scalability

CKKS runtime (packed, mean over 5 runs), in milliseconds:

| N | Encrypt | Add (compute) | Multiply (compute) | Sum (compute) | Mean (compute) |
| --- | --- | --- | --- | --- | --- |
| 200 | 3.1 | 0.06 | 2.3 | 30 | 33 |
| 1,000 | 2.8 | 0.06 | 2.3 | 65 | 75 |
| 10,000 | 8.7 | 0.14 | 6.6 | 89 | 90 |
| 100,000 | 73 | 1.5 | 54 | **503** | 501 |

Two patterns stand out. First, **encryption and aggregation dominate**: `sum` and `mean`
are far more expensive than element-wise `add`/`multiply` because reducing across slots
requires ~log₂(N) Galois-key rotations per ciphertext chunk, each costly. Second, the cost
scales with the number of ciphertext chunks — at N = 100,000 the data occupies 25
ciphertexts, so per-chunk work multiplies; the headline `sum` figure is 503 ± 20 ms
(mean ± std over the five runs). Decryption is cheap for a scalar result (~0.7 ms) but
larger for a full vector result (~14–20 ms at N = 100,000). A bookkeeping note: the
timed totals count **one** dataset encryption — for the two-operand ops (add, multiply)
the second operand is encrypted once outside the timed phase, so an end-to-end two-operand
workflow would add roughly one more encryption.

Note that `mean` is not independent of `sum`: it is computed as `sum` followed by a plaintext
scalar multiply (mean = sum × 1/N), so its cost necessarily *includes* sum's. The two are
therefore near-identical (e.g. 501 vs 503 ms at N = 100,000), as expected — not two separate
data points.

*[Figure: ckks_runtime_vs_size.png — CKKS total time vs dataset size, per operation.]*
*[Figure: ckks_cost_breakdown.png — encrypt / compute / decrypt split at the largest size.]*

### 4.3 Size overhead — ciphertext and the fixed key/context cost

**Ciphertext size is the authoritative memory/size signal in this report.** (Process-level
peak-memory sampling proved too noisy to report responsibly; see Limitations.) The encrypted
dataset versus the raw plaintext:

| N | Plaintext | CKKS ciphertext | Chunks | Overhead |
| --- | --- | --- | --- | --- |
| 200 | 1.6 KB | 334 KB | 1 | ~209× |
| 1,000 | 8 KB | 334 KB | 1 | ~42× |
| 10,000 | 80 KB | 1.0 MB | 3 | ~12.5× |
| 100,000 | 800 KB | 8.36 MB | 25 | ~10.4× |

A subtle but important finding: **a single CKKS ciphertext is ~334 KB regardless of how many
of its 4,096 slots are filled.** Small datasets therefore waste most of a ciphertext, which
is why the overhead ratio is enormous at N = 200 (~209×) and falls toward ~10× only once the
data is large enough to fill the slots. By contrast, AES adds just 28 bytes (a 12-byte nonce
plus a 16-byte authentication tag) to the plaintext, at any size.

**The fixed key/context overhead is a finding in its own right.** Before any value is
encrypted or any computation is run, the public context — public key plus the Galois keys
(for rotations) and relinearization keys (for multiplication) — is **≈ 34 MB
(35,476,814 bytes; 33.8 MiB)**, independent of dataset size. This fixed cost is frequently
overlooked in discussions of HE. An 8 MB ciphertext for 100,000 records is expected; a fixed
34 MB of keys that must be generated and shipped to the compute party before any work begins
is exactly what makes HE impractical for small or latency-sensitive workloads. For a dataset
of 200 records the keys outweigh the encrypted data by ~100×.

*[Figure: ciphertext_size_vs_size.png — encrypted vs raw data size across dataset sizes.]*

### 4.4 Packing: why batching is mandatory

CKKS supports two encryption granularities: **packed** (the dataset is batched into
4,096-slot ciphertexts) and **element-wise** (one ciphertext per value). The difference is
decisive. At **N = 200** (the only size at which element-wise completes — see below):

| Granularity | Encrypt time | Ciphertext size |
| --- | --- | --- |
| Packed | 3.1 ms | 0.33 MB |
| Element-wise | ~540–570 ms | 66.9 MB |
| **Ratio** | **~180× slower** | **~200× larger** |

This is the project's clearest practical lesson: **packing is not an optimization, it is a
prerequisite.** Honest caveat about the evidence: element-wise encryption is O(N) in large
ciphertexts and exhausts container memory above roughly N = 200, so it **does not run at
larger sizes.** The figure below compares the two granularities across the four operations at
N = 200; it is a per-operation comparison at one size, **not** a size-scaling trend. The
finding is simply: at N = 200 element-wise is ~180× slower and ~200× larger, and beyond that
it does not run at all.

*[Figure: packed_vs_elementwise.png — packed vs element-wise total time, per operation, N=200.]*

### 4.5 The headline: computing on encrypted data, traditional vs HE

This is the comparison the project exists to make. **Scenario:** both sides start from data
**stored encrypted at rest** under their own scheme. To compute an aggregate (here, `sum`)
over that encrypted data:

- **Traditional (AES):** decrypt the dataset → compute on plaintext → (re-encrypt result).
  The data is **exposed in plaintext** to the compute party during the computation.
- **HE (CKKS), steady-state:** compute directly on the ciphertext, decrypt only the result.
  The data is **never exposed**. (A *one-shot* variant additionally pays the CKKS
  encryption of the dataset — the first upload to an untrusted cloud.)

At **N = 100,000**:

| Workflow | Time for `sum` | Plaintext exposed during compute? |
| --- | --- | --- |
| AES: decrypt + compute | **~0.16 ms** | **Yes** |
| HE steady-state: compute + result decrypt | **~503 ms** (≈576 ms one-shot incl. CKKS encryption) | **No** |

HE is roughly **3,000× slower** for this task. That number is meaningless without its other
half: the HE result was produced **without the compute party ever seeing the data.** The cost
*is* the security property.

*[Figure: workflow_comparison.png — analytics cost: AES (decrypt+compute) vs HE steady-state and one-shot.]*
*[Figure: protection_cost.png — whole-dataset encrypt+decrypt cost: CKKS vs AES, plus RSA for a single record.]*

---

## 5. Security analysis

Performance is only half of the assignment; the schemes also differ fundamentally in what
they secure.

| Scheme | Type | Security level | Computes on ciphertext? | Protects data **in use**? | Quantum posture |
| --- | --- | --- | --- | --- | --- |
| AES-256-GCM | Symmetric (authenticated) | ~256-bit (~128-bit vs Grover) | No | No | Resists Grover (key doubling) |
| RSA-2048-OAEP | Asymmetric | ~112-bit (NIST SP 800-57) | No | No | **Broken by Shor** |
| CKKS (TenSEAL) | Homomorphic (RLWE lattice) | ~128-bit | **Yes** | **Yes** | Believed quantum-resistant |

**HE's unique security benefit** is protecting data *in use*: an untrusted server can compute
on ciphertext it cannot read. Two honest qualifications keep this from being overstated:

- **Scope of protection.** AES-GCM also provides **integrity/authenticity**; CKKS provides
  **confidentiality only** and is **approximate**, so results carry (measured) error and
  integrity must be added separately (e.g., verifiable computation) if required.
- **Approximate-decryption caveat.** For CKKS specifically, releasing the decrypted
  approximate results to an adversary can leak information about the secret key
  (Li & Micciancio, 2021); this is mitigated by noise flooding before sharing decryptions.

On the quantum axis, CKKS (lattice/RLWE) is a post-quantum candidate and RSA-2048 is not,
which is a genuine point in HE's favour beyond the in-use property.

---

## 6. Discussion — when is HE practical?

The measurements support a clear, bounded recommendation.

**HE is worth the cost when** the requirement is precisely *"compute on data the compute
party must never see"* — for example, analytics outsourced to an untrusted cloud, or
aggregation across parties that cannot share raw data — and when the workload tolerates a
~10²–10³× slowdown and ~10× data inflation. Our errors confirm the results remain accurate
enough for such analytics.

**HE is impractical when** latency or footprint dominate. Three measured reasons:
1. **The fixed ~34 MB key/context overhead** makes small or per-request workloads absurd —
   for 200 records the keys dwarf the data ~100×.
2. **Packing is mandatory.** Without batching, both time and size blow up ~200×; element-wise
   use is effectively unusable.
3. **HE is depth-limited, not merely slow — and we measured the limit.** Our parameter
   chain `[60, 40, 40, 60]` provides two rescaling levels. A dedicated experiment
   (`experiments/depth_sweep.py`, repeated ciphertext-squaring of unit-scale values)
   confirms it empirically: the mean relative error is **~1.3 × 10⁻⁷ after one
   multiplication**, grows ~7× to **~9.4 × 10⁻⁷ after two**, and a **third sequential
   multiplication fails outright** ("scale out of bounds" — the modulus chain is
   exhausted). Deeper circuits require *bootstrapping* (an expensive noise-refresh) or
   larger parameters (more cost and memory). Our benchmark deliberately stays within this
   budget — single multiplications — which is why the multiply error stays at ~10⁻⁷ rather
   than diverging. The take-away is that HE's real constraint is not only runtime but
   **circuit depth**: it suits shallow analytics (sums, averages, single products), not
   arbitrary deep computation, without further machinery.

*[Figure: ckks_error_vs_depth.png — measured CKKS error per multiplicative depth, and the depth-3 failure.]*

---

## 7. Limitations

- **Synthetic data.** We benchmark uniform-random `float64` arrays, not a named real-world
  dataset. The pipeline is dataset-agnostic; substituting a real numeric dataset is
  straightforward and is future work.
- **Process memory not reported as a headline.** We sampled process resident-set size (RSS)
  per operation — via psutil rather than `tracemalloc`, since TenSEAL allocates in C++
  memory that `tracemalloc` cannot see — but RSS deltas are unreliable for short operations
  (freed pages are reused, so later operations read near-zero). Rather than present a number
  we do not trust, we treat **ciphertext size plus the fixed key/context size** as the
  authoritative memory signal; the raw RSS column stays in the CSV, and no RSS figure is
  generated.
- **CKKS only.** We did not implement BFV/BGV. Exact integer `sum`/`count` under BFV would
  complement CKKS's approximate real arithmetic and is the most natural next step.
- **Element-wise capped at N ≈ 200.** Element-wise encryption exhausts container memory above
  that size; the packed path runs to 100,000.

---

## 8. Conclusion and future work

We built a reproducible benchmark that runs four numerical operations under CKKS HE and under
AES-256/RSA-2048 baselines, verifies every HE result against a plaintext reference, and
measures runtime, ciphertext size, approximation error, and scalability to 100,000 records.
The central, honest finding is that HE buys a capability the baselines lack — computation on
encrypted data, with data never exposed in use — at a measured cost of roughly three orders
of magnitude in time, ~10× in ciphertext size, a fixed ~34 MB of keys, and a measured hard
limit of two multiplications before the modulus chain is exhausted (bootstrapping or larger
parameters would be needed beyond that). Whether that trade is acceptable depends entirely
on whether protecting data *in use* is a requirement.

**Future work:** add BFV for exact integer aggregation; benchmark a real public numeric
dataset; explore deeper circuits with bootstrapping and alternative parameter sets; and
evaluate GPU-accelerated HE backends.

---

## References

1. J. H. Cheon, A. Kim, M. Kim, Y. Song. *Homomorphic Encryption for Arithmetic of
   Approximate Numbers (CKKS).* ASIACRYPT 2017.
2. A. Benaissa, B. Retiat, B. Cebere, A. E. Belfedhal. *TenSEAL: A Library for Encrypted
   Tensor Operations Using Homomorphic Encryption.* 2021.
3. M. Albrecht et al. *Homomorphic Encryption Security Standard.* HomomorphicEncryption.org,
   2018.
4. National Institute of Standards and Technology. *SP 800-57 Part 1 Rev. 5: Recommendation
   for Key Management.* 2020.
5. B. Li, D. Micciancio. *On the Security of Homomorphic Encryption on Approximate Numbers.*
   EUROCRYPT 2021.
6. NIST. *FIPS 197: Advanced Encryption Standard (AES).* 2001.
7. K. Moriarty et al. *PKCS #1 v2.2 / RFC 8017: RSA Cryptography Specifications.* 2016.

---

## Appendix A — Reproduction

```bash
docker compose build
docker compose run --rm benchmark pytest -v     # 21 tests
docker compose run --rm benchmark               # -> results/results.csv + run_config.json
docker compose run --rm benchmark python experiments/depth_sweep.py  # -> depth_sweep.json
docker compose run --rm plots                   # -> figures/*.png
docker compose up dashboard                      # interactive UI at http://localhost:8501
```

All numbers in this report are derived from `results/results.csv` (48 rows: every
scheme × operation × dataset size, mean of 5 repeats after a warm-up) and
`results/depth_sweep.json` (the error-vs-depth experiment). CKKS parameters and
environment are recorded in `results/run_config.json`.

## Appendix B — Baseline reference numbers

- **AES-256-GCM**, protect (encrypt + decrypt), total: 0.030 ms (N=200), 0.018 ms (1k),
  0.033 ms (10k), 0.330 ms (100k); ciphertext size = N × 8 + 28 bytes.
- **RSA-2048-OAEP**, per single 8-byte record: ~0.02 ms encrypt, ~0.20 ms decrypt; ciphertext
  256 bytes per record. Bulk data would use hybrid encryption (RSA wraps an AES key).
- *Reading these numbers:* they are microseconds, near `perf_counter`'s practical noise
  floor at five repeats — the non-monotonicity across sizes (1k faster than 200) is timing
  jitter, not a real effect. The measurement harness now batches sub-millisecond calls per
  timing sample to suppress exactly this.
