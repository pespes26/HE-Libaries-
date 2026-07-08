// Generates report/FINAL_REPORT.docx from the measured data + figures.
// Run: node report/build_docx.js   (requires global `docx` package)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  TableOfContents, PageBreak, Footer, PageNumber,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "figures");
const CONTENT_W = 9360; // US Letter, 1" margins

// ---- helpers ----------------------------------------------------------------
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const P = (t, opts = {}) => new Paragraph({ spacing: { after: 140 }, ...opts,
  children: [new TextRun({ text: t })] });

// rich paragraph from [ {text, bold, italics} ... ]
const RP = (runs, opts = {}) => new Paragraph({ spacing: { after: 140 }, ...opts,
  children: runs.map((r) => new TextRun(r)) });

const bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 },
  spacing: { after: 60 }, children: [new TextRun(t)] });
const numItem = (t) => new Paragraph({ numbering: { reference: "nums", level: 0 },
  spacing: { after: 60 }, children: [new TextRun(t)] });

const border = { style: BorderStyle.SINGLE, size: 1, color: "C0B8AA" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, w, { head = false, bold = false } = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: { fill: head ? "DDEDEC" : "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: String(text),
      bold: head || bold, size: 18 })] })],
  });
}

function table(headerRow, rows, widths) {
  const trows = [new TableRow({ tableHeader: true,
    children: headerRow.map((h, i) => cell(h, widths[i], { head: true })) })];
  for (const r of rows) {
    trows.push(new TableRow({ children: r.map((c, i) => cell(c, widths[i])) }));
  }
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths, rows: trows });
}

function figure(file, caption, figW = 7, figH = 4.5) {
  const data = fs.readFileSync(path.join(FIG, file));
  const w = 600, h = Math.round(w * figH / figW);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data,
        transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 180 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: "5c574e" })] }),
  ];
}

// ---- content ----------------------------------------------------------------
const children = [];

// Title page
children.push(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 2400, after: 120 },
    children: [new TextRun({ text: "Benchmarking Homomorphic Encryption", bold: true, size: 52 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "A reproducible study of the overhead of computing on encrypted data",
      italics: true, size: 26, color: "5c574e" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: "Bar Pesso  ·  Shay Harush  ·  Shon Platok", size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Confidential Computing", size: 24, color: "5c574e" })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// TOC
children.push(
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: "Contents", bold: true, size: 32 })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
);

// Abstract
children.push(H1("Abstract"));
children.push(P("Homomorphic Encryption (HE) allows computation directly on encrypted data, a capability that conventional ciphers such as AES and RSA do not provide. This project builds a reproducible benchmark that measures the practical overhead of HE for simple numerical analytics — runtime, ciphertext size, and correctness — against AES-256 and RSA-2048 baselines, and frames the comparison honestly. The central result, measured on a dataset of 100,000 records, is one line:"));
children.push(new Paragraph({ spacing: { before: 80, after: 80 }, indent: { left: 360 },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: "00857F", space: 12 } },
  children: [new TextRun({ text: "Computing a sum the traditional way costs ~0.16 ms but exposes the plaintext to the compute party. Computing the same sum homomorphically costs ~500 ms and never exposes the data.", italics: true, bold: true })] }));
children.push(P("HE is therefore roughly three orders of magnitude slower for this task, carries a ~10× ciphertext-size overhead, and requires a fixed ~34 MB of keys before any computation begins — yet it is the only one of the three schemes that keeps data encrypted during processing. The benchmark quantifies exactly when that trade is worth making."));

// 1. Introduction
children.push(H1("1. Introduction and research question"));
children.push(P("Confidential computing addresses a gap left by conventional cryptography: AES and RSA protect data at rest and in transit, but to compute on that data it must first be decrypted, exposing it to whoever runs the computation. Homomorphic Encryption removes that exposure by computing on ciphertext directly."));
children.push(RP([{ text: "Research question. ", bold: true }, { text: "For basic numerical operations, what overhead does HE add in runtime, memory, ciphertext size, and correctness — and when is that overhead acceptable?" }]));
children.push(P("We answer it empirically with a reproducible prototype that runs a fixed set of operations (addition, multiplication, sum, average) under HE and under classical baselines, verifies every HE result against a plaintext reference, and records the cost."));

// 2. Asymmetry
children.push(H1("2. The fundamental asymmetry (why this comparison is “unfair” by design)"));
children.push(P("The most important framing of this project is that comparing HE to AES/RSA is inherently unfair, because they solve different problems. Stating runtime numbers without this frame would be misleading."));
children.push(table(
  ["Property", "AES-256 / RSA-2048", "CKKS (HE)"],
  [
    ["Protects data at rest", "Yes", "Yes"],
    ["Protects data in transit", "Yes", "Yes"],
    ["Protects data in use (during computation)", "No", "Yes"],
    ["Can compute on ciphertext", "No", "Yes"],
  ], [4360, 2500, 2500]));
children.push(P("To analyze AES- or RSA-protected data you must decrypt it first, perform the computation on plaintext, then optionally re-encrypt. During that computation the data is fully exposed to the compute party. HE computes on the ciphertext and only the final result is decrypted, by the key holder.", { spacing: { before: 120, after: 140 } }));
children.push(P("Consequently this report does not claim HE is “better” than AES or RSA. It measures the cost of a capability that AES and RSA simply do not have. Every result below should be read as the price of keeping data encrypted while it is being used."));

// 3. Methodology
children.push(H1("3. Methodology"));
children.push(H2("3.1 Schemes and tools"));
children.push(table(
  ["Layer", "Choice", "Version"],
  [
    ["HE scheme / library", "CKKS via TenSEAL", "0.3.16"],
    ["Classical baselines", "AES-256-GCM, RSA-2048-OAEP (cryptography)", "42.0.8"],
    ["Numerics / reference", "NumPy", "1.26.4"],
    ["Data / analysis", "pandas", "2.2.2"],
    ["Plotting", "matplotlib", "3.8.4"],
    ["Tests", "pytest", "8.2.2"],
    ["Runtime", "Docker (python:3.11-slim)", "—"],
  ], [3120, 4240, 2000]));
children.push(P("CKKS was chosen because it operates on real-valued vectors and supports division by a constant, which covers all four target operations including the average. CKKS is an approximate scheme, so every decrypted result is compared against an exact plaintext reference and its error is reported.", { spacing: { before: 120, after: 140 } }));
children.push(H2("3.2 CKKS parameters"));
children.push(P("poly_modulus_degree = 8192 (4096 usable slots per ciphertext), coeff_mod_bit_sizes = [60, 40, 40, 60], global_scale = 2^40. The coefficient-modulus chain totals 200 bits, within the limit for ~128-bit security at this ring dimension. Datasets larger than 4096 values are split into ceil(N / 4096) ciphertexts (“chunking”); sum is computed per chunk via Galois-key rotations and then combined."));
children.push(H2("3.3 Datasets"));
children.push(P("Synthetic numerical data: 1-D arrays of float64 values drawn uniformly from [0, 100), generated with a fixed NumPy seed so every run is reproducible. The same data is encrypted under all schemes (a controlled comparison, stronger than two unrelated datasets). Sizes benchmarked: 200, 1,000, 10,000, and 100,000 records. The data is synthetic and non-sensitive, chosen for reproducibility and because CKKS is real-valued."));
children.push(H2("3.4 Fair-comparison rules and measurement"));
children.push(P("Costs are separated into three buckets so no scheme is judged unfairly:"));
children.push(numItem("Data protection — encryption / decryption (all schemes)."));
children.push(numItem("Computation — HE on ciphertext vs. plaintext computation (measured separately)."));
children.push(numItem("Size overhead — ciphertext vs. plaintext bytes."));
children.push(P("Each measurement runs five times after a discarded warm-up; we report the mean (standard deviations are recorded in the CSV and shown as error bars in the figures). Timing uses perf_counter; sub-millisecond calls (AES, RSA, plaintext NumPy) sit near the timer's practical noise floor, so the harness times a batch of calls per sample and reports the per-call mean — small non-monotonicities across sizes in such microsecond figures should be read as jitter. Correctness is checked against the NumPy reference (exact equality for AES/RSA round-trips; tolerance for approximate CKKS). The full configuration, library versions, and machine details are written to a run-configuration JSON for reproducibility (current runs also record the exact CPU model and RAM). RSA-2048-OAEP can encrypt at most ~190 bytes, so it cannot encrypt bulk data; it is measured per single record and reported as such, never as a bulk figure. count is plaintext metadata, not an HE computation, and is labelled accordingly.", { spacing: { before: 120, after: 140 } }));

// 4. Results
children.push(H1("4. Results"));
children.push(H2("4.1 Correctness and approximation error"));
children.push(P("Every operation decrypted correctly under all schemes. AES and RSA round-trips are exact. CKKS is approximate; its error is small and stable across dataset sizes, but differs by operation. We report both the mean relative error and the maximum absolute error, because the two tell different stories (see the note below the table)."));
children.push(table(
  ["Operation", "Mean relative error", "Max absolute error (N=100k)", "Note"],
  [
    ["Addition", "~1.7 × 10⁻¹¹", "~1.4 × 10⁻⁸", "Pure addition; relatively very precise"],
    ["Sum", "~2 × 10⁻¹¹ – 6 × 10⁻¹⁰", "~1.1 × 10⁻⁴", "Additions only; tiny relative error, but absolute error grows with the large aggregate"],
    ["Multiplication", "~1.3 × 10⁻⁷", "~1.3 × 10⁻³", "Ciphertext × ciphertext + rescale sets a ~10⁻⁷ relative floor"],
    ["Average (mean)", "~1.1 – 1.4 × 10⁻⁷", "~5.4 × 10⁻⁶", "= sum × (1/N); the scalar multiply sets the same ~10⁻⁷ floor"],
  ], [1700, 2100, 2300, 3260]));
children.push(RP([{ text: "Reading the table — a normalization caveat. ", bold: true }, { text: "Relative error is normalized by the result’s magnitude, and that magnitude differs by ~5 orders of magnitude across these operations (sum ≈ 5 × 10⁶ vs mean ≈ 50). So the apparent gap — sum at ~10⁻¹¹ versus mean at ~10⁻⁷ — is largely a normalization artifact, not a real precision difference: mean is simply sum × (1/N), so it cannot be “more wrong” than the sum it is built from. Absolute error is the more honest signal for aggregates. The genuine pattern is by operation type: pure additions (add, sum) are relatively very precise (~10⁻¹¹), while operations that include a ciphertext rescale (multiply, mean) share a ~10⁻⁷ relative floor. Either way the errors are negligible for analytics but non-zero, as CKKS requires." }], { spacing: { before: 120, after: 140 } }));
children.push(...figure("ckks_error_vs_size.png", "Figure 1. CKKS approximation error vs dataset size, per operation."));
children.push(H2("4.2 Runtime and scalability"));
children.push(P("CKKS runtime (packed, mean over 5 runs), in milliseconds:"));
children.push(table(
  ["N", "Encrypt", "Add (compute)", "Multiply (compute)", "Sum (compute)", "Mean (compute)"],
  [
    ["200", "3.1", "0.06", "2.3", "30", "33"],
    ["1,000", "2.8", "0.06", "2.3", "65", "75"],
    ["10,000", "8.7", "0.14", "6.6", "89", "90"],
    ["100,000", "73", "1.5", "54", "503", "501"],
  ], [1360, 1500, 1700, 1800, 1500, 1500]));
children.push(P("Two patterns stand out. First, encryption and aggregation dominate: sum and mean are far more expensive than element-wise add/multiply because reducing across slots requires ~log₂(N) Galois-key rotations per ciphertext chunk, each costly. Second, the cost scales with the number of ciphertext chunks — at N = 100,000 the data occupies 25 ciphertexts, so per-chunk work multiplies; the headline sum figure is 503 ± 20 ms (mean ± std over the five runs). Decryption is cheap for a scalar result (~0.7 ms) but larger for a full vector result (~14–20 ms at N = 100,000). A bookkeeping note: the timed totals count ONE dataset encryption — for the two-operand ops (add, multiply) the second operand is encrypted once outside the timed phase, so an end-to-end two-operand workflow would add roughly one more encryption.", { spacing: { before: 120, after: 140 } }));
children.push(P("Note that mean is not independent of sum: it is computed as sum followed by a plaintext scalar multiply (mean = sum × 1/N), so its cost necessarily includes sum’s. The two are therefore near-identical (501 vs 503 ms at N = 100,000), as expected — not two separate data points.", { spacing: { after: 140 } }));
children.push(...figure("ckks_runtime_vs_size.png", "Figure 2. CKKS total time vs dataset size, per operation."));
children.push(...figure("ckks_cost_breakdown.png", "Figure 3. CKKS cost breakdown (encrypt / compute / decrypt), N = 100,000."));
children.push(H2("4.3 Size overhead — ciphertext and the fixed key/context cost"));
children.push(RP([{ text: "Ciphertext size is the authoritative memory/size signal in this report. ", bold: true }, { text: "(Process-level peak-memory sampling proved too noisy to report responsibly; see Limitations.) The encrypted dataset versus the raw plaintext:" }]));
children.push(table(
  ["N", "Plaintext", "CKKS ciphertext", "Chunks", "Overhead"],
  [
    ["200", "1.6 KB", "334 KB", "1", "~209×"],
    ["1,000", "8 KB", "334 KB", "1", "~42×"],
    ["10,000", "80 KB", "1.0 MB", "3", "~12.5×"],
    ["100,000", "800 KB", "8.36 MB", "25", "~10.4×"],
  ], [1560, 1950, 2350, 1500, 2000]));
children.push(P("A subtle but important finding: a single CKKS ciphertext is ~334 KB regardless of how many of its 4,096 slots are filled. Small datasets therefore waste most of a ciphertext, which is why the overhead ratio is enormous at N = 200 (~209×) and falls toward ~10× only once the data is large enough to fill the slots. By contrast, AES adds just 28 bytes (a 12-byte nonce plus a 16-byte authentication tag) to the plaintext, at any size.", { spacing: { before: 120, after: 140 } }));
children.push(RP([{ text: "The fixed key/context overhead is a finding in its own right. ", bold: true }, { text: "Before any value is encrypted or any computation is run, the public context — public key plus the Galois keys (for rotations) and relinearization keys (for multiplication) — is ≈ 34 MB (35,476,814 bytes; 33.8 MiB), independent of dataset size. This fixed cost is frequently overlooked in discussions of HE. An 8 MB ciphertext for 100,000 records is expected; a fixed 34 MB of keys that must be generated and shipped to the compute party before any work begins is exactly what makes HE impractical for small or latency-sensitive workloads. For a dataset of 200 records the keys outweigh the encrypted data by ~100×." }]));
children.push(...figure("ciphertext_size_vs_size.png", "Figure 4. Encrypted vs raw data size across dataset sizes."));
children.push(H2("4.4 Packing: why batching is mandatory"));
children.push(P("CKKS supports two encryption granularities: packed (the dataset is batched into 4,096-slot ciphertexts) and element-wise (one ciphertext per value). The difference is decisive. At N = 200 (the only size at which element-wise completes — see below):"));
children.push(table(
  ["Granularity", "Encrypt time", "Ciphertext size"],
  [
    ["Packed", "3.1 ms", "0.33 MB"],
    ["Element-wise", "~540–570 ms", "66.9 MB"],
    ["Ratio", "~180× slower", "~200× larger"],
  ], [3120, 3120, 3120]));
children.push(RP([{ text: "This is the project’s clearest practical lesson: packing is not an optimization, it is a prerequisite. ", bold: true }, { text: "Honest caveat about the evidence: element-wise encryption is O(N) in large ciphertexts and exhausts container memory above roughly N = 200, so it does not run at larger sizes. The figure below compares the two granularities across the four operations at N = 200; it is a per-operation comparison at one size, not a size-scaling trend. The finding is simply: at N = 200 element-wise is ~180× slower and ~200× larger, and beyond that it does not run at all." }], { spacing: { before: 120, after: 140 } }));
children.push(...figure("packed_vs_elementwise.png", "Figure 5. Packed vs element-wise total time, per operation, N = 200 (log scale)."));
children.push(H2("4.5 The headline: computing on encrypted data, traditional vs HE"));
children.push(P("This is the comparison the project exists to make. Scenario: both sides start from data stored encrypted at rest under their own scheme. To compute an aggregate (here, sum) over that encrypted data:"));
children.push(bullet("Traditional (AES): decrypt the dataset → compute on plaintext → (re-encrypt result). The data is exposed in plaintext to the compute party during the computation."));
children.push(bullet("HE (CKKS), steady-state: compute directly on the ciphertext, decrypt only the result. The data is never exposed. (A one-shot variant additionally pays the CKKS encryption of the dataset — the first upload to an untrusted cloud.)"));
children.push(P("At N = 100,000:", { spacing: { before: 100, after: 80 } }));
children.push(table(
  ["Workflow", "Time for sum", "Plaintext exposed during compute?"],
  [
    ["AES: decrypt + compute", "~0.16 ms", "Yes"],
    ["HE steady-state: compute + result decrypt", "~503 ms (≈576 ms one-shot incl. CKKS encryption)", "No"],
  ], [3360, 3000, 3000]));
children.push(RP([{ text: "HE is roughly 3,000× slower for this task. That number is meaningless without its other half: the HE result was produced without the compute party ever seeing the data. The cost ", }, { text: "is", italics: true }, { text: " the security property." }], { spacing: { before: 120, after: 140 } }));
children.push(...figure("workflow_comparison.png", "Figure 6. Analytics cost: AES (decrypt + compute) vs HE steady-state and one-shot.", 7.8, 4.6));
children.push(...figure("protection_cost.png", "Figure 7. Whole-dataset encrypt + decrypt cost: CKKS vs AES, plus RSA for a single record."));

// 5. Security
children.push(H1("5. Security analysis"));
children.push(P("Performance is only half of the assignment; the schemes also differ fundamentally in what they secure."));
children.push(table(
  ["Scheme", "Type", "Security level", "Computes on ciphertext?", "Protects data in use?", "Quantum posture"],
  [
    ["AES-256-GCM", "Symmetric (authenticated)", "~256-bit (~128 vs Grover)", "No", "No", "Resists Grover"],
    ["RSA-2048-OAEP", "Asymmetric", "~112-bit (NIST)", "No", "No", "Broken by Shor"],
    ["CKKS (TenSEAL)", "Homomorphic (RLWE)", "~128-bit", "Yes", "Yes", "Believed quantum-resistant"],
  ], [1700, 1660, 1700, 1300, 1300, 1700]));
children.push(RP([{ text: "HE’s unique security benefit ", bold: true }, { text: "is protecting data in use: an untrusted server can compute on ciphertext it cannot read. Two honest qualifications keep this from being overstated:" }], { spacing: { before: 120, after: 100 } }));
children.push(bullet("Scope of protection. AES-GCM also provides integrity/authenticity; CKKS provides confidentiality only and is approximate, so results carry (measured) error and integrity must be added separately (e.g., verifiable computation) if required."));
children.push(bullet("Approximate-decryption caveat. For CKKS specifically, releasing the decrypted approximate results to an adversary can leak information about the secret key (Li & Micciancio, 2021); this is mitigated by noise flooding before sharing decryptions."));
children.push(P("On the quantum axis, CKKS (lattice/RLWE) is a post-quantum candidate and RSA-2048 is not, which is a genuine point in HE’s favour beyond the in-use property.", { spacing: { before: 100, after: 140 } }));

// 6. Discussion
children.push(H1("6. Discussion — when is HE practical?"));
children.push(P("The measurements support a clear, bounded recommendation."));
children.push(RP([{ text: "HE is worth the cost when ", bold: true }, { text: "the requirement is precisely “compute on data the compute party must never see” — for example, analytics outsourced to an untrusted cloud, or aggregation across parties that cannot share raw data — and when the workload tolerates a ~10²–10³× slowdown and ~10× data inflation. Our errors confirm the results remain accurate enough for such analytics." }]));
children.push(RP([{ text: "HE is impractical when ", bold: true }, { text: "latency or footprint dominate. Three measured reasons:" }]));
children.push(numItem("The fixed ~34 MB key/context overhead makes small or per-request workloads absurd — for 200 records the keys dwarf the data ~100×."));
children.push(numItem("Packing is mandatory. Without batching, both time and size blow up ~200×; element-wise use is effectively unusable."));
children.push(numItem("HE is depth-limited, not merely slow — and we measured the limit. Our parameter chain [60, 40, 40, 60] provides two rescaling levels. A dedicated experiment (experiments/depth_sweep.py, repeated ciphertext-squaring of unit-scale values) confirms it empirically: the mean relative error is ~1.3 × 10⁻⁷ after one multiplication, grows ~7× to ~9.4 × 10⁻⁷ after two, and a third sequential multiplication fails outright (\"scale out of bounds\" — the modulus chain is exhausted). Deeper circuits require bootstrapping (an expensive noise-refresh) or larger parameters. Our benchmark deliberately stays within this budget — single multiplications — which is why the multiply error stays at ~10⁻⁷ rather than diverging. The real constraint is not only runtime but circuit depth: HE suits shallow analytics (sums, averages, single products), not arbitrary deep computation, without further machinery."));
children.push(...figure("ckks_error_vs_depth.png", "Figure 8. Measured CKKS error per multiplicative depth — and the depth-3 failure (modulus chain exhausted)."));

// 7. Limitations
children.push(H1("7. Limitations"));
children.push(bullet("Synthetic data. We benchmark uniform-random float64 arrays, not a named real-world dataset. The pipeline is dataset-agnostic; substituting a real numeric dataset is straightforward and is future work."));
children.push(bullet("Process memory not reported as a headline. We sampled process resident-set size (RSS) per operation — via psutil rather than tracemalloc, since TenSEAL allocates in C++ memory that tracemalloc cannot see — but RSS deltas are unreliable for short operations (freed pages are reused, so later operations read near-zero). Rather than present a number we do not trust, we treat ciphertext size plus the fixed key/context size as the authoritative memory signal; the raw RSS column stays in the CSV, and no RSS figure is generated."));
children.push(bullet("CKKS only. We did not implement BFV/BGV. Exact integer sum/count under BFV would complement CKKS’s approximate real arithmetic and is the most natural next step."));
children.push(bullet("Element-wise capped at N ≈ 200. Element-wise encryption exhausts container memory above that size; the packed path runs to 100,000."));

// 8. Conclusion
children.push(H1("8. Conclusion and future work"));
children.push(P("We built a reproducible benchmark that runs four numerical operations under CKKS HE and under AES-256/RSA-2048 baselines, verifies every HE result against a plaintext reference, and measures runtime, ciphertext size, approximation error, and scalability to 100,000 records. The central, honest finding is that HE buys a capability the baselines lack — computation on encrypted data, with data never exposed in use — at a measured cost of roughly three orders of magnitude in time, ~10× in ciphertext size, a fixed ~34 MB of keys, and a measured hard limit of two multiplications before the modulus chain is exhausted (bootstrapping or larger parameters would be needed beyond that). Whether that trade is acceptable depends entirely on whether protecting data in use is a requirement."));
children.push(RP([{ text: "Future work: ", bold: true }, { text: "add BFV for exact integer aggregation; benchmark a real public numeric dataset; explore deeper circuits with bootstrapping and alternative parameter sets; and evaluate GPU-accelerated HE backends." }]));

// References
children.push(H1("References"));
const refs = [
  "J. H. Cheon, A. Kim, M. Kim, Y. Song. Homomorphic Encryption for Arithmetic of Approximate Numbers (CKKS). ASIACRYPT 2017.",
  "A. Benaissa, B. Retiat, B. Cebere, A. E. Belfedhal. TenSEAL: A Library for Encrypted Tensor Operations Using Homomorphic Encryption. 2021.",
  "M. Albrecht et al. Homomorphic Encryption Security Standard. HomomorphicEncryption.org, 2018.",
  "NIST. SP 800-57 Part 1 Rev. 5: Recommendation for Key Management. 2020.",
  "B. Li, D. Micciancio. On the Security of Homomorphic Encryption on Approximate Numbers. EUROCRYPT 2021.",
  "NIST. FIPS 197: Advanced Encryption Standard (AES). 2001.",
  "K. Moriarty et al. PKCS #1 v2.2 / RFC 8017: RSA Cryptography Specifications. 2016.",
];
refs.forEach((r) => children.push(numItem(r)));

// Appendices
children.push(H1("Appendix A — Reproduction"));
children.push(P("All numbers in this report are derived from results/results.csv (48 rows: every scheme × operation × dataset size, mean of 5 repeats after a warm-up) and results/depth_sweep.json (the error-vs-depth experiment). CKKS parameters and environment are recorded in results/run_config.json."));
[
  "docker compose build",
  "docker compose run --rm benchmark pytest -v        # 21 tests",
  "docker compose run --rm benchmark                  # -> results/results.csv",
  "docker compose run --rm benchmark python experiments/depth_sweep.py  # -> depth_sweep.json",
  "docker compose run --rm plots                      # -> figures/*.png",
  "docker compose up dashboard                         # http://localhost:8501",
].forEach((c) => children.push(new Paragraph({ spacing: { after: 20 },
  shading: { type: ShadingType.CLEAR, fill: "F0EBE3" },
  children: [new TextRun({ text: c, font: "Consolas", size: 18 })] })));

children.push(H1("Appendix B — Baseline reference numbers"));
children.push(bullet("AES-256-GCM, protect (encrypt + decrypt), total: 0.030 ms (N=200), 0.018 ms (1k), 0.033 ms (10k), 0.330 ms (100k); ciphertext size = N × 8 + 28 bytes."));
children.push(bullet("RSA-2048-OAEP, per single 8-byte record: ~0.02 ms encrypt, ~0.20 ms decrypt; ciphertext 256 bytes per record. Bulk data would use hybrid encryption (RSA wraps an AES key)."));
children.push(bullet("Reading these numbers: they are microseconds, near perf_counter's practical noise floor at five repeats — the non-monotonicity across sizes (1k faster than 200) is timing jitter, not a real effect. The measurement harness now batches sub-millisecond calls per timing sample to suppress exactly this."));

// ---- document ---------------------------------------------------------------
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "1a1814" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: "00857F" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
      { reference: "nums", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Benchmarking Homomorphic Encryption  —  ", size: 16, color: "9c9589" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "9c9589" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "FINAL_REPORT.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote " + out + " (" + (buf.length / 1024).toFixed(0) + " KB)");
});
