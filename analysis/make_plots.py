"""Generate figures from results/results.csv (and results/depth_sweep.json).

Runs headless (Agg backend). Each plot is guarded so a missing slice of data does
not abort the whole script. Produces:
  1. ckks_runtime_vs_size.png    - CKKS total time vs dataset size (packed), per op, +-std.
  2. ckks_cost_breakdown.png     - encrypt/compute/decrypt split (packed, largest size).
  3. ciphertext_size_vs_size.png - encrypted size vs dataset size (CKKS vs AES vs plaintext).
  4. ckks_error_vs_size.png      - CKKS mean relative error vs dataset size, per op.
  5. protection_cost.png         - whole-dataset encrypt+decrypt: CKKS vs AES (+RSA/record).
  6. packed_vs_elementwise.png   - CKKS total time, packed vs element-wise (where both exist).
  7. workflow_comparison.png     - analytics cost: AES (decrypt+compute) vs HE, in both the
                                   steady-state and one-shot scenarios.
                                   Also writes results/workflow_comparison.csv.
  8. ckks_error_vs_depth.png     - CKKS error vs multiplicative depth, incl. the failure
                                   point (from experiments/depth_sweep.py).

No RSS-memory figure is generated: per-operation RSS deltas proved too noisy to plot
responsibly (see README / report Limitations); ciphertext + key sizes are the
authoritative size signal.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def _save(fig, name):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    path = os.path.join(config.FIGURES_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  wrote {path}")


def _total_std_ms(g):
    """Combined std of encrypt+compute+decrypt (independent phases), in ms."""
    import numpy as np
    return np.sqrt(g.encrypt_time_std.fillna(0) ** 2
                   + g.compute_time_std.fillna(0) ** 2
                   + g.decrypt_time_std.fillna(0) ** 2) * 1e3


def plot_runtime_vs_size(df):
    ck = df[(df.scheme == "CKKS") & (df.granularity == "packed")]
    if ck.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for op, g in ck.groupby("operation"):
        g = g.sort_values("dataset_size")
        ax.errorbar(g.dataset_size, g.total_time_mean * 1e3, yerr=_total_std_ms(g),
                    marker="o", capsize=3, label=op)
    ax.set_xscale("log")  # sizes span 200..100k; linear x bunches three of four points
    ax.set_xticks(sorted(ck.dataset_size.unique()))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("dataset size (records, log scale)")
    ax.set_ylabel("total time (ms)  [encrypt + compute + decrypt]")
    ax.set_title("CKKS runtime vs dataset size (packed, mean ± std over 5 runs)\n"
                 "totals include one dataset encryption; add/mul's 2nd operand "
                 "is not timed", fontsize=10)
    ax.legend(title="operation")
    ax.grid(True, alpha=0.3)
    _save(fig, "ckks_runtime_vs_size.png")


def plot_cost_breakdown(df):
    ck = df[(df.scheme == "CKKS") & (df.granularity == "packed")]
    if ck.empty:
        return
    size = ck.dataset_size.max()
    g = ck[ck.dataset_size == size].sort_values("operation")
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ops = g.operation.tolist()
    enc = g.encrypt_time_mean.values * 1e3
    comp = g.compute_time_mean.values * 1e3
    dec = g.decrypt_time_mean.values * 1e3
    ax.bar(ops, enc, label="encrypt")
    ax.bar(ops, comp, bottom=enc, label="compute")
    ax.bar(ops, dec, bottom=enc + comp, label="decrypt")
    ax.set_ylabel("time (ms)")
    ax.set_title(f"CKKS cost breakdown (packed, N={size})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "ckks_cost_breakdown.png")


def plot_ciphertext_size(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    ck = df[(df.scheme == "CKKS") & (df.granularity == "packed") & (df.operation == "sum")]
    if not ck.empty:
        ck = ck.sort_values("dataset_size")
        ax.plot(ck.dataset_size, ck.ciphertext_size_bytes / 1024, marker="o", label="CKKS (packed)")
        plotted = True
    a = df[df.scheme == "AES-256-GCM"].sort_values("dataset_size")
    if not a.empty:
        ax.plot(a.dataset_size, a.ciphertext_size_bytes / 1024, marker="s", label="AES-256-GCM")
        plotted = True
    p = df[(df.scheme == "plaintext") & (df.operation == "sum")].sort_values("dataset_size")
    if not p.empty:
        ax.plot(p.dataset_size, p.ciphertext_size_bytes / 1024, marker="^", label="plaintext (raw)")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")  # 1.6 KB..8 MB in one frame; log-log keeps both ends readable
    ax.set_xticks(sorted(df.dataset_size.dropna().unique()))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("dataset size (records, log scale)")
    ax.set_ylabel("encrypted/raw size (KB, log scale)")
    ax.set_title("Data size: encrypted vs raw\n"
                 "one CKKS ciphertext is ~334 KB however few slots are used",
                 fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    _save(fig, "ciphertext_size_vs_size.png")


def plot_error_vs_size(df):
    ck = df[(df.scheme == "CKKS") & (df.granularity == "packed")]
    if ck.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for op, g in ck.groupby("operation"):
        g = g.sort_values("dataset_size")
        ax.plot(g.dataset_size, g.mean_rel_error, marker="o", label=op)
    ax.set_xscale("log")
    ax.set_xticks(sorted(ck.dataset_size.unique()))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("dataset size (records, log scale)")
    ax.set_ylabel("mean relative error")
    ax.set_yscale("log")
    ax.set_title("CKKS approximation error vs dataset size (packed)\n"
                 "mul/mean share the ~1e-7 rescale floor; add/sum are pure additions",
                 fontsize=10)
    ax.legend(title="operation")
    ax.grid(True, alpha=0.3)
    _save(fig, "ckks_error_vs_size.png")


def plot_protection_cost(df):
    """Whole-dataset encrypt+decrypt cost at the largest common size: CKKS vs AES (+RSA).

    The CKKS bar uses the `add` row: its decrypt phase decrypts the full vector result
    (same shape as the dataset), so encrypt+decrypt is a true dataset round-trip —
    symmetric with the AES bar. (The `sum` row's decrypt is a length-1 result and would
    understate CKKS here.)
    """
    import numpy as np
    # Largest size where all three schemes are present, so no scheme is silently dropped.
    ck_add = df[(df.scheme == "CKKS") & (df.granularity == "packed") & (df.operation == "add")]
    common = (set(ck_add.dataset_size)
              & set(df[df.scheme == "AES-256-GCM"].dataset_size)
              & set(df[df.scheme == "RSA-2048-OAEP"].dataset_size))
    if not common:
        print("  skip protection_cost: no size has CKKS+AES+RSA together")
        return
    size = max(common)
    labels, vals, errs = [], [], []
    ck = ck_add[ck_add.dataset_size == size]
    if not ck.empty:
        r = ck.iloc[0]
        labels.append("CKKS\n(whole dataset)")
        vals.append((r.encrypt_time_mean + r.decrypt_time_mean) * 1e3)
        errs.append(np.sqrt((r.encrypt_time_std or 0) ** 2
                            + (r.decrypt_time_std or 0) ** 2) * 1e3)
    a = df[(df.scheme == "AES-256-GCM") & (df.dataset_size == size)]
    if not a.empty:
        r = a.iloc[0]
        labels.append("AES\n(whole dataset)")
        vals.append((r.encrypt_time_mean + r.decrypt_time_mean) * 1e3)
        errs.append(np.sqrt((r.encrypt_time_std or 0) ** 2
                            + (r.decrypt_time_std or 0) ** 2) * 1e3)
    rr = df[(df.scheme == "RSA-2048-OAEP") & (df.dataset_size == size)]
    if not rr.empty:
        r = rr.iloc[0]
        labels.append("RSA\n(one record only)")
        vals.append((r.encrypt_time_mean + r.decrypt_time_mean) * 1e3)
        errs.append(np.sqrt((r.encrypt_time_std or 0) ** 2
                            + (r.decrypt_time_std or 0) ** 2) * 1e3)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, yerr=errs, capsize=4,
                  color=["#4C72B0", "#55A868", "#C44E52"])
    ax.bar_label(bars, fmt="%.2f ms", padding=3, fontsize=9)
    ax.set_ylabel("encrypt + decrypt time (ms)")
    ax.set_title(f"Data-protection cost: encrypt + decrypt the dataset (N={size:,})\n"
                 "AES/RSA cannot compute on ciphertext; RSA bar = one 8-byte "
                 "record", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "protection_cost.png")


def plot_packed_vs_elementwise(df):
    ck = df[df.scheme == "CKKS"]
    sizes = sorted(set(ck[ck.granularity == "packed"].dataset_size) &
                   set(ck[ck.granularity == "elementwise"].dataset_size))
    if not sizes:
        return
    size = sizes[-1]
    g = ck[ck.dataset_size == size]
    ops = sorted(g.operation.unique())
    import numpy as np
    x = np.arange(len(ops))
    width = 0.38
    packed = [g[(g.operation == o) & (g.granularity == "packed")].total_time_mean.mean() * 1e3 for o in ops]
    elem = [g[(g.operation == o) & (g.granularity == "elementwise")].total_time_mean.mean() * 1e3 for o in ops]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, packed, width, label="packed")
    ax.bar(x + width / 2, elem, width, label="element-wise")
    ax.set_xticks(x)
    ax.set_xticklabels(ops)
    ax.set_ylabel("total time (ms)")
    ax.set_yscale("log")
    ax.set_title(f"CKKS packing strategy: packed vs element-wise (N={size})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "packed_vs_elementwise.png")


def workflow_table(df, op="sum"):
    """Per-size cost of computing `op` over encrypted data, under two explicit scenarios.

    Both sides start from data stored encrypted at rest under their own scheme:
      - Traditional (AES): DECRYPT the dataset, compute in plaintext (data exposed to
        the compute party), re-encrypt the small result (negligible).
        aes_total_ms = decrypt + plaintext compute.
      - HE steady-state:   compute on the ciphertext, decrypt only the result.
        he_steady_ms = compute + result decrypt. This is the symmetric comparison.
      - HE one-shot:       additionally pays the CKKS encryption because the data was
        not yet CKKS-encrypted (e.g. first upload to an untrusted cloud).
        he_total_ms = encrypt + compute + result decrypt.
    """
    import numpy as np
    rows = []
    for size in sorted(df.dataset_size.unique()):
        aes = df[(df.scheme == "AES-256-GCM") & (df.dataset_size == size)]
        pt = df[(df.scheme == "plaintext") & (df.operation == op) & (df.dataset_size == size)]
        he = df[(df.scheme == "CKKS") & (df.granularity == "packed")
                & (df.operation == op) & (df.dataset_size == size)]
        if aes.empty or pt.empty or he.empty:
            continue
        rows.append({
            "dataset_size": size,
            "aes_decrypt_ms": aes.iloc[0].decrypt_time_mean * 1e3,
            "plaintext_compute_ms": pt.iloc[0].compute_time_mean * 1e3,
            "he_encrypt_ms": he.iloc[0].encrypt_time_mean * 1e3,
            "he_compute_ms": he.iloc[0].compute_time_mean * 1e3,
            "he_decrypt_ms": he.iloc[0].decrypt_time_mean * 1e3,
        })
    d = pd.DataFrame(rows)
    if not d.empty:
        d["aes_total_ms"] = d.aes_decrypt_ms + d.plaintext_compute_ms
        d["he_steady_ms"] = d.he_compute_ms + d.he_decrypt_ms
        d["he_total_ms"] = d.he_encrypt_ms + d.he_compute_ms + d.he_decrypt_ms
        d["plaintext_exposed_aes"] = True
        d["plaintext_exposed_he"] = False
    return d


def plot_workflow_comparison(df, op="sum"):
    import numpy as np
    d = workflow_table(df, op=op)
    if d.empty:
        return
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    d.to_csv(os.path.join(config.RESULTS_DIR, "workflow_comparison.csv"), index=False)

    x = np.arange(len(d))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(x - w, d.aes_total_ms, w,
           label="AES: decrypt + compute (plaintext EXPOSED)", color="#C44E52")
    ax.bar(x, d.he_steady_ms, w,
           label="HE steady-state: compute + result decrypt (NEVER exposed)",
           color="#4C72B0")
    ax.bar(x + w, d.he_total_ms, w,
           label="HE one-shot: + CKKS encryption of the dataset", color="#9BB8DC")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s):,}" for s in d.dataset_size])
    ax.set_xlabel("dataset size (records)")
    ax.set_ylabel(f"time to compute '{op}' (ms, log scale)")
    ax.set_title("Analytics on encrypted data: traditional (AES) vs HE\n"
                 "both sides start from data encrypted at rest", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "workflow_comparison.png")


def plot_error_vs_depth():
    """CKKS error vs multiplicative depth, from experiments/depth_sweep.py output."""
    import json
    path = os.path.join(config.RESULTS_DIR, "depth_sweep.json")
    if not os.path.exists(path):
        print("  skip error_vs_depth: no results/depth_sweep.json "
              "(run experiments/depth_sweep.py)")
        return
    with open(path) as f:
        sweep = json.load(f)
    levels = sweep.get("levels", [])
    ok = [lv for lv in levels if lv.get("ok")]
    failed = next((lv for lv in levels if not lv.get("ok")), None)
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [lv["depth"] for lv in ok]
    ys = [lv["mean_rel_error"] for lv in ok]
    bars = ax.bar([str(lv["depth"]) + f"\n({lv['expr']})" for lv in ok], ys,
                  color="#4C72B0", width=0.5)
    ax.bar_label(bars, labels=[f"{y:.1e}" for y in ys], padding=3, fontsize=9)
    if failed:
        # Draw the wall: a hatched marker where the chain runs out.
        ax.bar([str(failed["depth"]) + f"\n({failed['expr']})"], [max(ys) * 3],
               color="none", edgecolor="#C44E52", hatch="//", width=0.5,
               label=f"depth {failed['depth']}: FAILS — modulus chain exhausted")
        ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("sequential ciphertext multiplications (depth)")
    ax.set_ylabel("mean relative error (log scale)")
    chain = sweep.get("ckks", {}).get("coeff_mod_bit_sizes")
    ax.set_title(f"CKKS error grows with multiplicative depth (chain {chain})\n"
                 "repeated ct×ct squaring of values in [1, 2)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "ckks_error_vs_depth.png")


def main():
    if not os.path.exists(config.RESULTS_CSV):
        print(f"No results CSV at {config.RESULTS_CSV}; run the benchmark first.")
        sys.exit(1)
    try:
        df = pd.read_csv(config.RESULTS_CSV)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        print(f"{config.RESULTS_CSV} is empty or unreadable; run the benchmark first.")
        sys.exit(1)
    if df.empty:
        print(f"{config.RESULTS_CSV} has no rows; run the benchmark first.")
        sys.exit(1)
    print(f"Loaded {len(df)} rows from {config.RESULTS_CSV}")
    plot_runtime_vs_size(df)
    plot_cost_breakdown(df)
    plot_ciphertext_size(df)
    plot_error_vs_size(df)
    plot_protection_cost(df)
    plot_packed_vs_elementwise(df)
    plot_workflow_comparison(df)
    plot_error_vs_depth()
    print("Done.")


if __name__ == "__main__":
    main()
