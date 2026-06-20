"""Generate figures from results/results.csv.

Runs headless (Agg backend). Each plot is guarded so a missing slice of data does
not abort the whole script. Produces:
  1. ckks_runtime_vs_size.png    - CKKS total time vs dataset size (packed), per op.
  2. ckks_cost_breakdown.png     - encrypt/compute/decrypt split (packed, largest size).
  3. ciphertext_size_vs_size.png - encrypted size vs dataset size (CKKS vs AES vs plaintext).
  4. ckks_error_vs_size.png      - CKKS mean relative error vs dataset size, per op.
  5. protection_cost.png         - encrypt+decrypt cost: CKKS vs AES vs RSA(per-block).
  6. packed_vs_elementwise.png   - CKKS total time, packed vs element-wise (where both exist).
  7. memory_vs_size.png          - peak process memory (RSS) vs dataset size, CKKS vs AES.
  8. workflow_comparison.png     - analytics cost: AES (decrypt+compute) vs HE (on ciphertext).
                                   Also writes results/workflow_comparison.csv.
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


def plot_runtime_vs_size(df):
    ck = df[(df.scheme == "CKKS") & (df.granularity == "packed")]
    if ck.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for op, g in ck.groupby("operation"):
        g = g.sort_values("dataset_size")
        ax.plot(g.dataset_size, g.total_time_mean * 1e3, marker="o", label=op)
    ax.set_xlabel("dataset size (records)")
    ax.set_ylabel("total time (ms)  [encrypt + compute + decrypt]")
    ax.set_title("CKKS runtime vs dataset size (packed)")
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
    ax.set_xlabel("dataset size (records)")
    ax.set_ylabel("encrypted/raw size (KB)")
    ax.set_title("Data size: encrypted vs raw")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "ciphertext_size_vs_size.png")


def plot_error_vs_size(df):
    ck = df[(df.scheme == "CKKS") & (df.granularity == "packed")]
    if ck.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for op, g in ck.groupby("operation"):
        g = g.sort_values("dataset_size")
        ax.plot(g.dataset_size, g.mean_rel_error, marker="o", label=op)
    ax.set_xlabel("dataset size (records)")
    ax.set_ylabel("mean relative error")
    ax.set_yscale("log")
    ax.set_title("CKKS approximation error vs dataset size (packed)")
    ax.legend(title="operation")
    ax.grid(True, alpha=0.3)
    _save(fig, "ckks_error_vs_size.png")


def plot_protection_cost(df):
    """Encrypt+decrypt cost at the largest common size: CKKS(sum) vs AES vs RSA."""
    # Largest size where all three schemes are present, so no scheme is silently dropped.
    ck_sum = df[(df.scheme == "CKKS") & (df.granularity == "packed") & (df.operation == "sum")]
    common = (set(ck_sum.dataset_size)
              & set(df[df.scheme == "AES-256-GCM"].dataset_size)
              & set(df[df.scheme == "RSA-2048-OAEP"].dataset_size))
    if not common:
        print("  skip protection_cost: no size has CKKS+AES+RSA together")
        return
    size = max(common)
    labels, vals = [], []
    ck = df[(df.scheme == "CKKS") & (df.granularity == "packed") &
            (df.operation == "sum") & (df.dataset_size == size)]
    if not ck.empty:
        r = ck.iloc[0]
        labels.append("CKKS (enc+dec)")
        vals.append((r.encrypt_time_mean + r.decrypt_time_mean) * 1e3)
    a = df[(df.scheme == "AES-256-GCM") & (df.dataset_size == size)]
    if not a.empty:
        r = a.iloc[0]
        labels.append("AES (whole dataset)")
        vals.append((r.encrypt_time_mean + r.decrypt_time_mean) * 1e3)
    rr = df[(df.scheme == "RSA-2048-OAEP") & (df.dataset_size == size)]
    if not rr.empty:
        r = rr.iloc[0]
        labels.append("RSA (per record)")
        vals.append((r.encrypt_time_mean + r.decrypt_time_mean) * 1e3)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("encrypt + decrypt time (ms)")
    ax.set_title(f"Data-protection cost (N={size})\n"
                 "note: AES/RSA cannot compute on ciphertext; RSA shown per single record")
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


def plot_memory_vs_size(df):
    """Peak process memory (RSS delta) vs dataset size: CKKS ops vs AES reference."""
    ck = df[(df.scheme == "CKKS") & (df.granularity == "packed")]
    if ck.empty or ck.peak_memory_mb.dropna().empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for op, g in ck.groupby("operation"):
        g = g.sort_values("dataset_size")
        ax.plot(g.dataset_size, g.peak_memory_mb, marker="o", label=f"CKKS {op}")
    a = df[df.scheme == "AES-256-GCM"].sort_values("dataset_size")
    if not a.empty and not a.peak_memory_mb.dropna().empty:
        ax.plot(a.dataset_size, a.peak_memory_mb, marker="s", linestyle="--",
                color="black", label="AES (reference)")
    ax.set_xlabel("dataset size (records)")
    ax.set_ylabel("peak memory delta (MB)")
    ax.set_title("Peak process memory vs dataset size (RSS)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, "memory_vs_size.png")


def workflow_table(df, op="sum"):
    """Per-size cost of computing `op` over encrypted data, two workflows.

    Traditional (AES): the data is stored encrypted, so to compute you must DECRYPT the
    dataset, compute in plaintext (data exposed to the compute party), then re-encrypt the
    small result (negligible). HE (CKKS): encrypt -> compute on ciphertext (never exposed)
    -> decrypt only the result.
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
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(x - w / 2, d.aes_total_ms, w,
           label="AES: decrypt + compute (plaintext EXPOSED)", color="#C44E52")
    ax.bar(x + w / 2, d.he_total_ms, w,
           label="HE: encrypt + compute + decrypt (NEVER exposed)", color="#4C72B0")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(d.dataset_size)
    ax.set_xlabel("dataset size (records)")
    ax.set_ylabel(f"time to compute '{op}' (ms, log scale)")
    ax.set_title("Analytics on encrypted data: traditional (AES) vs HE\n"
                 "HE costs more, but the data is never decrypted during computation")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "workflow_comparison.png")


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
    plot_memory_vs_size(df)
    plot_workflow_comparison(df)
    print("Done.")


if __name__ == "__main__":
    main()
