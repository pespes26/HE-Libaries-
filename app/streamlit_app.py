"""Streamlit UI for the HE benchmark.

Three tabs:
  1. Dashboard          - interactive charts/table from results/results.csv (+ run config).
  2. Live HE demo       - run CKKS encrypt -> compute -> decrypt for chosen params, verify.
  3. Confidential use case - aggregate private values under encryption (the "why HE" story).

Run locally:  streamlit run app/streamlit_app.py
In Docker:    docker compose up dashboard   ->   http://localhost:8501

The app reuses the exact same modules as the benchmark (he_ckks, reference, data,
baseline_aes), so what you see here matches the measured pipeline.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

# Make the project root importable when Streamlit runs this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from he_benchmark import baseline_aes as aes  # noqa: E402
from he_benchmark import data as data_mod  # noqa: E402
from he_benchmark import he_ckks  # noqa: E402
from he_benchmark import reference as ref  # noqa: E402
from he_benchmark import security  # noqa: E402

st.set_page_config(page_title="Benchmarking Homomorphic Encryption", layout="wide")

# --- Bar's Design Signature (dark) + Emil Kowalski's motion craft.
#     Signature = Ubuntu fonts, pill buttons, mono technical labels, border-based depth.
#     Motion    = crisp & minimal: custom ease-out, <200ms, transform/opacity only, press
#                 feedback, focus rings, a subtle opacity-led entrance, reduced-motion safe.
#     Base palette comes from .streamlit/config.toml. ---
_DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700&family=Ubuntu+Mono:wght@400;700&display=swap');

:root {
  --bg:#0C1412; --s1:#131E1C; --s2:#1A2826; --b1:#1E2E2C; --b2:#2a3e3c;
  --tp:#e8e2da; --ts:#7a8f8d; --tm:#3d5452;
  --accent:#00857F; --accent-bd:rgba(0,133,127,0.40);
  /* Motion tokens (Emil): strong custom ease-out; never ease-in; fast durations */
  --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
  --ease: cubic-bezier(0.4, 0, 0.2, 1);
  --d-fast: 120ms; --d: 160ms;
}

/* Base font: Ubuntu everywhere */
html, body, .stApp, [data-testid="stAppViewContainer"], [class*="css"] {
  font-family:'Ubuntu', ui-sans-serif, system-ui, sans-serif;
}
h1,h2,h3,h4 { font-family:'Ubuntu', sans-serif; letter-spacing:-0.01em; font-weight:500; }
h1 { font-weight:700; }
/* Section hierarchy: a quiet divider under headers */
h2, h3 { padding-bottom:6px; border-bottom:1px solid var(--b1); }

/* Technical content -> Ubuntu Mono */
code, kbd, pre, samp { font-family:'Ubuntu Mono', ui-monospace, monospace !important; }

/* Captions read as technical labels: mono, muted */
[data-testid="stCaptionContainer"] { font-family:'Ubuntu Mono', monospace; color:var(--ts); }

/* Metric: card with border depth; label is uppercase tracked mono */
[data-testid="stMetric"] {
  background:var(--s1); border:1px solid var(--b1); border-radius:10px; padding:12px 16px;
  transition: border-color var(--d) var(--ease), transform var(--d) var(--ease-out);
}
[data-testid="stMetricLabel"] {
  font-family:'Ubuntu Mono', monospace !important;
  text-transform:uppercase; letter-spacing:0.12em; color:var(--ts) !important;
}
[data-testid="stMetricValue"] { font-weight:500; letter-spacing:-0.02em; }

/* Buttons -> pill + press feedback. Specific properties only (no `all`), custom ease-out. */
.stButton > button, .stDownloadButton > button {
  border-radius:9999px !important; font-weight:500;
  transition: transform var(--d) var(--ease-out), border-color var(--d) var(--ease),
              background-color var(--d) var(--ease), opacity var(--d) var(--ease);
}
.stButton > button:active, .stDownloadButton > button:active { transform: scale(0.97); }

/* Focus-visible accent ring (feedback + accessibility) */
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
input:focus-visible, textarea:focus-visible, [data-baseweb="tab"]:focus-visible {
  outline:none; box-shadow: 0 0 0 3px rgba(0,133,127,0.20);
}

/* Tabs: smooth colour, accent on the selected tab */
[data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--b1); }
[data-baseweb="tab"] { font-family:'Ubuntu', sans-serif; transition: color var(--d) var(--ease); }
button[aria-selected="true"][data-baseweb="tab"] { color:var(--accent) !important; }

/* Dataframes / tables: border + radius (depth via borders, not shadow) */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border:1px solid var(--b1); border-radius:10px; overflow:hidden;
}

/* Expander as a card */
[data-testid="stExpander"] {
  border:1px solid var(--b1); border-radius:10px; background:var(--s1);
  transition: border-color var(--d) var(--ease);
}

/* Inputs: 8px radius */
input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] { border-radius:8px !important; }

/* Alerts and links */
[data-testid="stAlert"] { border-radius:10px; }
a, a:visited { color:var(--accent); transition: opacity var(--d-fast) var(--ease); }
a:hover { opacity:0.8; }

/* Spinner tinted to accent (perceived speed) */
[data-testid="stSpinner"] svg { color: var(--accent) !important; }

/* Subtle, opacity-led page entrance. Short so Streamlit reruns stay calm
   (Emil: do not animate frequently-seen things). */
@keyframes fadeUp { from { opacity:0; transform: translateY(4px); } to { opacity:1; transform:none; } }
.block-container { animation: fadeUp var(--d) var(--ease-out); }

/* Hover affordances only on real hover-capable pointers (no touch false-positives) */
@media (hover: hover) and (pointer: fine) {
  .stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--accent-bd) !important; }
  [data-testid="stMetric"]:hover { border-color:var(--b2); transform: translateY(-1px); }
  [data-testid="stExpander"]:hover { border-color:var(--b2); }
}

/* Respect reduced motion: keep colour/opacity, drop all movement */
@media (prefers-reduced-motion: reduce) {
  .block-container { animation:none !important; }
  .stButton > button:active, .stDownloadButton > button:active,
  [data-testid="stMetric"]:hover { transform:none !important; }
}
</style>
"""
st.markdown(_DESIGN_CSS, unsafe_allow_html=True)

# Operation metadata shared by the live tab.
OPS = {
    "add (element-wise)": {"key": "add", "two": True, "scalar": False},
    "multiply (element-wise)": {"key": "mul", "two": True, "scalar": False},
    "sum": {"key": "sum", "two": False, "scalar": True},
    "average (mean)": {"key": "mean", "two": False, "scalar": True},
}


# --- cached resources ------------------------------------------------------------

@st.cache_resource(show_spinner="Building CKKS context and keys (one-time)...")
def get_context():
    """The CKKS context + keys are expensive (~34 MB of Galois keys); build once."""
    return he_ckks.make_context()


@st.cache_data
def load_results():
    if not os.path.exists(config.RESULTS_CSV):
        return None
    try:
        return pd.read_csv(config.RESULTS_CSV)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        # Truncated/empty CSV (e.g. an interrupted write) -> treat as "no results".
        return None


@st.cache_data
def load_run_config():
    if not os.path.exists(config.RUN_CONFIG_JSON):
        return None
    with open(config.RUN_CONFIG_JSON) as f:
        return json.load(f)


# --- helpers ---------------------------------------------------------------------

def encrypt_fn(ctx, arr, granularity):
    if granularity == "packed":
        return he_ckks.encrypt_packed(ctx, arr)
    return he_ckks.encrypt_elementwise(ctx, arr)


def compute_he(op, enc_a, enc_b, n):
    return {
        "add": lambda: he_ckks.he_add(enc_a, enc_b),
        "mul": lambda: he_ckks.he_mul(enc_a, enc_b),
        "sum": lambda: he_ckks.he_sum(enc_a),
        "mean": lambda: he_ckks.he_mean(enc_a, n),
    }[op]()


def compute_ref(op, a, b):
    return {
        "add": lambda: ref.ref_add(a, b),
        "mul": lambda: ref.ref_mul(a, b),
        "sum": lambda: ref.ref_sum(a),
        "mean": lambda: ref.ref_mean(a),
    }[op]()


def hex_preview(raw: bytes, n: int = 48) -> str:
    return raw[:n].hex(" ") + (" ..." if len(raw) > n else "")


def build_workflow_table(df, op="sum"):
    """Cost of computing `op` over encrypted data: AES (must decrypt) vs HE (on ciphertext)."""
    rows = []
    for size in sorted(df.dataset_size.unique()):
        aes_r = df[(df.scheme == "AES-256-GCM") & (df.dataset_size == size)]
        pt = df[(df.scheme == "plaintext") & (df.operation == op) & (df.dataset_size == size)]
        he = df[(df.scheme == "CKKS") & (df.granularity == "packed")
                & (df.operation == op) & (df.dataset_size == size)]
        if aes_r.empty or pt.empty or he.empty:
            continue
        aes_total = (aes_r.iloc[0].decrypt_time_mean + pt.iloc[0].compute_time_mean) * 1e3
        he_total = (he.iloc[0].encrypt_time_mean + he.iloc[0].compute_time_mean
                    + he.iloc[0].decrypt_time_mean) * 1e3
        rows.append({
            "dataset_size": int(size),
            "AES decrypt+compute (ms)": round(aes_total, 3),
            "HE enc+compute+dec (ms)": round(he_total, 3),
            "AES exposes plaintext": True,
            "HE exposes plaintext": False,
        })
    return pd.DataFrame(rows)


# --- header ----------------------------------------------------------------------

st.title("Benchmarking Homomorphic Encryption")
st.caption("Bar Pesso, Shay Harush, Shon Platok — Confidential Computing")

with st.expander("How to read this (fair-comparison framing)", expanded=False):
    st.markdown(
        "- **CKKS (HE)** computes directly on ciphertext, then decrypts only the result. "
        "It is **approximate** — every result reports its error vs the plaintext value.\n"
        "- **AES-256 / RSA-2048 cannot compute on ciphertext.** They only protect data; "
        "to analyze it you must decrypt first. RSA-2048 also cannot encrypt bulk data "
        "(~190-byte limit), so it is measured per single record.\n"
        "- Costs are separated into **data protection** (encrypt/decrypt), **computation**, "
        "and **size overhead** so no scheme is compared unfairly."
    )

tab_dash, tab_live, tab_case, tab_sec = st.tabs(
    ["Dashboard", "Live HE demo", "Confidential use case", "Security & HE vs traditional"]
)


# =================================================================================
# Tab 1 — Dashboard
# =================================================================================
with tab_dash:
    df = load_results()
    rc = load_run_config()

    if df is None:
        st.warning(
            "No results found. Run the benchmark first:\n\n"
            "`docker compose run --rm benchmark`"
        )
    else:
        if rc:
            # Use .get with fallbacks so a stale/partial run_config.json never crashes.
            ckks = rc.get("ckks", {})
            ctx_bytes = rc.get("ckks_context_size_bytes")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CKKS poly_modulus_degree", ckks.get("poly_modulus_degree", "—"))
            c2.metric("CKKS slots / ciphertext", ckks.get("slots", "—"))
            c3.metric("Key/context size",
                      f"{ctx_bytes/1024/1024:.1f} MB" if ctx_bytes else "—")
            c4.metric("Repeats per measurement", rc.get("n_repeats", "—"))

        st.subheader("CKKS runtime vs dataset size (packed)")
        ck = df[(df.scheme == "CKKS") & (df.granularity == "packed")]
        if not ck.empty:
            piv = ck.pivot_table(index="dataset_size", columns="operation",
                                 values="total_time_mean") * 1e3
            st.line_chart(piv)
            st.caption("Total time = encrypt + compute + decrypt, in milliseconds.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Packing strategy: packed vs element-wise")
            both = sorted(set(df[(df.scheme == "CKKS") & (df.granularity == "packed")].dataset_size)
                          & set(df[(df.scheme == "CKKS") & (df.granularity == "elementwise")].dataset_size))
            if both:
                size = both[-1]
                g = df[(df.scheme == "CKKS") & (df.dataset_size == size)]
                pe = g.pivot_table(index="operation", columns="granularity",
                                   values="total_time_mean") * 1e3
                st.bar_chart(pe)
                st.caption(f"Total time (ms) at N={size}. Element-wise = one ciphertext "
                           "per value; packed = batched. Note the scale difference.")
            else:
                st.info("No size has both granularities to compare.")
        with col_b:
            st.subheader("CKKS approximation error (packed)")
            if not ck.empty:
                perr = ck.pivot_table(index="dataset_size", columns="operation",
                                      values="mean_rel_error")
                st.line_chart(perr)
                st.caption("Mean relative error vs plaintext. Small but non-zero — CKKS "
                           "is an approximate scheme.")

        st.subheader("Data-protection cost vs computation")
        st.caption("AES/RSA columns are protection only — they cannot compute on ciphertext.")
        prot = df[df.scheme.isin(["CKKS", "AES-256-GCM", "RSA-2048-OAEP"])].copy()
        # Pick the largest size at which ALL three schemes have data, so the chart never
        # silently drops a scheme (e.g. if a long 100k run was interrupted).
        ck_sum = df[(df.scheme == "CKKS") & (df.granularity == "packed") & (df.operation == "sum")]
        common = (set(ck_sum.dataset_size)
                  & set(df[df.scheme == "AES-256-GCM"].dataset_size)
                  & set(df[df.scheme == "RSA-2048-OAEP"].dataset_size))
        if not prot.empty and common:
            size = max(common)
            rows = []
            cks = prot[(prot.scheme == "CKKS") & (prot.granularity == "packed")
                       & (prot.operation == "sum") & (prot.dataset_size == size)]
            if not cks.empty:
                r = cks.iloc[0]
                rows.append(("CKKS protect (enc+dec)",
                             (r.encrypt_time_mean + r.decrypt_time_mean) * 1e3))
                rows.append(("CKKS compute (sum)", r.compute_time_mean * 1e3))
            a = prot[(prot.scheme == "AES-256-GCM") & (prot.dataset_size == size)]
            if not a.empty:
                r = a.iloc[0]
                rows.append(("AES protect (whole set)",
                             (r.encrypt_time_mean + r.decrypt_time_mean) * 1e3))
            rr = prot[(prot.scheme == "RSA-2048-OAEP") & (prot.dataset_size == size)]
            if not rr.empty:
                r = rr.iloc[0]
                rows.append(("RSA protect (per record)",
                             (r.encrypt_time_mean + r.decrypt_time_mean) * 1e3))
            if rows:
                bar = pd.DataFrame(rows, columns=["step", "ms"]).set_index("step")
                st.bar_chart(bar)
                st.caption(f"Milliseconds at N={size}.")

        st.subheader("Peak memory vs dataset size (RSS)")
        if not ck.empty and not ck["peak_memory_mb"].dropna().empty:
            pmem = ck.pivot_table(index="dataset_size", columns="operation",
                                  values="peak_memory_mb")
            st.line_chart(pmem)
            aes_mem = df[df.scheme == "AES-256-GCM"]["peak_memory_mb"].dropna()
            ref_txt = (f" For reference, AES peaks at ~{aes_mem.max():.1f} MB."
                       if not aes_mem.empty else "")
            st.caption("Peak process memory delta (MB) for CKKS packed." + ref_txt)

        st.subheader("All results")
        schemes = st.multiselect("Filter scheme", sorted(df.scheme.unique()),
                                 default=sorted(df.scheme.unique()))
        view = df[df.scheme.isin(schemes)] if schemes else df
        show_cols = ["scheme", "operation", "dataset_size", "granularity",
                     "encrypt_time_mean", "compute_time_mean", "decrypt_time_mean",
                     "ciphertext_size_bytes", "mean_rel_error", "correct", "notes"]
        st.dataframe(view[show_cols], use_container_width=True, hide_index=True)

        with st.expander("Static figures (from analysis/make_plots.py)"):
            figs = ["ckks_runtime_vs_size.png", "ckks_cost_breakdown.png",
                    "ciphertext_size_vs_size.png", "ckks_error_vs_size.png",
                    "protection_cost.png", "packed_vs_elementwise.png",
                    "memory_vs_size.png", "workflow_comparison.png"]
            cols = st.columns(2)
            shown = False
            for i, fig in enumerate(figs):
                p = os.path.join(config.FIGURES_DIR, fig)
                if os.path.exists(p):
                    cols[i % 2].image(p, caption=fig, use_column_width=True)
                    shown = True
            if not shown:
                st.info("No figures yet. Run `docker compose run --rm plots`.")


# =================================================================================
# Tab 2 — Live HE demo
# =================================================================================
with tab_live:
    st.subheader("Run CKKS on the fly")
    st.caption("Encrypt synthetic data, compute on the ciphertext, decrypt only the "
               "result, and check it against the plaintext reference.")

    c1, c2, c3 = st.columns(3)
    op_label = c1.selectbox("Operation", list(OPS.keys()), index=2)
    gran = c2.radio("Granularity", ["packed", "elementwise"], horizontal=True)
    seed = c3.number_input("Seed", min_value=0, max_value=10_000, value=config.RANDOM_SEED)

    max_size = config.ELEMENTWISE_MAX_SIZE if gran == "elementwise" else 5_000
    size = st.slider("Dataset size (records)", min_value=10, max_value=max_size,
                     value=min(1_000, max_size), step=10)
    if gran == "elementwise":
        st.caption(f"Element-wise encrypts one ciphertext per record and is O(N) in time "
                   f"and memory, so it is capped at {config.ELEMENTWISE_MAX_SIZE} records here.")

    if st.button("Run encrypted computation", type="primary"):
        spec = OPS[op_label]
        op = spec["key"]
        ctx = get_context()
        a = data_mod.generate_synthetic(size, int(seed), config.DATA_LOW, config.DATA_HIGH)
        b = data_mod.generate_synthetic(size, int(seed) + 1, config.DATA_LOW, config.DATA_HIGH)

        with st.spinner("Encrypting, computing, decrypting..."):
            t0 = time.perf_counter()
            enc_a = encrypt_fn(ctx, a, gran)
            enc_b = encrypt_fn(ctx, b, gran) if spec["two"] else None
            enc_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            result = compute_he(op, enc_a, enc_b, size)
            comp_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            decrypted = (he_ckks.decrypt_scalar(result) if spec["scalar"]
                         else he_ckks.decrypt_vectors(result))
            dec_t = time.perf_counter() - t0

        reference = compute_ref(op, a, b)
        ct_size = he_ckks.ciphertext_size_bytes(enc_a)
        n_ct = len(enc_a)

        ref_arr = np.atleast_1d(np.asarray(reference, dtype=np.float64))
        got_arr = np.atleast_1d(np.asarray(decrypted, dtype=np.float64))
        max_abs = float(np.abs(got_arr - ref_arr).max())
        mean_rel = float(np.mean(np.abs(got_arr - ref_arr) / np.maximum(np.abs(ref_arr), 1e-12)))
        correct = bool(np.allclose(got_arr, ref_arr, rtol=config.CKKS_REL_TOL,
                                   atol=config.CKKS_ABS_TOL))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Encrypt", f"{enc_t*1e3:.1f} ms")
        m2.metric("Compute (on ciphertext)", f"{comp_t*1e3:.1f} ms")
        m3.metric("Decrypt", f"{dec_t*1e3:.1f} ms")
        m4.metric("Ciphertext size", f"{ct_size/1024:.1f} KB", help=f"{n_ct} ciphertext(s)")

        if correct:
            st.success(f"Decrypted result matches plaintext within tolerance "
                       f"(mean rel. error {mean_rel:.2e}, max abs {max_abs:.2e}).")
        else:
            st.error(f"Result outside tolerance (mean rel. error {mean_rel:.2e}). "
                     "CKKS noise budget may be exhausted for this operation.")

        if spec["scalar"]:
            st.write(pd.DataFrame({
                "value": ["plaintext reference", "decrypted (HE)"],
                "result": [float(reference), float(decrypted)],
            }))
        else:
            preview = pd.DataFrame({
                "index": np.arange(min(10, size)),
                "plaintext": ref_arr[:10],
                "decrypted (HE)": got_arr[:10],
            })
            st.caption("First 10 elements (errors computed over the full vector):")
            st.dataframe(preview, hide_index=True, use_container_width=True)


# =================================================================================
# Tab 3 — Confidential use case
# =================================================================================
with tab_case:
    st.subheader("Confidential analytics: average without revealing individuals")
    st.markdown(
        "**Scenario.** A team wants the *average* of private salaries, but no one — not "
        "even the analytics server — should see any individual salary. With HE, each "
        "value is encrypted on the client; the server computes the average **on the "
        "ciphertexts** and returns an encrypted result that only the client can decrypt."
    )

    default = "3200, 4100, 2750, 5300, 3900, 4600, 3100, 4800"
    MAX_VALUES = config.ELEMENTWISE_MAX_SIZE  # each value -> its own ciphertext (O(N) memory)
    raw = st.text_area("Private values (comma-separated)", value=default, height=80)
    try:
        values = np.array([float(x.strip()) for x in raw.split(",") if x.strip() != ""],
                          dtype=np.float64)
    except ValueError:
        st.error("Could not parse the values. Use comma-separated numbers.")
        values = np.array([], dtype=np.float64)
    # Guard the demo: reject non-finite values and cap the count so a large paste
    # cannot exhaust memory (every value is encrypted as a separate ciphertext).
    if values.size and not np.all(np.isfinite(values)):
        st.warning("Dropped non-finite values (inf/nan).")
        values = values[np.isfinite(values)]
    if values.size > MAX_VALUES:
        st.warning(f"Using the first {MAX_VALUES} values (demo cap).")
        values = values[:MAX_VALUES]

    if st.button("Encrypt and compute the average homomorphically", type="primary") \
            and values.size > 0:
        ctx = get_context()
        # Encrypt each value separately so we can show the server only sees ciphertext.
        enc = he_ckks.encrypt_elementwise(ctx, values)

        st.markdown("**What the server stores (ciphertext, not the values):**")
        st.code("\n".join(
            f"record {i}: {hex_preview(enc[i].serialize())}"
            for i in range(min(4, len(enc)))
        ) + ("\n..." if len(enc) > 4 else ""), language="text")

        # Compute the mean entirely on ciphertext, then decrypt only the aggregate.
        enc_mean = he_ckks.he_mean(enc, len(values))
        he_avg = he_ckks.decrypt_scalar(enc_mean)
        true_avg = float(np.mean(values))

        c1, c2, c3 = st.columns(3)
        c1.metric("Records (kept private)", len(values))
        c2.metric("Average (decrypted result)", f"{he_avg:,.2f}")
        c3.metric("Error vs plaintext", f"{abs(he_avg - true_avg):.2e}")

        st.success(
            "The server computed the average on encrypted data. Individual values were "
            "never decrypted server-side — only the final aggregate was revealed to the "
            f"client. (Plaintext check: true average = {true_avg:,.2f}.)"
        )
        st.caption("This is the capability AES and RSA do not provide: useful computation "
                   "while the data stays encrypted.")


# =================================================================================
# Tab 4 — Security & HE vs traditional
# =================================================================================
with tab_sec:
    st.subheader("Security comparison of the schemes")
    profiles = security.security_profiles()
    sec_df = pd.DataFrame(profiles).set_index("scheme").T
    st.dataframe(sec_df, use_container_width=True)

    st.markdown("**Key takeaways**")
    for t in security.key_takeaways():
        st.markdown(f"- {t}")

    st.divider()
    st.subheader("Computing on encrypted data: traditional (AES) vs HE")
    st.markdown(
        "To analyze **AES**-protected data you must **decrypt it first** — the plaintext is "
        "exposed to whoever runs the computation. **HE** computes directly on the ciphertext, "
        "so the data is never exposed. The benchmark below shows HE costs more time, which is "
        "the price of that protection."
    )

    dfw = load_results()
    if dfw is None:
        st.warning("Run the benchmark to populate this comparison: "
                   "`docker compose run --rm benchmark`")
    else:
        op = st.selectbox("Aggregate operation", ["sum", "mean"], key="wf_op")
        wt = build_workflow_table(dfw, op)
        if wt.empty:
            st.info("Not enough data for this comparison.")
        else:
            st.dataframe(wt, hide_index=True, use_container_width=True)
            p = os.path.join(config.FIGURES_DIR, "workflow_comparison.png")
            if os.path.exists(p):
                st.image(p, use_column_width=True)
            st.caption("AES exposes plaintext during computation; HE does not. HE's higher "
                       "cost is the measured overhead of protecting data in use.")
