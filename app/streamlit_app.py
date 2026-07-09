"""Streamlit UI for the HE benchmark.

Four tabs:
  1. Dashboard          - interactive charts/table from results/results.csv (+ run config).
  2. Live HE demo       - run CKKS encrypt -> compute -> decrypt for chosen params, verify.
  3. Confidential use case - aggregate private values under encryption (the "why HE" story).
  4. Security & HE vs traditional - qualitative comparison + the workflow-cost contrast.

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
import plotly.graph_objects as go
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

# --- Emerging-tech style (DARK / OLED) + the same coherent motion layer.
#     Look   = deep-slate canvas with a faint grid + emerald/cyan glow, Fira Code/Sans
#              type, electric-emerald accent, minimal glow, glassy elevated surfaces.
#     Motion = lively but coherent: staggered card/chart entrance, hover lift + glow,
#              press feedback, focus rings, animated tabs. transform/opacity only (GPU);
#              fully disabled under prefers-reduced-motion.
#     Base palette comes from .streamlit/config.toml. ---
_DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

:root {
  /* Dark / deep-slate emerging-tech palette */
  --bg:#0F172A; --s1:#1E293B; --s2:#172033; --s3:#0B1220; --b1:#2A3852; --b2:#3B4C6B;
  --tp:#F8FAFC; --ts:#94A3B8; --tm:#64748B;
  --accent:#34D399; --accent-2:#38BDF8;
  --accent-bd:rgba(52,211,153,0.55); --accent-soft:rgba(52,211,153,0.14);
  --glow:0 0 10px rgba(52,211,153,0.55);
  /* Motion tokens: strong custom ease-out; never ease-in */
  --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
  --ease: cubic-bezier(0.4, 0, 0.2, 1);
  --d-fast: 120ms; --d: 180ms; --d-slow: 340ms;
  --shadow: 0 2px 14px rgba(0,0,0,0.45);
  --shadow-lg: 0 10px 34px rgba(0,0,0,0.55);
  --shadow-glow: 0 8px 28px rgba(0,0,0,0.5), 0 0 0 1px var(--accent-bd), 0 0 22px rgba(52,211,153,0.20);
}

/* Emerging-tech backdrop: emerald + cyan glow washes over a faint technical grid */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 620px at 12% -12%, rgba(52,211,153,0.10), transparent 60%),
    radial-gradient(1000px 540px at 100% 0%, rgba(56,189,248,0.08), transparent 55%),
    linear-gradient(rgba(148,163,184,0.045) 1px, transparent 1px) 0 0 / 44px 44px,
    linear-gradient(90deg, rgba(148,163,184,0.045) 1px, transparent 1px) 0 0 / 44px 44px,
    var(--bg);
}
[data-testid="stHeader"] { background: transparent; }

/* Base font: Fira Sans everywhere */
html, body, .stApp, [data-testid="stAppViewContainer"], [class*="css"] {
  font-family:'Fira Sans', ui-sans-serif, system-ui, sans-serif;
}
h1,h2,h3,h4 { font-family:'Fira Sans', sans-serif; letter-spacing:-0.01em; font-weight:600; color:var(--tp); }
h1 { font-weight:700; text-shadow: 0 0 18px rgba(52,211,153,0.28); }
/* Section headers: glowing accent tick + quiet divider, animated in */
h2, h3 {
  padding:2px 0 6px 12px; border-bottom:1px solid var(--b1);
  border-left:3px solid var(--accent); margin-left:-12px; padding-left:12px;
  box-shadow: -3px 0 12px -4px var(--accent-bd);
  animation: slideIn var(--d-slow) var(--ease-out) both;
}

/* Technical content -> Fira Code */
code, kbd, pre, samp { font-family:'Fira Code', ui-monospace, monospace !important; }
code { color:var(--accent) !important; }

/* Captions read as technical labels: mono, muted */
[data-testid="stCaptionContainer"] { font-family:'Fira Code', monospace; color:var(--ts); }

/* --- Keyframes --- */
@keyframes fadeUp { from { opacity:0; transform: translateY(6px); } to { opacity:1; transform:none; } }
@keyframes riseIn { from { opacity:0; transform: translateY(12px); } to { opacity:1; transform:none; } }
@keyframes popIn  { from { opacity:0; transform: scale(0.96); }     to { opacity:1; transform: scale(1); } }
@keyframes slideIn{ from { opacity:0; transform: translateX(-8px); } to { opacity:1; transform:none; } }

/* Page + element entrances (livelier, still opacity-led) */
.block-container { animation: fadeUp var(--d) var(--ease-out); }
[data-testid="stMetric"] { animation: popIn var(--d-slow) var(--ease-out) both; }
[data-testid="element-container"]:has(.js-plotly-plot), [data-testid="stDataFrame"], [data-testid="stTable"],
[data-testid="stExpander"] { animation: riseIn var(--d-slow) var(--ease-out) both; }

/* Staggered entrance across columns in a row (e.g. the headline metric cards) */
[data-testid="stHorizontalBlock"] > [data-testid="column"] { animation: riseIn var(--d-slow) var(--ease-out) both; }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(1) { animation-delay: 0ms; }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) { animation-delay: 60ms; }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) { animation-delay: 120ms; }
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(4) { animation-delay: 180ms; }

/* Metric: glassy card with a top accent hairline; label uppercase tracked mono */
[data-testid="stMetric"] {
  position:relative; background:linear-gradient(180deg, var(--s1), var(--s2));
  border:1px solid var(--b1); border-radius:12px; padding:14px 18px; overflow:hidden;
  transition: transform var(--d) var(--ease-out), box-shadow var(--d) var(--ease),
              border-color var(--d) var(--ease);
}
[data-testid="stMetric"]::before {
  content:""; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg, var(--accent), var(--accent-2)); opacity:0.85;
}
[data-testid="stMetricLabel"] {
  font-family:'Fira Code', monospace !important;
  text-transform:uppercase; letter-spacing:0.12em; color:var(--ts) !important;
}
[data-testid="stMetricValue"] {
  font-family:'Fira Code', monospace; font-weight:600; letter-spacing:-0.01em;
  color:var(--accent); text-shadow: 0 0 14px rgba(52,211,153,0.30);
}

/* Charts sit on cards (layered depth) — style the WRAPPER, never the .js-plotly-plot
   div itself (that breaks Plotly's size calculation) */
[data-testid="element-container"]:has(.js-plotly-plot) {
  background:var(--s1); border:1px solid var(--b1); border-radius:14px; padding:8px 8px 4px;
  box-shadow: var(--shadow);
  transition: transform var(--d) var(--ease-out), box-shadow var(--d) var(--ease),
              border-color var(--d) var(--ease);
}

/* Buttons -> crisp emerging-tech chip + lively press/hover. transform/opacity only. */
.stButton > button, .stDownloadButton > button {
  border-radius:8px !important; font-family:'Fira Code', monospace; font-weight:500;
  letter-spacing:0.02em;
  transition: transform var(--d) var(--ease-out), box-shadow var(--d) var(--ease),
              border-color var(--d) var(--ease), background-color var(--d) var(--ease);
}
.stButton > button:active, .stDownloadButton > button:active { transform: scale(0.96); }
/* Primary CTA glows */
.stButton > button[kind="primary"] {
  box-shadow: 0 0 0 1px var(--accent-bd), 0 0 18px rgba(52,211,153,0.30);
}

/* Focus-visible accent ring */
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
input:focus-visible, textarea:focus-visible, [data-baseweb="tab"]:focus-visible {
  outline:none; box-shadow: 0 0 0 3px rgba(52,211,153,0.30);
}

/* Tabs: animated, glowing accent on the selected tab */
[data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--b1); }
[data-baseweb="tab"] {
  font-family:'Fira Code', monospace;
  transition: color var(--d) var(--ease), background-color var(--d) var(--ease),
              box-shadow var(--d) var(--ease), transform var(--d-fast) var(--ease-out);
  border-radius:8px 8px 0 0;
}
button[aria-selected="true"][data-baseweb="tab"] {
  color:var(--accent) !important; box-shadow: inset 0 -2px 0 0 var(--accent), 0 6px 16px -8px var(--accent-bd);
}

/* Dataframes / tables / expander as cards */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border:1px solid var(--b1); border-radius:10px; overflow:hidden;
}
[data-testid="stExpander"] {
  border:1px solid var(--b1); border-radius:10px; background:var(--s1);
  transition: box-shadow var(--d) var(--ease), border-color var(--d) var(--ease);
}

/* Inputs: 8px radius */
input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] { border-radius:8px !important; }

/* Alerts and links */
[data-testid="stAlert"] { border-radius:10px; animation: riseIn var(--d-slow) var(--ease-out) both; }
a, a:visited { color:var(--accent); transition: opacity var(--d-fast) var(--ease); }
a:hover { opacity:0.78; }

/* Spinner tinted to accent */
[data-testid="stSpinner"] svg { color: var(--accent) !important; }

/* Hover affordances only on real hover-capable pointers (no touch false-positives) */
@media (hover: hover) and (pointer: fine) {
  .stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px) scale(1.02); box-shadow: var(--shadow), 0 0 16px rgba(52,211,153,0.25);
    border-color:var(--accent-bd) !important;
  }
  [data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: var(--shadow-glow); border-color:var(--accent-bd); }
  [data-testid="element-container"]:has(.js-plotly-plot):hover { transform: translateY(-2px); box-shadow: var(--shadow-lg); border-color:var(--b2); }
  [data-testid="stExpander"]:hover { box-shadow: var(--shadow); border-color:var(--b2); }
  [data-baseweb="tab"]:hover { color:var(--accent) !important; transform: translateY(-1px); }
}

/* Respect reduced motion: drop our entrance + movement (but leave the spinner spinning) */
@media (prefers-reduced-motion: reduce) {
  .block-container, [data-testid="stMetric"], [data-testid="element-container"]:has(.js-plotly-plot),
  [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stExpander"],
  [data-testid="stHorizontalBlock"] > [data-testid="column"], [data-testid="stAlert"],
  h2, h3 { animation: none !important; }
  .stButton > button:active, .stButton > button:hover,
  [data-testid="stMetric"]:hover, [data-testid="element-container"]:has(.js-plotly-plot):hover,
  [data-baseweb="tab"]:hover { transform: none !important; }
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


@st.cache_data
def load_depth_sweep():
    path = os.path.join(config.RESULTS_DIR, "depth_sweep.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


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


# --- Plotly theming (emerging-tech, dark) ----------------------------------------
# Consistent series colours across every chart: EMERALD = HE/CKKS, GRAY = AES/plaintext
# baseline; CORAL/AMBER for additional series. The reader learns the mapping once.
# (Kept the TEAL name as an alias so downstream chart code is untouched.)
PALETTE = ["#34D399", "#FB7185", "#FBBF24", "#64748B"]
TEAL, CORAL, AMBER, GRAY = PALETTE


# Chart titles are rendered in Streamlit (st.subheader) ABOVE each chart, not inside the
# Plotly figure, so the top legend never collides with a title. Modebar hidden for cleanliness.
PLOTLY_CFG = {"displayModeBar": False}


def _style_fig(fig, ylog=False):
    """Apply the dark emerging-tech style to a Plotly figure (no in-figure title)."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Fira Sans, sans-serif", color="#F8FAFC", size=13),
        margin=dict(l=55, r=20, t=44, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E1")),
        colorway=PALETTE,
        hoverlabel=dict(bgcolor="#1E293B", bordercolor="#3B4C6B",
                        font=dict(family="Fira Code, monospace", color="#F8FAFC")),
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    axis = dict(gridcolor="#22304A", zerolinecolor="#22304A", linecolor="#475569",
                tickcolor="#475569", tickfont=dict(color="#94A3B8"),
                title_font=dict(color="#94A3B8"))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if ylog:
        fig.update_yaxes(type="log")
    return fig


def _line(piv, ytitle="", ylog=False):
    """Themed line chart from a pivot table (index = x, one column per series)."""
    fig = go.Figure()
    for col in piv.columns:
        fig.add_scatter(x=piv.index.astype(str), y=piv[col], name=str(col),
                        mode="lines+markers", line=dict(width=2.5), marker=dict(size=7))
    fig.update_xaxes(title="dataset size (records)", type="category")
    fig.update_yaxes(title=ytitle)
    return _style_fig(fig, ylog=ylog)


def headline_table(df, op="sum"):
    """Cost of `op` over data encrypted at rest: HE on-ciphertext vs AES decrypt-then-compute.

    he_steady_ms (compute + result decrypt) is the scenario-symmetric HE figure: both
    sides start from data already stored encrypted. he_oneshot_ms additionally pays the
    CKKS encryption (first upload). he_compute_ms is the pure on-ciphertext compute.
    """
    rows = []
    for size in sorted(df.dataset_size.unique()):
        he = df[(df.scheme == "CKKS") & (df.granularity == "packed")
                & (df.operation == op) & (df.dataset_size == size)]
        aes_r = df[(df.scheme == "AES-256-GCM") & (df.dataset_size == size)]
        pt = df[(df.scheme == "plaintext") & (df.operation == op) & (df.dataset_size == size)]
        if he.empty or aes_r.empty or pt.empty:
            continue
        h = he.iloc[0]
        rows.append({
            "size": int(size),
            "he_compute_ms": h.compute_time_mean * 1e3,
            "he_steady_ms": (h.compute_time_mean + h.decrypt_time_mean) * 1e3,
            "he_oneshot_ms": (h.encrypt_time_mean + h.compute_time_mean
                              + h.decrypt_time_mean) * 1e3,
            "aes_ms": (aes_r.iloc[0].decrypt_time_mean + pt.iloc[0].compute_time_mean) * 1e3,
        })
    return pd.DataFrame(rows)


def build_workflow_table(df, op="sum"):
    """Cost of computing `op` over encrypted data, under two explicit scenarios.

    Both sides start from data stored encrypted at rest under their own scheme.
    Steady-state HE = compute + result decrypt (symmetric with AES decrypt+compute);
    one-shot HE additionally pays the CKKS encryption of the dataset.
    """
    rows = []
    for size in sorted(df.dataset_size.unique()):
        aes_r = df[(df.scheme == "AES-256-GCM") & (df.dataset_size == size)]
        pt = df[(df.scheme == "plaintext") & (df.operation == op) & (df.dataset_size == size)]
        he = df[(df.scheme == "CKKS") & (df.granularity == "packed")
                & (df.operation == op) & (df.dataset_size == size)]
        if aes_r.empty or pt.empty or he.empty:
            continue
        h = he.iloc[0]
        aes_total = (aes_r.iloc[0].decrypt_time_mean + pt.iloc[0].compute_time_mean) * 1e3
        he_steady = (h.compute_time_mean + h.decrypt_time_mean) * 1e3
        he_oneshot = (h.encrypt_time_mean + h.compute_time_mean + h.decrypt_time_mean) * 1e3
        rows.append({
            "dataset_size": int(size),
            "AES: decrypt + compute (ms)": round(aes_total, 3),
            "HE steady-state: compute + result decrypt (ms)": round(he_steady, 3),
            "HE one-shot: + CKKS encryption (ms)": round(he_oneshot, 3),
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
        ck = df[(df.scheme == "CKKS") & (df.granularity == "packed")]

        # === BLUF: the bottom line first — headline numbers, then the one key chart ===
        head = headline_table(df, "sum")
        if not head.empty:
            last = head.iloc[-1]
            slowdown = last.he_steady_ms / last.aes_ms if last.aes_ms else float("nan")
            ctx_mb = (rc or {}).get("ckks_context_size_bytes")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"HE sum @ N={int(last['size']):,}", f"~{last.he_steady_ms:.0f} ms",
                      help="Steady-state: compute on ciphertext + decrypt only the result. "
                           f"One-shot (incl. CKKS encryption): ~{last.he_oneshot_ms:.0f} ms.")
            m2.metric("AES path (decrypt + compute)", f"~{last.aes_ms:.2f} ms",
                      help="Traditional path must decrypt first — plaintext is exposed "
                           "to the compute party.")
            m3.metric("HE slowdown", f"~{slowdown:,.0f}×",
                      help="Steady-state HE vs AES decrypt+compute, same data at rest on "
                           "both sides. The measured price of never exposing the data.")
            if ctx_mb:
                m4.metric("Fixed HE keys/context", f"{ctx_mb / 1e6:.0f} MB",
                          help="Public + Galois + relinearization keys, generated and "
                               "shipped once before any computation — independent of "
                               "dataset size. AES needs 32 bytes of key material.")
            else:
                m4.metric("Fixed HE keys/context", "—",
                          help="Run the benchmark to record the context size.")

            st.subheader("Computing a sum on encrypted data: HE vs traditional (AES)")
            fig = go.Figure()
            fig.add_bar(x=head["size"].astype(str), y=head.he_steady_ms,
                        name="HE steady-state — compute + result decrypt (never exposed)",
                        marker_color=TEAL,
                        hovertemplate="N=%{x}<br>%{y:.1f} ms<extra>HE steady-state</extra>")
            fig.add_bar(x=head["size"].astype(str), y=head.he_oneshot_ms,
                        name="HE one-shot — + CKKS encryption of the dataset",
                        marker_color="#6EE7B7",
                        hovertemplate="N=%{x}<br>%{y:.1f} ms<extra>HE one-shot</extra>")
            fig.add_bar(x=head["size"].astype(str), y=head.aes_ms,
                        name="AES — decrypt + compute (plaintext EXPOSED)", marker_color=GRAY,
                        hovertemplate="N=%{x}<br>%{y:.3f} ms<extra>AES</extra>")
            fig.update_xaxes(title="dataset size (records)", type="category")
            fig.update_yaxes(title="time for sum (ms, log scale)")
            st.plotly_chart(_style_fig(fig, ylog=True),
                            use_container_width=True, config=PLOTLY_CFG)
            st.caption("Scenario: both sides start from data stored encrypted at rest. "
                       "AES must decrypt it to compute (plaintext exposed); HE computes on "
                       "the ciphertext and decrypts only the result. HE's higher cost is "
                       "the price of that protection — not a defect of the implementation.")

        # === supporting detail ===
        if not ck.empty:
            st.subheader("CKKS runtime vs dataset size (packed)")
            fig = go.Figure()
            for op_name, g in ck.groupby("operation"):
                g = g.sort_values("dataset_size")
                total_std = (g.encrypt_time_std.fillna(0) ** 2
                             + g.compute_time_std.fillna(0) ** 2
                             + g.decrypt_time_std.fillna(0) ** 2) ** 0.5 * 1e3
                fig.add_scatter(
                    x=g.dataset_size.astype(int).astype(str), y=g.total_time_mean * 1e3,
                    name=str(op_name), mode="lines+markers",
                    line=dict(width=2.5), marker=dict(size=7),
                    error_y=dict(type="data", array=total_std, visible=True, width=3),
                    hovertemplate="N=%{x}<br>%{y:.1f} ms"
                                  "<extra>" + str(op_name) + "</extra>")
            fig.update_xaxes(title="dataset size (records)", type="category")
            fig.update_yaxes(title="total time (ms) — encrypt + compute + decrypt")
            st.plotly_chart(_style_fig(fig), use_container_width=True, config=PLOTLY_CFG)
            st.caption("Mean ± std over 5 timed runs (1 discarded warm-up). Totals count "
                       "one dataset encryption; add/multiply take a second operand that is "
                       "encrypted once outside the timed phase.")

        col_a, col_b = st.columns(2)
        with col_a:
            both = sorted(set(ck.dataset_size)
                          & set(df[(df.scheme == "CKKS") & (df.granularity == "elementwise")].dataset_size))
            if both:
                size = both[-1]
                st.subheader(f"Packing: packed vs element-wise (N={size})")
                g = df[(df.scheme == "CKKS") & (df.dataset_size == size)]
                pe = g.pivot_table(index="operation", columns="granularity",
                                   values="total_time_mean") * 1e3
                fig = go.Figure()
                if "packed" in pe:
                    fig.add_bar(x=pe.index, y=pe["packed"], name="packed", marker_color=TEAL)
                if "elementwise" in pe:
                    fig.add_bar(x=pe.index, y=pe["elementwise"], name="element-wise", marker_color=CORAL)
                fig.update_yaxes(title="total time (ms, log scale)")
                st.plotly_chart(_style_fig(fig, ylog=True),
                                use_container_width=True, config=PLOTLY_CFG)
                st.caption("Packing is a prerequisite, not an optimization: element-wise "
                           "is ~180× slower and ~200× larger, and does not run at all "
                           f"above N={config.ELEMENTWISE_MAX_SIZE}.")
            else:
                st.info("No size has both granularities to compare.")
        with col_b:
            if not ck.empty:
                st.subheader("CKKS approximation error (packed)")
                err_metric = st.radio("Error metric", ["mean relative", "max absolute"],
                                      horizontal=True, label_visibility="collapsed")
                col = "mean_rel_error" if err_metric == "mean relative" else "max_abs_error"
                perr = ck.pivot_table(index="dataset_size", columns="operation", values=col)
                st.plotly_chart(
                    _line(perr, ytitle=f"{err_metric} error (log scale)", ylog=True),
                    use_container_width=True, config=PLOTLY_CFG)
                st.caption("Toggle the metric: relative error is normalized by the result's "
                           "magnitude (sum ≈ 5×10⁶ vs mean ≈ 50), so sum's tiny relative "
                           "figure is partly a normalization artifact — absolute error is "
                           "the honest signal for aggregates. mul/mean share the ~10⁻⁷ "
                           "rescale floor.")

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Size: encrypted vs raw")
            sizes_all = sorted(int(s) for s in df.dataset_size.dropna().unique())
            ck_sz = df[(df.scheme == "CKKS") & (df.granularity == "packed")
                       & (df.operation == "sum")].set_index("dataset_size")
            aes_sz = df[df.scheme == "AES-256-GCM"].set_index("dataset_size")
            pt_sz = df[(df.scheme == "plaintext")
                       & (df.operation == "sum")].set_index("dataset_size")
            fig = go.Figure()
            xs = [f"{s:,}" for s in sizes_all]
            if not pt_sz.empty:
                fig.add_bar(x=xs, y=[pt_sz.ciphertext_size_bytes.get(s) / 1024 for s in sizes_all],
                            name="raw plaintext", marker_color=GRAY)
            if not aes_sz.empty:
                fig.add_bar(x=xs, y=[aes_sz.ciphertext_size_bytes.get(s) / 1024 for s in sizes_all],
                            name="AES-256-GCM (+28 B)", marker_color=AMBER)
            if not ck_sz.empty:
                ratios = [ck_sz.ciphertext_size_bytes.get(s) / pt_sz.ciphertext_size_bytes.get(s)
                          if s in ck_sz.index and s in pt_sz.index else None for s in sizes_all]
                fig.add_bar(x=xs, y=[ck_sz.ciphertext_size_bytes.get(s) / 1024 for s in sizes_all],
                            name="CKKS (packed)", marker_color=TEAL,
                            text=[("" if not r else (f"×{r:.0f}" if r >= 20 else f"×{r:.1f}"))
                                  for r in ratios],
                            textposition="outside")
            fig.update_xaxes(title="dataset size (records)", type="category")
            fig.update_yaxes(title="stored size (KB, log scale)")
            st.plotly_chart(_style_fig(fig, ylog=True),
                            use_container_width=True, config=PLOTLY_CFG)
            st.caption("×N = CKKS inflation vs raw. One CKKS ciphertext is ~334 KB however "
                       "few of its 4,096 slots are used — the overhead falls from ~209× "
                       "(N=200) toward ~10× only once the slots fill up. AES adds 28 bytes.")
        with col_d:
            st.subheader("Error vs multiplication depth")
            sweep = load_depth_sweep()
            if sweep and sweep.get("levels"):
                ok = [lv for lv in sweep["levels"] if lv.get("ok")]
                failed = next((lv for lv in sweep["levels"] if not lv.get("ok")), None)
                fig = go.Figure()
                fig.add_bar(x=[f"depth {lv['depth']} ({lv['expr']})" for lv in ok],
                            y=[lv["mean_rel_error"] for lv in ok],
                            name="mean relative error", marker_color=TEAL,
                            text=[f"{lv['mean_rel_error']:.1e}" for lv in ok],
                            textposition="outside")
                if failed:
                    top = max(lv["mean_rel_error"] for lv in ok) * 4
                    fig.add_bar(x=[f"depth {failed['depth']} ({failed['expr']})"], y=[top],
                                name=f"depth {failed['depth']}: FAILS (chain exhausted)",
                                marker=dict(color="rgba(251,113,133,0.15)",
                                            line=dict(color=CORAL, width=2),
                                            pattern=dict(shape="/", fgcolor=CORAL)))
                fig.update_yaxes(title="mean relative error (log scale)")
                st.plotly_chart(_style_fig(fig, ylog=True),
                                use_container_width=True, config=PLOTLY_CFG)
                chain = sweep.get("ckks", {}).get("coeff_mod_bit_sizes")
                st.caption(f"Measured by repeated ct×ct squaring of values in [1, 2) "
                           f"(experiments/depth_sweep.py). The chain {chain} affords two "
                           "rescales: error grows ~7× from depth 1 to 2, and a third "
                           "multiplication is refused outright — HE is depth-limited, "
                           "not merely slow.")
            else:
                st.info("No depth-sweep data. Run: `docker compose run --rm benchmark "
                        "python experiments/depth_sweep.py`")

        # Where the time goes: encrypt / compute / decrypt, at a selectable size.
        if not ck.empty:
            st.subheader("Where the time goes (CKKS, packed)")
            sizes_ck = sorted(int(s) for s in ck.dataset_size.unique())
            pick = st.radio("Dataset size", sizes_ck, index=len(sizes_ck) - 1,
                            horizontal=True, format_func=lambda s: f"{s:,}",
                            label_visibility="collapsed")
            g = ck[ck.dataset_size == pick].sort_values("operation")
            fig = go.Figure()
            fig.add_bar(x=g.operation, y=g.encrypt_time_mean * 1e3, name="encrypt",
                        marker_color=TEAL)
            fig.add_bar(x=g.operation, y=g.compute_time_mean * 1e3,
                        name="compute (on ciphertext)", marker_color=CORAL)
            fig.add_bar(x=g.operation, y=g.decrypt_time_mean * 1e3, name="decrypt result",
                        marker_color=AMBER)
            fig.update_layout(barmode="stack")
            fig.update_yaxes(title="time (ms)")
            st.plotly_chart(_style_fig(fig), use_container_width=True, config=PLOTLY_CFG)
            st.caption("sum/mean are dominated by compute: reducing a vector takes "
                       "~log₂(N) Galois-key rotations per ciphertext chunk. add/mul are "
                       "slot-wise and cheap; their cost is mostly encryption.")

        # Data-protection vs computation: largest size where all three schemes are present.
        # The CKKS protect bar uses the `add` row, whose decrypt phase is the full vector
        # result — a true whole-dataset round-trip, symmetric with the AES bar. (The `sum`
        # row's decrypt is a length-1 result and would understate CKKS here.)
        ck_add = df[(df.scheme == "CKKS") & (df.granularity == "packed") & (df.operation == "add")]
        ck_sum = df[(df.scheme == "CKKS") & (df.granularity == "packed") & (df.operation == "sum")]
        common = (set(ck_add.dataset_size)
                  & set(df[df.scheme == "AES-256-GCM"].dataset_size)
                  & set(df[df.scheme == "RSA-2048-OAEP"].dataset_size))
        if common:
            size = max(common)
            labels, vals, colors = [], [], []
            cka = ck_add[ck_add.dataset_size == size]
            if not cka.empty:
                r = cka.iloc[0]
                labels.append("CKKS protect<br>(enc + dec whole dataset)")
                vals.append((r.encrypt_time_mean + r.decrypt_time_mean) * 1e3)
                colors.append(TEAL)
            cks = ck_sum[ck_sum.dataset_size == size]
            if not cks.empty:
                labels.append("CKKS compute<br>(sum on ciphertext)")
                vals.append(cks.iloc[0].compute_time_mean * 1e3)
                colors.append("#6EE7B7")
            a = df[(df.scheme == "AES-256-GCM") & (df.dataset_size == size)]
            if not a.empty:
                r = a.iloc[0]
                labels.append("AES protect<br>(whole dataset)")
                vals.append((r.encrypt_time_mean + r.decrypt_time_mean) * 1e3)
                colors.append(GRAY)
            rr = df[(df.scheme == "RSA-2048-OAEP") & (df.dataset_size == size)]
            if not rr.empty:
                r = rr.iloc[0]
                labels.append("RSA protect<br>(ONE 8-byte record)")
                vals.append((r.encrypt_time_mean + r.decrypt_time_mean) * 1e3)
                colors.append(CORAL)
            if labels:
                st.subheader(f"Data-protection vs computation cost (N={size:,})")
                fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors,
                                       text=[f"{v:,.2f} ms" for v in vals],
                                       textposition="outside"))
                fig.update_yaxes(title="time (ms, log scale)")
                st.plotly_chart(_style_fig(fig, ylog=True),
                                use_container_width=True, config=PLOTLY_CFG)
                st.caption("Protection = encrypt + decrypt the dataset itself. AES/RSA are "
                           "protection only — they cannot compute on ciphertext — and the "
                           "RSA bar covers a single 8-byte record (RSA-2048-OAEP cannot "
                           "hold bulk data; real systems wrap an AES key instead).")

        # Run configuration & environment (reproducibility context).
        if rc:
            with st.expander("Run configuration & environment (reproducibility)"):
                ckks = rc.get("ckks", {})
                env = rc.get("environment", {})
                ctx_bytes = rc.get("ckks_context_size_bytes")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**CKKS / experiment**")
                    st.markdown(
                        f"- `poly_modulus_degree` = {ckks.get('poly_modulus_degree', '—')} "
                        f"→ {ckks.get('slots', '—')} slots/ciphertext\n"
                        f"- coeff modulus chain = {ckks.get('coeff_mod_bit_sizes', '—')} "
                        f"(~128-bit security)\n"
                        f"- global scale = 2⁴⁰\n"
                        f"- keys/context ≈ {ctx_bytes / 1e6:.0f} MB (fixed, one-time)\n"
                        f"- seed = {rc.get('seed', '—')}, "
                        f"{rc.get('n_repeats', '—')} repeats + 1 discarded warm-up\n"
                        f"- sizes = {rc.get('dataset_sizes', '—')}, element-wise capped at "
                        f"{rc.get('elementwise_max_size', '—')}")
                with c2:
                    st.markdown("**Environment of the committed run**")
                    st.markdown(
                        f"- Python {env.get('python_version', '—')} on "
                        f"`{env.get('platform', '—')}`\n"
                        f"- CPU: {env.get('cpu_model', env.get('processor', '—'))} "
                        f"({env.get('cpu_count', '—')} cores"
                        + (f", {env.get('memory_total_gb')} GB RAM" if env.get('memory_total_gb') else "")
                        + ")\n"
                        f"- TenSEAL {env.get('tenseal_version', '—')}, "
                        f"NumPy {env.get('numpy_version', '—')}, "
                        f"cryptography {env.get('cryptography_version', '—')}\n"
                        f"- run timestamp: `{rc.get('timestamp', '—')}`")

        st.subheader("All results")
        fcol, dcol = st.columns([3, 1])
        with fcol:
            schemes = st.multiselect("Filter scheme", sorted(df.scheme.unique()),
                                     default=sorted(df.scheme.unique()))
        with dcol:
            st.download_button("Download results.csv",
                               data=df.to_csv(index=False).encode(),
                               file_name="results.csv", mime="text/csv",
                               use_container_width=True)
        view = (df[df.scheme.isin(schemes)] if schemes else df).copy()
        for c in ("encrypt_time_mean", "compute_time_mean", "decrypt_time_mean"):
            view[c.replace("_time_mean", "_ms")] = view[c] * 1e3
        view["ciphertext_KB"] = view.ciphertext_size_bytes / 1024
        show_cols = ["scheme", "operation", "dataset_size", "granularity",
                     "encrypt_ms", "compute_ms", "decrypt_ms",
                     "ciphertext_KB", "mean_rel_error", "correct", "notes"]
        st.dataframe(
            view[show_cols], use_container_width=True, hide_index=True,
            column_config={
                "dataset_size": st.column_config.NumberColumn("N", format="%d"),
                "encrypt_ms": st.column_config.NumberColumn("encrypt (ms)", format="%.3f"),
                "compute_ms": st.column_config.NumberColumn("compute (ms)", format="%.3f"),
                "decrypt_ms": st.column_config.NumberColumn("decrypt (ms)", format="%.3f"),
                "ciphertext_KB": st.column_config.NumberColumn("ciphertext (KB)",
                                                               format="%.1f"),
                "mean_rel_error": st.column_config.NumberColumn("mean rel. error",
                                                                format="%.2e"),
            })
        st.caption("Times are means over 5 runs (stds are in the CSV). AES/RSA rows have "
                   "no compute column — they cannot compute on ciphertext. RSA rows repeat "
                   "one per-record measurement at each N for context; the cost does not "
                   "scale with the dataset.")

        # Memory last and de-emphasized: RSS is noisy, not the headline signal.
        with st.expander("Peak memory (RSS) — indicative only, not the headline signal"):
            if not ck.empty and not ck["peak_memory_mb"].dropna().empty:
                pmem = ck.pivot_table(index="dataset_size", columns="operation",
                                      values="peak_memory_mb")
                st.plotly_chart(
                    _line(pmem, ytitle="peak memory delta (MB)"),
                    use_container_width=True, config=PLOTLY_CFG)
                st.caption("Process-RSS deltas are only meaningful on a cold first "
                           "operation; once pages are resident, later ops reuse them and "
                           "read near-zero. The authoritative size/memory signal is "
                           "serialized ciphertext size plus the fixed key/context size.")
            else:
                st.info("No memory data available.")

        st.caption("Full static figures are in figures/ and embedded in "
                   "report/FINAL_REPORT.docx.")


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
        m4.metric("Ciphertext size", f"{ct_size/1024:.1f} KB",
                  help=f"{n_ct} ciphertext(s), {config.CKKS_SLOTS} slots each. On top of "
                       "this comes the fixed ~34 MB public context (keys), shipped once — "
                       "see the Dashboard tab.")

        if correct:
            st.success(f"Decrypted result matches plaintext within tolerance "
                       f"(mean rel. error {mean_rel:.2e}, max abs {max_abs:.2e}; "
                       f"tolerance: rel {config.CKKS_REL_TOL:g} / abs {config.CKKS_ABS_TOL:g}).")
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
        "so the data is never exposed. Both columns below start from data **stored encrypted "
        "at rest** under each scheme: *steady-state* HE is compute + result decrypt (the "
        "scenario-symmetric comparison); *one-shot* HE additionally pays the CKKS encryption "
        "of the dataset (first upload to an untrusted cloud)."
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
            if op == "sum":
                p = os.path.join(config.FIGURES_DIR, "workflow_comparison.png")
                if os.path.exists(p):
                    st.image(p, use_column_width=True,
                             caption="Static figure (op = sum) — regenerate via "
                                     "`docker compose run --rm plots`.")
            st.caption("AES exposes plaintext during computation; HE does not. HE's higher "
                       "cost is the measured overhead of protecting data in use.")
