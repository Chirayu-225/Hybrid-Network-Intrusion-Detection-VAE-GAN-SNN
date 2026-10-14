"""Streamlit analyst dashboard.

    streamlit run src/nids/dashboard/app.py

Replays flows (the bundled demo sample or an uploaded CICFlowMeter CSV) through the inference API - or the
local pipeline when no API is reachable - and shows what an analyst would have to review.

Environment:
    API_URL      inference API base URL (default http://localhost:8000)
    NIDS_CONFIG  YAML config, used for the demo sample path and for local mode
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from nids.config import load_config

CFG_PATH = os.environ.get("NIDS_CONFIG", "configs/default.yaml")
DEFAULT_API = os.environ.get("API_URL", "http://localhost:8000")
BATCH = 250

ACTION_ORDER = ["allow", "monitor", "block"]
# Semantic status colours: teal = fine, amber = look at it, red = act. They also differ in lightness,
# so the three stay distinguishable without relying on hue alone.
ACTION_COLORS = {"allow": "#2DD4A7", "monitor": "#F5B83D", "block": "#F2545B"}
ACTION_RANGE = [ACTION_COLORS[a] for a in ACTION_ORDER]

INK = "#E6EBF5"
MUTED = "#8A96AD"
GRID = "#24304A"
SURFACE = "#121A2B"
ACCENT = "#4C8DFF"

st.set_page_config(page_title="Hybrid NIDS | SOC console", page_icon="🛡️", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 85% -10%, #14213d 0%, #0A0F1C 55%); }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ---- hero ---- */
.hero { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
        padding: 1.1rem 1.4rem; border:1px solid #24304A; border-radius:16px;
        background: linear-gradient(135deg, #131d33 0%, #0f1729 100%); margin-bottom: 1.2rem; }
.hero-left { display:flex; align-items:center; gap: 1rem; }
.hero-logo { width:46px; height:46px; border-radius:12px; display:grid; place-items:center;
             background: linear-gradient(135deg, #4C8DFF 0%, #2DD4A7 100%); box-shadow: 0 6px 24px rgba(76,141,255,.35); }
.hero h1 { font-size: 1.45rem; font-weight: 700; margin:0; padding:0; letter-spacing:-.01em; color:#E6EBF5; }
.hero p { margin:.15rem 0 0 0; color:#8A96AD; font-size:.88rem; }
.pills { display:flex; gap:.5rem; flex-wrap:wrap; }
.pill { display:inline-flex; align-items:center; gap:.45rem; padding:.3rem .75rem; border-radius:999px;
        font-size:.78rem; font-weight:500; color:#C9D2E3; background:#0d1424; border:1px solid #24304A; }
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
.dot.ok { background:#2DD4A7; box-shadow:0 0 8px #2DD4A7; }
.dot.bad { background:#F2545B; box-shadow:0 0 8px #F2545B; }
.dot.idle { background:#8A96AD; }

/* ---- KPI cards ---- */
.kpi { position:relative; background:#121A2B; border:1px solid #24304A; border-radius:14px;
       padding: 1rem 1.1rem 1rem 1.25rem; overflow:hidden; height:100%; }
.kpi::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background: var(--c, #4C8DFF); }
.kpi .lbl { font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color:#8A96AD; font-weight:600; }
.kpi .val { font-size:1.9rem; font-weight:700; color:#E6EBF5; line-height:1.15; margin-top:.25rem;
            font-variant-numeric: tabular-nums; }
.kpi .sub { font-size:.78rem; color:#8A96AD; margin-top:.15rem; }

/* ---- section headings ---- */
.sec { display:flex; align-items:baseline; gap:.6rem; margin: 1.5rem 0 .6rem 0; }
.sec h3 { font-size:1.02rem; font-weight:600; color:#E6EBF5; margin:0; padding:0; }
.sec span { font-size:.8rem; color:#8A96AD; }

/* ---- verdict banner ---- */
.verdict { border-radius:14px; padding:1rem 1.3rem; border:1px solid var(--c); background: var(--bg);
           display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
.verdict .act { font-size:1.5rem; font-weight:800; letter-spacing:.06em; color:var(--c); }
.verdict .cls { font-size:1rem; color:#E6EBF5; font-weight:600; }
.verdict .meta { font-size:.82rem; color:#8A96AD; }

/* ---- score bars ---- */
.bar { background:#121A2B; border:1px solid #24304A; border-radius:12px; padding:.85rem 1rem; }
.bar .top { display:flex; justify-content:space-between; font-size:.8rem; color:#8A96AD; margin-bottom:.5rem; }
.bar .top b { color:#E6EBF5; font-size:1rem; font-variant-numeric: tabular-nums; }
.track { height:8px; border-radius:999px; background:#0A0F1C; overflow:hidden; }
.fill { height:100%; border-radius:999px; background: linear-gradient(90deg, #4C8DFF, var(--c, #4C8DFF)); }

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom:1px solid #24304A; }
.stTabs [data-baseweb="tab"] { height: 42px; padding: 0 1rem; color:#8A96AD; font-weight:500; background:transparent; }
.stTabs [aria-selected="true"] { color:#E6EBF5 !important; }

/* ---- sidebar ---- */
section[data-testid="stSidebar"] { background:#0d1424; border-right:1px solid #1b2640; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { font-size:.72rem !important;
    letter-spacing:.1em; text-transform:uppercase; color:#8A96AD !important; font-weight:600; margin-top:1rem; }
.brand { display:flex; align-items:center; gap:.6rem; font-weight:700; color:#E6EBF5; font-size:1rem; margin-bottom:.4rem; }
.brand .mark { width:28px; height:28px; border-radius:8px; display:grid; place-items:center;
               background: linear-gradient(135deg, #4C8DFF 0%, #2DD4A7 100%); }

/* ---- buttons, tables ---- */
.stButton > button[kind="primary"], .stDownloadButton > button { border-radius:10px; font-weight:600; border:0; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #4C8DFF, #3366d6); }
[data-testid="stDataFrame"] { border:1px solid #24304A; border-radius:12px; overflow:hidden; }
.note { color:#8A96AD; font-size:.8rem; border-left:3px solid #24304A; padding:.1rem 0 .1rem .75rem; margin:.6rem 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SHIELD_SVG = (
    '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 3v6c0 4.5-3.2 8.3-8 9.5C7.2 20.3 4 16.5 4 12V6l8-3z"/>'
    '<path d="M9 12l2 2 4-4"/></svg>'
)


# ------------------------------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading local models...")
def _local_pipeline():
    from nids.inference import HybridNIDS

    return HybridNIDS.load(load_config(CFG_PATH))


def _api_health(url: str) -> bool:
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=1.5)
        return r.ok and bool(r.json().get("model_loaded"))
    except Exception:
        return False


def _score_via_api(df: pd.DataFrame, url: str) -> tuple[list[dict], float]:
    results, latencies = [], []
    progress = st.progress(0.0, text="Scoring flows...")
    for start in range(0, len(df), BATCH):
        chunk = df.iloc[start:start + BATCH]
        payload = {"flows": chunk.replace({np.nan: None}).to_dict(orient="records")}
        resp = requests.post(f"{url.rstrip('/')}/predict", json=payload, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        results += body["results"]
        latencies.append(body["latency_ms"] / max(1, body["n_flows"]))
        done = min(start + BATCH, len(df))
        progress.progress(min(1.0, done / len(df)), text=f"Scoring flows... {done}/{len(df)}")
    progress.empty()
    return results, float(np.mean(latencies))


def _score_locally(df: pd.DataFrame) -> tuple[list[dict], float]:
    import time

    pipe = _local_pipeline()
    t0 = time.perf_counter()
    records = pipe.to_records(pipe.predict_frame(df))
    return records, (time.perf_counter() - t0) * 1000 / max(1, len(df))


def _flatten(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if k != "top_classes"} for r in records])


def _load_demo() -> pd.DataFrame | None:
    cfg = load_config(CFG_PATH)
    path = Path(cfg.paths.artifacts) / "data" / "demo_flows.csv"
    return pd.read_csv(path) if path.exists() else None


def kpi(label: str, value: str, sub: str = "", color: str = ACCENT) -> str:
    return (f'<div class="kpi" style="--c:{color}"><div class="lbl">{html.escape(label)}</div>'
            f'<div class="val">{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>')


def section(title: str, hint: str = "") -> None:
    st.markdown(f'<div class="sec"><h3>{html.escape(title)}</h3><span>{html.escape(hint)}</span></div>',
                unsafe_allow_html=True)


def score_bar(label: str, value: float, color: str, note: str = "") -> str:
    pct = max(0.0, min(1.0, float(value))) * 100
    return (f'<div class="bar" style="--c:{color}"><div class="top"><span>{html.escape(label)}</span>'
            f'<b>{value:.3f}</b></div><div class="track"><div class="fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="top" style="margin:.45rem 0 0 0"><span>{html.escape(note)}</span></div></div>')


def style_chart(chart: alt.Chart | alt.LayerChart) -> alt.Chart:
    return (chart.configure(background="rgba(0,0,0,0)")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=GRID, tickColor=GRID,
                            labelFontSize=11, titleFontSize=11, titleFontWeight=500)
            .configure_legend(labelColor=INK, titleColor=MUTED, labelFontSize=11, titleFontSize=11, symbolType="circle")
            .configure_title(color=INK))


def action_color_scale(legend: alt.Legend | None = None) -> alt.Color:
    return alt.Color("action:N", sort=ACTION_ORDER, scale=alt.Scale(domain=ACTION_ORDER, range=ACTION_RANGE), legend=legend)


def friendly_failure(exc: Exception, mode: str, url: str | None) -> None:
    if mode == "API" and isinstance(exc, requests.exceptions.ConnectionError):
        st.error(f"Cannot reach the inference API at **{url}**.")
        st.markdown(
            "- Start it in a separate terminal and leave that terminal open: "
            "`uvicorn nids.serving.api:app --port 8010`\n"
            "- Then enter `http://localhost:8010` as **API URL** in the sidebar and press **Score flows**.\n"
            "- Or switch **Scoring backend** to **Local models** to run without a server."
        )
    else:
        st.error(f"Scoring failed: {exc}")
    st.stop()


# ------------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f'<div class="brand"><div class="mark">{SHIELD_SVG.format(s=16)}</div>Hybrid NIDS</div>',
                unsafe_allow_html=True)
    st.caption("VAE + GAN + SNN intrusion detection")

    st.header("Scoring backend")
    mode = st.radio("Backend", ["API", "Local models"], label_visibility="collapsed",
                    help="API calls the FastAPI service; local loads artifacts directly.")
    api_url = st.text_input("API URL", DEFAULT_API) if mode == "API" else None

    st.header("Traffic source")
    source = st.radio("Flows", ["Demo replay", "Upload CSV"], label_visibility="collapsed")
    upload = st.file_uploader("CICFlowMeter CSV", type="csv") if source == "Upload CSV" else None
    max_rows = st.slider("Max flows", 100, 5000, 1000, step=100)
    run = st.button("Score flows", type="primary", width="stretch")

backend_online = _api_health(api_url) if mode == "API" else True
if mode == "API":
    status_dot = "ok" if backend_online else "bad"
    status_text = f"API online · {api_url}" if backend_online else f"API offline · {api_url}"
else:
    status_dot, status_text = "ok", "Local models"

st.markdown(
    f"""<div class="hero"><div class="hero-left"><div class="hero-logo">{SHIELD_SVG.format(s=24)}</div>
    <div><h1>Network intrusion detection console</h1>
    <p>Unsupervised VAE anomaly score fused with an SNN classifier into a risk score and an allow / monitor / block action.</p></div></div>
    <div class="pills"><span class="pill"><span class="dot {status_dot}"></span>{html.escape(status_text)}</span></div></div>""",
    unsafe_allow_html=True,
)

if source == "Demo replay":
    flows = _load_demo()
    if flows is None:
        st.info("No demo sample found. Run `nids prepare` first (it writes artifacts/data/demo_flows.csv).")
        st.stop()
else:
    if upload is None:
        st.info("Upload a CICFlowMeter CSV in the sidebar to score it.")
        st.stop()
    flows = pd.read_csv(upload, encoding="latin-1")
    flows.columns = [c.strip() for c in flows.columns]

flows = flows.head(max_rows)
truth = flows["Label"].astype(str) if "Label" in flows.columns else None
features = flows.drop(columns=[c for c in ("Label", "label_raw") if c in flows.columns])

if run or "scored" not in st.session_state:
    try:
        if mode == "API":
            records, per_flow_ms = _score_via_api(features, api_url)
        else:
            records, per_flow_ms = _score_locally(features)
    except Exception as exc:  # surface any backend failure instead of a traceback
        friendly_failure(exc, mode, api_url)
    st.session_state["scored"] = (records, per_flow_ms)

records, per_flow_ms = st.session_state["scored"]
res = _flatten(records)
if truth is not None:
    res["true_class"] = truth.to_numpy()[: len(res)]
res.index.name = "flow"

n = len(res)
share = res["action"].value_counts(normalize=True).reindex(ACTION_ORDER).fillna(0) * 100
counts = res["action"].value_counts().reindex(ACTION_ORDER).fillna(0).astype(int)

tab_overview, tab_queue, tab_inspect, tab_about = st.tabs(["Overview", "Alert queue", "Flow inspector", "About"])

# ------------------------------------------------------------------------------------------ overview
with tab_overview:
    cols = st.columns(5)
    cards = [
        ("Flows scored", f"{n:,}", "in this replay", ACCENT),
        ("Allowed", f"{share['allow']:.1f}%", f"{counts['allow']:,} flows", ACTION_COLORS["allow"]),
        ("Monitor", f"{share['monitor']:.1f}%", f"{counts['monitor']:,} flows for analyst review", ACTION_COLORS["monitor"]),
        ("Block", f"{share['block']:.1f}%", f"{counts['block']:,} flows", ACTION_COLORS["block"]),
        ("Latency / flow", f"{per_flow_ms:.2f} ms", "end to end, batched", "#9B87F5"),
    ]
    for col, (lbl, val, sub, colr) in zip(cols, cards, strict=True):
        col.markdown(kpi(lbl, val, sub, colr), unsafe_allow_html=True)

    if truth is not None:
        is_attack = res["true_class"].str.lower() != "benign"
        review = res["action"].isin(["monitor", "block"])
        section("Detection quality", "against the labels in this file")
        q1, q2, q3 = st.columns(3)
        q1.markdown(kpi("Attack recall", f"{review[is_attack].mean() * 100:.1f}%" if is_attack.any() else "n/a",
                        "attacks sent to monitor or block", ACTION_COLORS["allow"]), unsafe_allow_html=True)
        q2.markdown(kpi("Review precision", f"{is_attack[review].mean() * 100:.1f}%" if review.any() else "n/a",
                        "flagged flows that are real attacks", ACCENT), unsafe_allow_html=True)
        q3.markdown(kpi("Benign sent to review", f"{review[~is_attack].mean() * 100:.1f}%" if (~is_attack).any() else "n/a",
                        "false-alarm rate on benign flows", ACTION_COLORS["monitor"]), unsafe_allow_html=True)
        st.markdown('<div class="note">The demo sample is class-balanced (about half attacks), so precision is far higher '
                    'than it would be on live traffic.</div>', unsafe_allow_html=True)

    section("Risk and decisions", "how the fused score separates traffic")
    left, right = st.columns([2, 1])
    with left:
        hist = alt.Chart(res).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=30), title="Risk score"),
            y=alt.Y("count():Q", title="Flows"),
            color=action_color_scale(alt.Legend(title="Action", orient="top", direction="horizontal")),
            order=alt.Order("action:N", sort="ascending"),
            tooltip=[alt.Tooltip("count():Q", title="Flows"), alt.Tooltip("action:N", title="Action")],
        ).properties(height=300)
        st.altair_chart(style_chart(hist), width="stretch")
    with right:
        mix = pd.DataFrame({"action": ACTION_ORDER, "flows": [int(counts[a]) for a in ACTION_ORDER]})
        mix = mix[mix["flows"] > 0]
        donut = alt.Chart(mix).mark_arc(innerRadius=62, outerRadius=100, stroke="#0A0F1C", strokeWidth=2).encode(
            theta=alt.Theta("flows:Q"), color=action_color_scale(alt.Legend(title="Action", orient="bottom", direction="horizontal")),
            tooltip=["action", "flows"],
        ).properties(height=300)
        st.altair_chart(style_chart(donut), width="stretch")

    split_col = "true_class" if truth is not None else "predicted_class"
    section("Action mix by " + ("true class" if truth is not None else "predicted class"),
            "share of each class that was allowed, monitored or blocked")
    mixc = res.groupby([split_col, "action"]).size().reset_index(name="flows")
    mixc["total"] = mixc.groupby(split_col)["flows"].transform("sum")
    mixc["share"] = mixc["flows"] / mixc["total"]
    order = mixc.groupby(split_col)["total"].first().sort_values(ascending=False).index.tolist()
    stacked = alt.Chart(mixc).mark_bar().encode(
        y=alt.Y(f"{split_col}:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("share:Q", title="Share of class", axis=alt.Axis(format="%"), stack="normalize"),
        color=action_color_scale(alt.Legend(title="Action", orient="top", direction="horizontal")),
        order=alt.Order("action:N", sort="ascending"),
        tooltip=[alt.Tooltip(f"{split_col}:N", title="Class"), alt.Tooltip("action:N", title="Action"),
                 alt.Tooltip("flows:Q", title="Flows"), alt.Tooltip("share:Q", title="Share", format=".1%")],
    ).properties(height=max(180, 34 * len(order)))
    st.altair_chart(style_chart(stacked), width="stretch")

# ------------------------------------------------------------------------------------------ alert queue
with tab_queue:
    section("Alert queue", "highest risk first")
    f1, f2, f3 = st.columns([2, 2, 3])
    sel_actions = f1.multiselect("Action", ACTION_ORDER, default=["monitor", "block"])
    classes = sorted(res["predicted_class"].unique())
    sel_classes = f2.multiselect("Predicted class", classes, default=classes)
    min_risk = f3.slider("Minimum risk score", 0.0, 1.0, 0.0, step=0.05)

    view = res[res["action"].isin(sel_actions) & res["predicted_class"].isin(sel_classes) & (res["risk_score"] >= min_risk)]
    view = view.sort_values("risk_score", ascending=False)
    st.caption(f"{len(view):,} of {n:,} flows match.")

    if len(view):
        shown = view.head(500)
        lead = [c for c in ("action", "predicted_class", "risk_score", "anomaly_score", "attack_probability", "confidence") if c in shown.columns]
        rest = [c for c in shown.columns if c not in lead]
        st.dataframe(
            shown[lead + rest], width="stretch", height=430,
            column_config={
                "risk_score": st.column_config.ProgressColumn("Risk", min_value=0.0, max_value=1.0, format="%.3f"),
                "anomaly_score": st.column_config.ProgressColumn("Anomaly (VAE)", min_value=0.0, max_value=1.0, format="%.3f"),
                "attack_probability": st.column_config.ProgressColumn("Attack prob. (SNN)", min_value=0.0, max_value=1.0, format="%.3f"),
                "confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
                "action": st.column_config.TextColumn("Action"),
                "predicted_class": st.column_config.TextColumn("Predicted class"),
            },
        )
        st.download_button("Download filtered alerts (CSV)", view.to_csv().encode("utf-8"),
                           file_name="nids_alerts.csv", mime="text/csv")
        if len(view) > 500:
            st.caption("Showing the top 500; the download contains all matching flows.")
    else:
        st.info("No flows match the current filters.")

# ------------------------------------------------------------------------------------------ inspector
with tab_inspect:
    section("Flow inspector", "why a single flow got its verdict")
    top = res.sort_values("risk_score", ascending=False).head(100)
    flow_id = st.selectbox(
        "Flow (top 100 by risk)", list(top.index),
        format_func=lambda i: f"#{i}  ·  {res.loc[i, 'predicted_class']}  ·  risk {res.loc[i, 'risk_score']:.2f}  ·  {res.loc[i, 'action']}",
    )
    row, rec = res.loc[flow_id], records[int(flow_id)]
    colr = ACTION_COLORS.get(row["action"], ACCENT)
    truth_txt = f" · true class: {row['true_class']}" if "true_class" in row else ""
    st.markdown(
        f'<div class="verdict" style="--c:{colr};--bg:{colr}1A"><div><div class="act">{html.escape(str(row["action"]).upper())}</div>'
        f'<div class="cls">{html.escape(str(row["predicted_class"]))}</div></div>'
        f'<div class="meta">flow #{int(flow_id)} · confidence {row.get("confidence", float("nan")):.1%}{html.escape(truth_txt)}</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    b1, b2, b3 = st.columns(3)
    b1.markdown(score_bar("Fused risk score", row["risk_score"], colr, "noisy-OR of both detectors"), unsafe_allow_html=True)
    b2.markdown(score_bar("Anomaly score (VAE)", row["anomaly_score"], "#9B87F5", "distance from benign traffic"), unsafe_allow_html=True)
    b3.markdown(score_bar("Attack probability (SNN)", row["attack_probability"], ACCENT, "1 - P(Benign), calibrated"), unsafe_allow_html=True)

    section("Class probabilities", "top candidates from the SNN")
    probs = pd.DataFrame(rec["top_classes"]).rename(columns={"class": "Class", "probability": "Probability"})
    chart = alt.Chart(probs).mark_bar(color=ACCENT, cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=22).encode(
        y=alt.Y("Class:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5)),
        tooltip=["Class", alt.Tooltip("Probability:Q", format=".3f")],
    ).properties(height=max(110, 44 * len(probs)))
    st.altair_chart(style_chart(chart), width="stretch")

# ------------------------------------------------------------------------------------------ about
with tab_about:
    section("How a verdict is produced")
    st.markdown(
        """
**1. Anomaly score (VAE).** A variational autoencoder trained on benign traffic only. Reconstruction error and latent
distance are mapped through the benign validation distribution to a 0 to 1 score, so it can flag attacks it has never seen.

**2. Classification (SNN).** A spiking neural network (snnTorch, leaky integrate-and-fire) predicts the attack family. Rare
classes were augmented with a conditional WGAN-GP during training only.

**3. Fusion.** `risk = 1 - (1 - p_attack) * (1 - 0.8 * anomaly)`. Either detector can raise the alarm. A flow the classifier
calls benign but the VAE finds anomalous is labelled `Unknown-Anomaly` and is never silently allowed.

**4. Action.** Thresholds are calibrated on validation so benign false-positive rates hit configured targets.
"""
    )
    st.markdown('<div class="note">Dashboard figures on the synthetic or demo sample illustrate the code path. They are not a benchmark '
                'of real-world performance.</div>', unsafe_allow_html=True)
