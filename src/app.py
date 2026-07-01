# ======================================================
# File: app.py
# Project: Agentic ICU Delay Prediction — Streamlit UI
# Run:  streamlit run src/app.py
# Place: D:\agentic_clinical_workflow\src\
# Fixes:
#   - fillcolor rgba fix (violin + histogram)
#   - latest.get() → .to_dict() (BUG FIX)
#   - st.radio label_visibility safe guard
#   - Added: Radar chart, Scatter plot, Violin plot
#   - Added: Correlation heatmap, Delay distribution
#   - Added: Loss curve from actual training log
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="ICU Delay Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_data_up   = os.path.join(BASE_DIR, "..", "data")
_data_here = os.path.join(BASE_DIR, "data")
DATA_DIR   = os.path.normpath(_data_up if os.path.isdir(_data_up) else _data_here)

DATA_PATH   = os.path.join(DATA_DIR, "lstm_final_dataset_v3.csv")
MODEL_PATH  = os.path.join(DATA_DIR, "delay_lstm_model.pth")
SCALER_PATH = os.path.join(DATA_DIR, "scaler_lstm.pkl")

SEQ_LEN  = 5
FEATURES = [
    "trigger_encoded", "time_gap_min",
    "HR", "MAP", "RR", "SPO2",
    "hour_of_day", "is_night",
    "patient_age", "gender_encoded",
]

TRIGGER_MAP = {"HR_HIGH": 0, "MAP_LOW": 1, "SPO2_LOW": 2, "RR_HIGH": 3}
TRIGGER_COLOR = {
    "HR_HIGH":  "#ff7043",
    "MAP_LOW":  "#00c8f0",
    "SPO2_LOW": "#00e8a0",
    "RR_HIGH":  "#f0c040",
}

# ── FIXED: rgba fill colors for violin plots ───────────
VIOLIN_FILL = {
    "HR_HIGH":  "rgba(255,112,67,0.2)",
    "MAP_LOW":  "rgba(0,200,240,0.2)",
    "SPO2_LOW": "rgba(0,232,160,0.2)",
    "RR_HIGH":  "rgba(240,192,64,0.2)",
}

KNOWN_STAYS = {
    30000831: "30000831 — HR_HIGH · Cardiac",
    30002654: "30002654 — MAP_LOW · Cardiac",
    30003306: "30003306",
    30004391: "30004391",
    30005199: "30005199",
    30007175: "30007175",
    30008148: "30008148",
    30009505: "30009505",
    30009597: "30009597",
    30009914: "30009914",
    30011071: "30011071",
    30013315: "30013315",
    30014139: "30014139",
    30014468: "30014468",
    30015770: "30015770",
    30016351: "30016351",
    30016557: "30016557",
    30017005: "30017005",
    30019367: "30019367",
    30019763: "30019763",
}

BG_MAIN  = "#03070d"
BG_CARD  = "#080f1a"
BG_PANEL = "#0a1525"
BORDER   = "#162436"
TEXT_PRI = "#e8f4ff"
TEXT_SEC = "#5880a8"
ACCENT   = "#00c8f0"

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
  }
  .stApp { background-color: #03070d; }

  .metric-card {
    background: #080f1a;
    border: 1px solid #162436;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
  }
  .metric-label {
    font-size: 10px; color: #5880a8;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 6px;
  }
  .metric-value {
    font-size: 28px; font-weight: 700; color: #e8f4ff;
    line-height: 1;
  }
  .metric-sub { font-size: 11px; color: #5880a8; margin-top: 4px; }

  .severity-badge {
    display: inline-block;
    padding: 6px 16px; border-radius: 20px;
    font-size: 13px; font-weight: 700;
    letter-spacing: 1px;
  }
  .agent-step {
    background: #080f1a;
    border-left: 3px solid #00c8f0;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 12px;
  }
  .agent-label {
    color: #00c8f0; font-size: 10px;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 4px;
  }
  .agent-val { color: #e8f4ff; }
  .trigger-badge {
    display: inline-block;
    padding: 4px 10px; border-radius: 4px;
    font-size: 11px; font-weight: 700;
  }
  .section-head {
    font-size: 10px; color: #5880a8;
    text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #162436;
    padding-bottom: 6px; margin-bottom: 14px;
    margin-top: 8px;
  }
  .tab-content { padding-top: 16px; }
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ── LSTM Model definition ──────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len(FEATURES), hidden_size=128,
            num_layers=2, batch_first=True, dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Load model + scaler ────────────────────────────────
@st.cache_resource
def load_model():
    model = LSTMModel()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    )
    model.eval()
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    return model, scaler


# ── Load dataset ───────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["trigger_time"])

    if "trigger_encoded" not in df.columns:
        df["trigger_encoded"] = df["trigger_type"].map(
            TRIGGER_MAP).fillna(-1).astype(int)

    vitals = ["HR", "MAP", "RR", "SPO2"]
    for col in vitals:
        df[col] = df.groupby("trigger_type")[col].transform(
            lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    df = df.sort_values(["stay_id", "trigger_time"])

    df["time_gap_min"] = (
        df.groupby("stay_id")["trigger_time"]
        .diff().dt.total_seconds() / 60
    ).fillna(0).clip(0, 1440)

    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = df["trigger_time"].dt.hour
    if "is_night" not in df.columns:
        df["is_night"] = df["hour_of_day"].apply(
            lambda h: 1 if (h >= 22 or h <= 6) else 0)
    if "is_weekend" in df.columns:
        df = df.drop(columns=["is_weekend"])
    if "patient_age" not in df.columns:
        df["patient_age"] = 65
    else:
        df["patient_age"] = df["patient_age"].fillna(
            df["patient_age"].median())
    if "gender_encoded" not in df.columns:
        df["gender_encoded"] = -1
    else:
        df["gender_encoded"] = df["gender_encoded"].fillna(-1)

    eligible = df.groupby("stay_id").filter(
        lambda x: len(x) > SEQ_LEN
    )["stay_id"].unique()

    return df, eligible


# ── Helpers ────────────────────────────────────────────
def predict_delay(seq_df, model, scaler):
    seq_vals   = seq_df[FEATURES].astype(float).values
    seq_scaled = scaler.transform(seq_vals)
    tensor     = torch.tensor(
        seq_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_log = model(tensor).item()
    pred_min = np.expm1(pred_log)
    return pred_min / 60


def get_severity(deviation):
    if deviation <= 0:
        return "No Delay", "#00e8a0"
    elif deviation <= 1:
        return "Mild",     "#f0c040"
    elif deviation <= 3:
        return "Moderate", "#ff7043"
    else:
        return "Severe",   "#ff2050"


def plotly_base():
    return dict(
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(family="Space Mono", size=10, color=TEXT_SEC),
        margin=dict(l=10, r=10, t=30, b=10),
    )


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 ICU Delay Monitor")
    st.markdown(
        "<div style='font-size:11px;color:#5880a8;"
        "margin-bottom:20px'>LSTM · Multi-Agent · MIMIC-IV</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='section-head'>Agent Mode</div>",
                unsafe_allow_html=True)

    mode = st.radio("Agent Mode",
                    ["⚡ Reactive", "🔮 Proactive"],
                    label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-head'>Patient Stay</div>",
                unsafe_allow_html=True)

    known_label = st.selectbox(
        "Quick select (proven stays)",
        options=["— select —"] + list(KNOWN_STAYS.values()),
    )

    default_id = 30000831
    for sid, label in KNOWN_STAYS.items():
        if label == known_label:
            default_id = sid
            break

    stay_input = st.number_input(
        "Or type any Stay ID",
        min_value=0,
        value=int(default_id),
        step=1,
        help="Any stay_id from lstm_final_dataset_v3.csv"
    )

    run_btn = st.button("▶ Run Analysis", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-head'>Model Info</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px;color:#5880a8;line-height:1.8'>
    LSTM hidden: 128<br>Layers: 2 · Dropout: 0.3<br>
    Features: 10<br>Seq length: 5<br>
    Train/Val/Test: 70/15/15<br>
    MAE: 88.5 min · RMSE: 163.6 min
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# HEADER + KPI ROW
# ══════════════════════════════════════════════════════
st.markdown("""
<h1 style='font-size:22px;font-weight:700;color:#e8f4ff;
margin-bottom:4px'>ICU Clinical Response Delay Prediction</h1>
<p style='font-size:11px;color:#5880a8;margin-bottom:24px'>
Agentic monitoring system · MIMIC-IV · LSTM
</p>""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
kpi_data = [
    (k1, "MAE",          "88.5",  "minutes"),
    (k2, "RMSE",         "163.6", "minutes"),
    (k3, "Baseline MAE", "92.01", "mean predictor"),
    (k4, "R² Score",     "−0.05", "stochastic delay"),
    (k5, "Dataset",      "703K",  "trigger rows"),
]
for col, label, val, sub in kpi_data:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value'>{val}</div>
          <div class='metric-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════
if run_btn or "results" in st.session_state:

    stay_id = int(stay_input)

    with st.spinner("Loading data and model..."):
        df, eligible = load_data()
        model, scaler = load_model()

    if stay_id not in eligible:
        st.warning(
            f"⚠️ Stay ID **{stay_id}** not found or has fewer "
            f"than {SEQ_LEN} triggers."
        )
        stay_id = int(eligible[0])
        st.info(f"Using closest valid stay: **{stay_id}**")
    else:
        st.success(f"✅ Stay ID **{stay_id}** loaded successfully")

    stay_df = df[df["stay_id"] == stay_id].reset_index(drop=True)

    stay_delay = stay_df.dropna(subset=["delay_hours"])
    if len(stay_delay) <= SEQ_LEN:
        st.caption(
            f"ℹ️ Only {len(stay_delay)} rows with delay — "
            "using full stay for sequence context"
        )
        working_df = stay_df.copy()
    else:
        working_df = stay_delay.copy()

    # ── Run predictions ────────────────────────────────
    results = []
    if len(working_df) > SEQ_LEN:
        for i in range(SEQ_LEN, len(working_df)):
            row = working_df.iloc[i]
            if pd.isna(row.get("delay_hours")):
                continue
            seq_df        = working_df.iloc[i - SEQ_LEN: i]
            predicted_hrs = predict_delay(seq_df, model, scaler)
            actual_hrs    = float(row["delay_hours"])
            deviation     = actual_hrs - predicted_hrs
            sev_label, sev_color = get_severity(deviation)
            ttype  = row["trigger_type"]
            tcolor = TRIGGER_COLOR.get(ttype, "#fff")
            results.append({
                "time":      str(row["trigger_time"])[:16],
                "type":      ttype,
                "tcolor":    tcolor,
                "predicted": predicted_hrs,
                "actual":    actual_hrs,
                "deviation": deviation,
                "severity":  sev_label,
                "sev_color": sev_color,
                "HR":        row.get("HR",  0),
                "MAP":       row.get("MAP", 0),
                "RR":        row.get("RR",  0),
                "SPO2":      row.get("SPO2",0),
            })

    latest = working_df.iloc[-1].to_dict()

    # ──────────────────────────────────────────────────
    # TABS
    # ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺 Patient View",
        "📊 Model Analytics",
        "📈 Training History",
        "🤖 Agent Pipeline",
    ])

    # ══════════════════════════════════════════════════
    # TAB 1 — PATIENT VIEW
    # ══════════════════════════════════════════════════
    with tab1:
        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            st.markdown("<div class='section-head'>Vital Signs</div>",
                        unsafe_allow_html=True)

            v1, v2, v3, v4 = st.columns(4)
            vital_data = [
                (v1, "HR",   "bpm",  "❤️",  latest.get("HR",   "—"), 60,  100, "#ff7043"),
                (v2, "MAP",  "mmHg", "💉",  latest.get("MAP",  "—"), 70,  105, "#00c8f0"),
                (v3, "RR",   "/min", "🫁",  latest.get("RR",   "—"), 12,  20,  "#f0c040"),
                (v4, "SPO2", "%",    "🩸",  latest.get("SPO2", "—"), 95,  100, "#00e8a0"),
            ]
            for col, name, unit, icon, val, lo, hi, color in vital_data:
                with col:
                    try:
                        fval = float(val)
                        abnormal  = fval < lo or fval > hi
                        color_use = color if abnormal else TEXT_PRI
                        flag      = " ⚠" if abnormal else ""
                    except Exception:
                        fval, color_use, flag = None, TEXT_PRI, ""

                    st.markdown(f"""
                    <div class='metric-card'>
                      <div class='metric-label'>{icon} {name}</div>
                      <div class='metric-value' style='color:{color_use}'>
                        {round(fval,1) if fval is not None else '—'}
                        <span style='font-size:13px;color:#5880a8'> {unit}</span>
                        <span style='font-size:12px'>{flag}</span>
                      </div>
                      <div class='metric-sub'>Normal: {lo}–{hi} {unit}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(
                "<div class='section-head'>Trigger Timeline — Delay (hrs)</div>",
                unsafe_allow_html=True)

            trig_df = working_df[
                ["trigger_time", "trigger_type", "delay_hours",
                 "HR", "MAP", "RR", "SPO2"]
            ].copy()
            trig_df["trigger_time"] = trig_df["trigger_time"].astype(str)
            trig_df["delay_hours"]  = trig_df["delay_hours"].round(2)

            fig_tl = go.Figure()
            for ttype, color in TRIGGER_COLOR.items():
                sub = trig_df[trig_df["trigger_type"] == ttype]
                if sub.empty:
                    continue
                fig_tl.add_trace(go.Scatter(
                    x=sub["trigger_time"],
                    y=sub["delay_hours"],
                    mode="markers+lines",
                    name=ttype,
                    marker=dict(size=8, color=color,
                                line=dict(width=1, color="#000")),
                    line=dict(color=color, width=1.5, dash="dot"),
                    hovertemplate=(
                        "<b>%{x}</b><br>Delay: %{y:.2f} hr<br><extra></extra>"
                    ),
                ))
            fig_tl.update_layout(
                **plotly_base(),
                height=220,
                legend=dict(orientation="h", y=1.15,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                xaxis=dict(showgrid=False, color=TEXT_SEC, showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor=BORDER,
                           color=TEXT_SEC, title="Delay (hrs)"),
            )
            st.plotly_chart(fig_tl, use_container_width=True)

            st.markdown(
                "<div class='section-head'>Vitals Over Time</div>",
                unsafe_allow_html=True)

            fig_vitals = go.Figure()
            for vname, vcolor, visible in [
                ("HR",   "#ff7043", True),
                ("SPO2", "#00e8a0", True),
                ("MAP",  "#00c8f0", False),
                ("RR",   "#f0c040", False),
            ]:
                if vname not in working_df.columns:
                    continue
                fig_vitals.add_trace(go.Scatter(
                    x=working_df["trigger_time"].astype(str),
                    y=working_df[vname],
                    name=vname,
                    mode="lines",
                    line=dict(color=vcolor, width=1.5),
                    visible=True if visible else "legendonly",
                    hovertemplate=f"<b>{vname}</b>: %{{y:.1f}}<extra></extra>",
                ))
            fig_vitals.update_layout(
                **plotly_base(),
                height=200,
                legend=dict(orientation="h", y=1.15,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                xaxis=dict(showgrid=False, color=TEXT_SEC, showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor=BORDER, color=TEXT_SEC),
            )
            st.plotly_chart(fig_vitals, use_container_width=True)

        with col_right:
            st.markdown(
                "<div class='section-head'>Vital Normalisation Radar</div>",
                unsafe_allow_html=True)

            radar_cats = ["HR", "MAP", "RR", "SPO2"]
            radar_norm = {
                "HR":   (latest.get("HR",   80) - 30)  / (220 - 30),
                "MAP":  (latest.get("MAP",  80) - 30)  / (150 - 30),
                "RR":   (latest.get("RR",   15) - 5)   / (60  - 5),
                "SPO2": (latest.get("SPO2", 97) - 70)  / (100 - 70),
            }
            radar_vals  = [radar_norm[c] for c in radar_cats]
            normal_vals = [
                (80 - 30) / (220 - 30),
                (87 - 30) / (150 - 30),
                (16 - 5)  / (60  - 5),
                (98 - 70) / (100 - 70),
            ]
            cats_closed = radar_cats + [radar_cats[0]]
            vals_closed  = radar_vals  + [radar_vals[0]]
            norm_closed  = normal_vals + [normal_vals[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_closed, theta=cats_closed,
                fill="toself",
                fillcolor="rgba(0,200,240,0.08)",
                line=dict(color=ACCENT, width=1, dash="dash"),
                name="Normal range",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill="toself",
                fillcolor="rgba(255,112,67,0.15)",
                line=dict(color="#ff7043", width=2),
                name="Patient",
            ))
            fig_radar.update_layout(
                **plotly_base(),
                height=270,
                polar=dict(
                    bgcolor=BG_CARD,
                    radialaxis=dict(visible=True, range=[0, 1],
                                   gridcolor=BORDER, color=TEXT_SEC,
                                   tickfont=dict(size=8)),
                    angularaxis=dict(color=TEXT_PRI, tickfont=dict(size=10)),
                ),
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown(
                "<div class='section-head'>Dataset Trigger Distribution</div>",
                unsafe_allow_html=True)

            dist_data = pd.DataFrame({
                "Trigger":         ["MAP_LOW", "RR_HIGH", "HR_HIGH", "SPO2_LOW"],
                "Count":           [282409,    229823,    125101,    66475],
                "Mean Delay (hr)": [2.56,      2.80,      2.11,      3.41],
            })
            fig_dist = go.Figure(go.Bar(
                y=dist_data["Trigger"],
                x=dist_data["Count"],
                orientation="h",
                marker_color=[
                    TRIGGER_COLOR["MAP_LOW"], TRIGGER_COLOR["RR_HIGH"],
                    TRIGGER_COLOR["HR_HIGH"], TRIGGER_COLOR["SPO2_LOW"],
                ],
                text=dist_data["Count"].apply(lambda x: f"{x/1000:.0f}K"),
                textposition="outside",
                textfont=dict(size=10, color="#a8c8e8"),
                hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
            ))
            fig_dist.update_layout(
                **plotly_base(),
                height=200,
                xaxis=dict(showgrid=False, visible=False),
                yaxis=dict(showgrid=False, color="#a8c8e8"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # ══════════════════════════════════════════════════
    # TAB 2 — MODEL ANALYTICS
    # ══════════════════════════════════════════════════
    with tab2:
        if not results:
            st.warning("Run analysis first to see model analytics.")
        else:
            res_df = pd.DataFrame(results)

            r1c1, r1c2 = st.columns(2)

            with r1c1:
                st.markdown(
                    "<div class='section-head'>Predicted vs Actual Delay</div>",
                    unsafe_allow_html=True)

                max_val = max(res_df["predicted"].max(), res_df["actual"].max()) + 0.5
                fig_scat = go.Figure()
                fig_scat.add_trace(go.Scatter(
                    x=[0, max_val], y=[0, max_val],
                    mode="lines",
                    line=dict(color=ACCENT, width=1, dash="dash"),
                    name="Perfect prediction",
                    hoverinfo="skip",
                ))
                for sev, scolor in [
                    ("No Delay", "#00e8a0"), ("Mild", "#f0c040"),
                    ("Moderate", "#ff7043"), ("Severe", "#ff2050"),
                ]:
                    sub = res_df[res_df["severity"] == sev]
                    if sub.empty:
                        continue
                    fig_scat.add_trace(go.Scatter(
                        x=sub["predicted"], y=sub["actual"],
                        mode="markers", name=sev,
                        marker=dict(size=9, color=scolor,
                                    line=dict(width=0.5, color="#000")),
                        hovertemplate=(
                            f"<b>{sev}</b><br>"
                            "Pred: %{x:.2f} hr<br>"
                            "Actual: %{y:.2f} hr<extra></extra>"
                        ),
                    ))
                fig_scat.update_layout(
                    **plotly_base(), height=300,
                    xaxis=dict(title="Predicted (hr)", showgrid=True,
                               gridcolor=BORDER, color=TEXT_SEC),
                    yaxis=dict(title="Actual (hr)", showgrid=True,
                               gridcolor=BORDER, color=TEXT_SEC),
                    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_scat, use_container_width=True)

            # ── Violin — FIXED fillcolor ───────────────
            with r1c2:
                st.markdown(
                    "<div class='section-head'>Deviation Distribution by Trigger</div>",
                    unsafe_allow_html=True)

                fig_vio = go.Figure()
                for ttype, tcolor in TRIGGER_COLOR.items():
                    sub = res_df[res_df["type"] == ttype]
                    if len(sub) < 2:
                        continue
                    fig_vio.add_trace(go.Violin(
                        y=sub["deviation"],
                        name=ttype,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=VIOLIN_FILL.get(ttype, "rgba(128,128,128,0.2)"),
                        line_color=tcolor,
                        points="all",
                        pointpos=0,
                        marker=dict(size=5, color=tcolor, opacity=0.6),
                    ))
                fig_vio.add_hline(
                    y=0, line_dash="dash",
                    line_color=ACCENT, line_width=1,
                )
                fig_vio.update_layout(
                    **plotly_base(), height=300,
                    violinmode="group",
                    yaxis=dict(title="Deviation (hr)", showgrid=True,
                               gridcolor=BORDER, color=TEXT_SEC),
                    xaxis=dict(showgrid=False, color=TEXT_SEC),
                    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_vio, use_container_width=True)

            r2c1, r2c2 = st.columns(2)

            # ── Histogram — FIXED marker_color ────────
            with r2c1:
                st.markdown(
                    "<div class='section-head'>Delay Distribution</div>",
                    unsafe_allow_html=True)

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=res_df["actual"],
                    name="Actual",
                    marker_color="rgba(0,200,240,0.53)",
                    xbins=dict(size=0.5),
                    hovertemplate="Delay: %{x:.1f}hr<br>Count: %{y}<extra></extra>",
                ))
                fig_hist.add_trace(go.Histogram(
                    x=res_df["predicted"],
                    name="Predicted",
                    marker_color="rgba(255,112,67,0.53)",
                    xbins=dict(size=0.5),
                    hovertemplate="Delay: %{x:.1f}hr<br>Count: %{y}<extra></extra>",
                ))
                fig_hist.update_layout(
                    **plotly_base(), height=260,
                    barmode="overlay",
                    xaxis=dict(title="Delay (hr)", showgrid=True,
                               gridcolor=BORDER, color=TEXT_SEC),
                    yaxis=dict(title="Count", showgrid=True,
                               gridcolor=BORDER, color=TEXT_SEC),
                    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with r2c2:
                st.markdown(
                    "<div class='section-head'>Severity Breakdown</div>",
                    unsafe_allow_html=True)

                sev_counts = res_df["severity"].value_counts()
                fig_pie = go.Figure(go.Pie(
                    labels=sev_counts.index,
                    values=sev_counts.values,
                    hole=0.5,
                    marker=dict(
                        colors=[
                            {"No Delay": "#00e8a0", "Mild": "#f0c040",
                             "Moderate": "#ff7043", "Severe": "#ff2050"}
                            .get(s, ACCENT)
                            for s in sev_counts.index
                        ],
                        line=dict(color=BG_CARD, width=2)
                    ),
                    textfont=dict(size=10, color=TEXT_PRI),
                    hovertemplate=(
                        "<b>%{label}</b><br>%{value} triggers"
                        " (%{percent})<extra></extra>"
                    ),
                ))
                fig_pie.update_layout(
                    **plotly_base(), height=260,
                    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown(
                "<div class='section-head'>Feature Correlation (this stay)</div>",
                unsafe_allow_html=True)

            corr_cols = [c for c in
                         ["HR", "MAP", "RR", "SPO2", "hour_of_day", "time_gap_min"]
                         if c in working_df.columns]
            if len(corr_cols) >= 3:
                corr_mat = working_df[corr_cols].corr().round(2)
                fig_heat = go.Figure(go.Heatmap(
                    z=corr_mat.values,
                    x=corr_cols, y=corr_cols,
                    colorscale=[[0, "#ff2050"], [0.5, BG_PANEL], [1, ACCENT]],
                    zmin=-1, zmax=1,
                    text=corr_mat.values,
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                    hovertemplate="%{x} × %{y}<br>r = %{z:.2f}<extra></extra>",
                ))
                fig_heat.update_layout(
                    **plotly_base(), height=280,
                    xaxis=dict(color=TEXT_SEC, tickfont=dict(size=9)),
                    yaxis=dict(color=TEXT_SEC, tickfont=dict(size=9)),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

    # ══════════════════════════════════════════════════
    # TAB 3 — TRAINING HISTORY
    # ══════════════════════════════════════════════════
    with tab3:
        st.markdown(
            "<div class='section-head'>Loss Curves (actual training log)</div>",
            unsafe_allow_html=True)

        epochs     = list(range(1, 19))
        train_loss = [
            1.3735, 1.1184, 1.0803, 1.0502, 1.0303,
            1.0108, 1.0002, 0.9900, 0.9844, 0.9788,
            0.9720, 0.9642, 0.9579, 0.9510, 0.9415,
            0.9322, 0.9228, 0.9103,
        ]
        val_loss = [
            0.9956, 0.9939, 0.9904, 0.9881, 0.9936,
            0.9870, 0.9843, 0.9911, 0.9904, 0.9839,
            0.9867, 0.9815, 0.9863, 0.9871, 0.9858,
            1.0031, 0.9979, 1.0102,
        ]
        best_epoch = val_loss.index(min(val_loss)) + 1

        tc1, tc2 = st.columns([3, 1])
        with tc1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=train_loss, name="Train Loss",
                line=dict(color=ACCENT, width=2),
                mode="lines+markers", marker=dict(size=5),
                hovertemplate="Epoch %{x}<br>Train: %{y:.4f}<extra></extra>",
            ))
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=val_loss, name="Val Loss",
                line=dict(color="#f0c040", width=2),
                mode="lines+markers", marker=dict(size=5),
                hovertemplate="Epoch %{x}<br>Val: %{y:.4f}<extra></extra>",
            ))
            fig_loss.add_trace(go.Scatter(
                x=epochs + epochs[::-1],
                y=val_loss + train_loss[::-1],
                fill="toself",
                fillcolor="rgba(255,112,67,0.06)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=False, name="gap",
            ))
            fig_loss.add_vline(
                x=best_epoch, line_dash="dash",
                line_color="#00e8a0", line_width=1.5,
                annotation_text=f"best (ep {best_epoch})",
                annotation_font_color="#00e8a0",
                annotation_font_size=10,
            )
            fig_loss.update_layout(
                **plotly_base(), height=300,
                legend=dict(orientation="h", y=1.12,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                xaxis=dict(showgrid=False, color=TEXT_SEC,
                           title="Epoch", dtick=1),
                yaxis=dict(showgrid=True, gridcolor=BORDER,
                           color=TEXT_SEC, title="MSE Loss"),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with tc2:
            st.markdown(f"""
            <div class='metric-card' style='margin-top:10px'>
              <div class='metric-label'>Best Val Loss</div>
              <div class='metric-value' style='font-size:22px'>{min(val_loss):.4f}</div>
              <div class='metric-sub'>epoch {best_epoch} / {len(epochs)}</div>
            </div>
            <div class='metric-card'>
              <div class='metric-label'>Early Stop</div>
              <div class='metric-value' style='font-size:22px'>Epoch {len(epochs)}</div>
              <div class='metric-sub'>patience = 6</div>
            </div>
            <div class='metric-card'>
              <div class='metric-label'>Improvement</div>
              <div class='metric-value' style='font-size:22px;color:#00e8a0'>2.0%</div>
              <div class='metric-sub'>over baseline</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(
            "<div class='section-head'>Epoch-by-epoch log</div>",
            unsafe_allow_html=True)

        epoch_df = pd.DataFrame({
            "Epoch":      epochs,
            "Train Loss": [f"{v:.4f}" for v in train_loss],
            "Val Loss":   [f"{v:.4f}" for v in val_loss],
            "Best":       ["✅" if i + 1 == best_epoch else ""
                           for i in range(len(epochs))],
        })
        st.dataframe(epoch_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════
    # TAB 4 — AGENT PIPELINE
    # ══════════════════════════════════════════════════
    with tab4:
        if not results:
            st.warning("No predictions available.")
        else:
            agent_mode = "Reactive" if mode == "⚡ Reactive" else "Proactive"
            ac1, ac2 = st.columns([3, 2])

            with ac1:
                st.markdown(
                    "<div class='section-head'>Agent Pipeline Results</div>",
                    unsafe_allow_html=True)

                for r in results[:10]:
                    st.markdown(f"""
                    <div class='agent-step'>
                      <div class='agent-label'>
                        ⏱ {r['time']} &nbsp;
                        <span class='trigger-badge'
                          style='background:{r["tcolor"]}22;
                                 color:{r["tcolor"]};
                                 border:1px solid {r["tcolor"]}44'>
                          {r['type']}
                        </span>
                      </div>
                      <div class='agent-val'>
                        Predicted: <b>{r['predicted']:.2f} hr</b>
                        &nbsp;·&nbsp;
                        Actual: <b>{r['actual']:.2f} hr</b>
                      </div>
                      <div style='margin-top:6px'>
                        Deviation:
                        <span style='color:{r["sev_color"]};font-weight:700'>
                          {r['deviation']:+.2f} hr
                        </span>
                        &nbsp;&nbsp;
                        <span class='severity-badge'
                          style='background:{r["sev_color"]}18;
                                 color:{r["sev_color"]};
                                 border:1px solid {r["sev_color"]}44;
                                 font-size:10px;padding:3px 10px'>
                          {r['severity'].upper()}
                        </span>
                      </div>
                    </div>""", unsafe_allow_html=True)

            with ac2:
                st.markdown(
                    "<div class='section-head'>Severity Summary</div>",
                    unsafe_allow_html=True)

                sev_counts = pd.Series(
                    [r["severity"] for r in results]
                ).value_counts().reindex(
                    ["No Delay", "Mild", "Moderate", "Severe"], fill_value=0
                )
                fig_sev = go.Figure(go.Bar(
                    x=sev_counts.index,
                    y=sev_counts.values,
                    marker_color=["#00e8a0", "#f0c040", "#ff7043", "#ff2050"],
                    text=sev_counts.values,
                    textposition="outside",
                    textfont=dict(color="#a8c8e8", size=11),
                    hovertemplate="<b>%{x}</b><br>%{y} triggers<extra></extra>",
                ))
                fig_sev.update_layout(
                    **plotly_base(), height=200,
                    xaxis=dict(showgrid=False, color="#a8c8e8"),
                    yaxis=dict(showgrid=True, gridcolor=BORDER, color=TEXT_SEC),
                )
                st.plotly_chart(fig_sev, use_container_width=True)

                st.markdown(
                    "<div class='section-head'>System Alert</div>",
                    unsafe_allow_html=True)

                worst = max(results, key=lambda x: x["deviation"])
                wsev, wcol = worst["severity"], worst["sev_color"]
                icon = {"Severe": "🚨", "Moderate": "⚠️",
                        "Mild": "💛", "No Delay": "✅"}.get(wsev, "ℹ️")

                st.markdown(f"""
                <div style='background:{wcol}0d;
                     border:1px solid {wcol}33;
                     border-radius:10px;padding:14px 16px;margin-top:4px'>
                  <div style='font-size:11px;color:{wcol};
                       font-weight:700;letter-spacing:1px;margin-bottom:6px'>
                    {icon} {agent_mode.upper()} AGENT — {wsev.upper()}
                  </div>
                  <div style='font-size:11px;color:#a8c8e8;line-height:1.6'>
                    Worst trigger: <b>{worst['type']}</b> at {worst['time']}<br>
                    Predicted <b>{worst['predicted']:.2f} hr</b>
                    → Actual <b>{worst['actual']:.2f} hr</b><br>
                    Deviation:
                    <b style='color:{wcol}'>{worst['deviation']:+.2f} hr</b>
                  </div>
                </div>""", unsafe_allow_html=True)

                st.markdown(
                    "<div class='section-head'>Deviation Over Time</div>",
                    unsafe_allow_html=True)

                res_df2 = pd.DataFrame(results)
                fig_dev = go.Figure()
                fig_dev.add_hline(y=0, line_dash="dash",
                                  line_color=ACCENT, line_width=1)
                fig_dev.add_trace(go.Scatter(
                    x=res_df2["time"],
                    y=res_df2["deviation"],
                    mode="lines+markers",
                    line=dict(color="#f0c040", width=1.5),
                    marker=dict(size=7,
                                color=res_df2["sev_color"].tolist(),
                                line=dict(width=0.5, color="#000")),
                    hovertemplate="%{x}<br>Dev: %{y:+.2f} hr<extra></extra>",
                ))
                fig_dev.update_layout(
                    **plotly_base(), height=180,
                    xaxis=dict(showgrid=False, color=TEXT_SEC, showticklabels=False),
                    yaxis=dict(showgrid=True, gridcolor=BORDER,
                               color=TEXT_SEC, title="Dev (hr)"),
                    showlegend=False,
                )
                st.plotly_chart(fig_dev, use_container_width=True)

# ══════════════════════════════════════════════════════
# LANDING STATE
# ══════════════════════════════════════════════════════
else:
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;
         background:#080f1a;border:1px solid #162436;
         border-radius:12px;'>
      <div style='font-size:48px;margin-bottom:16px'>🏥</div>
      <div style='font-size:18px;color:#e8f4ff;font-weight:700;margin-bottom:8px'>
        Select a Patient Stay and Run Analysis
      </div>
      <div style='font-size:12px;color:#5880a8;
           max-width:400px;margin:0 auto;line-height:1.7'>
        Use the sidebar to select a Stay ID from the dropdown
        or type any Stay ID from your dataset.<br><br>
        The agent pipeline will run your actual LSTM model
        and display real predictions across 4 tabs.
      </div>
      <div style='margin-top:24px;font-size:11px;color:#2a4060'>
        Known good stays: 30000831 · 30002654
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-head'>Dataset Overview</div>",
                unsafe_allow_html=True)

    o1, o2, o3, o4 = st.columns(4)
    for col, ttype, count, color in [
        (o1, "MAP_LOW",  "282,409", "#00c8f0"),
        (o2, "RR_HIGH",  "229,823", "#f0c040"),
        (o3, "HR_HIGH",  "125,101", "#ff7043"),
        (o4, "SPO2_LOW", "66,475",  "#00e8a0"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-top:2px solid {color}'>
              <div class='metric-label'>{ttype}</div>
              <div class='metric-value' style='font-size:22px;color:{color}'>{count}</div>
              <div class='metric-sub'>triggers</div>
            </div>""", unsafe_allow_html=True)