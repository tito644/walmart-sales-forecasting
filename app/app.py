"""
app.py  —  Walmart Weekly Sales Forecast
=========================================
Professional Streamlit app for the end-to-end forecasting project.

Run:
    streamlit run app/app.py

Requires:
    1. data/processed/walmart_clean.csv   (produced by notebook 01)
    2. models/xgb_sales_model.pkl         (produced by train_model.py or notebook 03)
    3. models/model_features.pkl
    4. models/model_metrics.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Forecast",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background-color: #0f1117;
}
[data-testid="stSidebar"] {
    background-color: #1a1d27;
    border-right: 1px solid #2a2d3e;
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 60%, #1565c0 100%);
    padding: 2rem 2.5rem 1.6rem;
    border-radius: 12px;
    margin-bottom: 1.8rem;
    border: 1px solid #1e3a8a;
}
.app-header h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #93c5fd;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Metric cards ── */
.metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.6rem;
}
.metric-card .label {
    color: #7c8db5;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    color: #e8eaf6;
    font-size: 1.55rem;
    font-weight: 700;
}
.metric-card .delta-pos { color: #4ade80; font-size: 0.85rem; }
.metric-card .delta-neg { color: #f87171; font-size: 0.85rem; }

/* ── Prediction box ── */
.pred-box {
    background: linear-gradient(135deg, #0a2744 0%, #0d3460 100%);
    border: 2px solid #1d6fa4;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.pred-box .pred-label {
    color: #93c5fd;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.pred-box .pred-value {
    color: #ffffff;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
}
.pred-box .pred-delta {
    margin-top: 0.5rem;
    font-size: 1rem;
    font-weight: 600;
}

/* ── Section titles ── */
.section-title {
    color: #93c5fd;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid #2a2d3e;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── Info pill ── */
.info-pill {
    display: inline-block;
    background: #1e2d4a;
    border: 1px solid #2d4a6e;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    color: #7db3e0;
    margin: 0.15rem;
}

/* ── Feature table ── */
.feat-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.feat-table th {
    background: #1e2535;
    color: #7c8db5;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid #2a2d3e;
    text-align: left;
}
.feat-table td {
    color: #c5cae9;
    padding: 0.45rem 0.8rem;
    border-bottom: 1px solid #1e2535;
}
.feat-table tr:last-child td { border-bottom: none; }

/* ── Divider ── */
hr { border-color: #2a2d3e !important; }

/* ── Sidebar section headers ── */
.sidebar-section {
    color: #5c7cfa;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1.2rem 0 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "xgb_sales_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")
METRICS_PATH  = os.path.join(BASE_DIR, "models", "model_metrics.pkl")
DATA_PATH     = os.path.join(BASE_DIR, "data", "processed", "walmart_clean.csv")

# Week mid-point per month (ISO week, from actual dataset analysis)
MONTH_WEEK_MAP = {1:2, 2:6, 3:10, 4:15, 5:19, 6:23, 7:28, 8:32, 9:37, 10:41, 11:45, 12:50}
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}


# ── Load resources ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

@st.cache_data
def load_metrics():
    try:
        return joblib.load(METRICS_PATH)
    except Exception:
        return None

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


# ── Check model exists before anything else ───────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.markdown("""
    <div class="app-header">
      <h1>🛒 Walmart Weekly Sales Forecast</h1>
      <p>Model not found — follow the steps below to train it.</p>
    </div>
    """, unsafe_allow_html=True)
    st.error("⚠️  `models/xgb_sales_model.pkl` not found.")
    st.markdown("""
    Run the training script once from the project root:
    ```bash
    python train_model.py
    ```
    Then refresh this page.
    """)
    st.stop()

model, FEATURES = load_model()
metrics = load_metrics()
df = load_data()
stores = sorted(df["Store"].unique())


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🛒 Walmart Weekly Sales Forecast</h1>
  <p>
    Predict next week's sales for any of 45 stores using a trained XGBoost model —
    explore holiday effects, seasonal patterns, and economic conditions interactively.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">Store</div>', unsafe_allow_html=True)
    store_id = st.selectbox("Store ID", options=stores, label_visibility="collapsed")

    store_df = df[df["Store"] == store_id].copy()
    last = store_df.iloc[-1]

    st.markdown('<div class="sidebar-section">Forecast Conditions</div>', unsafe_allow_html=True)

    is_holiday = st.selectbox(
        "Holiday week?",
        options=["No", "Yes"],
        help="Super Bowl, Labor Day, Thanksgiving, or Christmas week"
    )
    month = st.select_slider(
        "Month",
        options=list(range(1, 13)),
        value=int(last["Month"]),
        format_func=lambda m: MONTH_NAMES[m],
    )

    st.markdown('<div class="sidebar-section">Economic Indicators</div>', unsafe_allow_html=True)
    temperature  = st.slider("Temperature (°F)",  -10, 110, int(last["Temperature"]))
    fuel_price   = st.slider("Fuel Price ($)",     2.0, 5.0, float(round(last["Fuel_Price"], 2)), 0.05)
    cpi          = st.number_input("CPI",           value=float(round(last["CPI"], 2)), step=1.0)
    unemployment = st.number_input("Unemployment Rate (%)", value=float(round(last["Unemployment"], 2)), step=0.1)

    st.markdown("---")
    st.caption("Sliders pre-filled with this store's latest values. Adjust to explore what-if scenarios.")


# ── Build prediction input ────────────────────────────────────────────────────
lag_1 = last["Weekly_Sales"]
lag_52_df = store_df[store_df["Date"] <= last["Date"] - pd.Timedelta(weeks=52)]
lag_52 = lag_52_df.iloc[-1]["Weekly_Sales"] if len(lag_52_df) > 0 else lag_1
rolling_mean_4 = store_df.tail(4)["Weekly_Sales"].mean()
week = MONTH_WEEK_MAP[month]

input_row = pd.DataFrame([{
    "Store": store_id, "Month": month, "Week": week,
    "Holiday_Flag": 1 if is_holiday == "Yes" else 0,
    "Temperature": temperature, "Fuel_Price": fuel_price,
    "CPI": cpi, "Unemployment": unemployment,
    "lag_1": lag_1, "lag_52": lag_52, "rolling_mean_4": rolling_mean_4,
}])[FEATURES]

prediction     = model.predict(input_row)[0]
store_avg      = store_df["Weekly_Sales"].mean()
store_max      = store_df["Weekly_Sales"].max()
delta_pct      = (prediction - store_avg) / store_avg * 100
delta_sign     = "+" if delta_pct >= 0 else ""
delta_class    = "delta-pos" if delta_pct >= 0 else "delta-neg"
pct_of_max     = prediction / store_max * 100


# ── Main layout: 3 columns ───────────────────────────────────────────────────
left, mid, right = st.columns([1.3, 1, 1], gap="large")

# ── LEFT — store metrics ──────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-title">Store Profile</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Average Weekly Sales</div>
        <div class="value">${store_avg:,.0f}</div>
    </div>
    <div class="metric-card">
        <div class="label">Last Known Weekly Sales</div>
        <div class="value">${last["Weekly_Sales"]:,.0f}</div>
    </div>
    <div class="metric-card">
        <div class="label">Peak Weekly Sales (all time)</div>
        <div class="value">${store_max:,.0f}</div>
    </div>
    <div class="metric-card">
        <div class="label">Last Data Point</div>
        <div class="value" style="font-size:1.1rem">{last["Date"].strftime("%b %d, %Y")}</div>
    </div>
    """, unsafe_allow_html=True)

    # Model quality info
    if metrics:
        st.markdown('<div class="section-title" style="margin-top:1.2rem">Model Quality (Test Set)</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <span class="info-pill">RMSE ${metrics["rmse"]:,.0f}</span>
        <span class="info-pill">MAE ${metrics["mae"]:,.0f}</span>
        <span class="info-pill">MAPE {metrics["mape"]:.2f}%</span>
        """, unsafe_allow_html=True)
        st.caption("Evaluated on a held-out 12-week test set using time-based split.")


# ── MID — prediction ─────────────────────────────────────────────────────────
with mid:
    st.markdown('<div class="section-title">Forecast Result</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pred-box">
        <div class="pred-label">Predicted Weekly Sales</div>
        <div class="pred-value">${prediction:,.0f}</div>
        <div class="pred-delta">
            <span class="{delta_class}">{delta_sign}{delta_pct:.1f}% vs store avg</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Gauge-style progress bar
    st.caption(f"As % of this store's all-time peak (${store_max:,.0f})")
    st.progress(min(pct_of_max / 100, 1.0))
    st.caption(f"{pct_of_max:.1f}% of peak")

    # Context note
    holiday_note = "🎉 Holiday week — expect a sales lift." if is_holiday == "Yes" else ""
    month_note   = "🎄 December: typically the strongest month." if month == 12 else \
                   "❄️ January: typically the weakest month." if month == 1 else ""
    for note in [holiday_note, month_note]:
        if note:
            st.info(note)

    # Feature transparency
    with st.expander("See features sent to the model"):
        feat_dict = input_row.iloc[0].to_dict()
        rows_html = "".join(
            f"<tr><td>{k}</td><td><b>{v:,.4f}</b></td></tr>"
            if isinstance(v, float) else
            f"<tr><td>{k}</td><td><b>{int(v)}</b></td></tr>"
            for k, v in feat_dict.items()
        )
        st.markdown(
            f'<table class="feat-table"><tr><th>Feature</th><th>Value</th></tr>'
            f'{rows_html}</table>',
            unsafe_allow_html=True,
        )


# ── RIGHT — sales history chart ───────────────────────────────────────────────
with right:
    st.markdown('<div class="section-title">Sales History (last 26 weeks)</div>',
                unsafe_allow_html=True)

    recent = store_df.tail(26).copy()
    holiday_weeks = recent[recent["Holiday_Flag"] == 1]

    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=recent["Date"], y=recent["Weekly_Sales"],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=5, color="#3b82f6"),
        name="Weekly Sales",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    # Holiday markers
    if len(holiday_weeks) > 0:
        fig.add_trace(go.Scatter(
            x=holiday_weeks["Date"], y=holiday_weeks["Weekly_Sales"],
            mode="markers",
            marker=dict(size=10, color="#f59e0b", symbol="star"),
            name="Holiday Week",
            hovertemplate="<b>Holiday Week</b><br>$%{y:,.0f}<extra></extra>",
        ))

    # Store average line
    fig.add_hline(
        y=store_avg,
        line_dash="dash", line_color="#64748b", line_width=1.5,
        annotation_text=f"Avg ${store_avg:,.0f}",
        annotation_font_color="#94a3b8",
        annotation_position="bottom right",
    )

    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        showlegend=True,
        legend=dict(
            font=dict(color="#94a3b8", size=11),
            bgcolor="rgba(0,0,0,0)",
            orientation="h", y=-0.15,
        ),
        xaxis=dict(
            gridcolor="#1e2535", tickfont=dict(color="#7c8db5", size=10),
            showline=False,
        ),
        yaxis=dict(
            gridcolor="#1e2535", tickfont=dict(color="#7c8db5", size=10),
            tickformat="$,.0f", showline=False,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Yearly seasonal pattern
    st.markdown('<div class="section-title" style="margin-top:0.6rem">Seasonal Pattern (all-time avg)</div>',
                unsafe_allow_html=True)

    monthly_avg = store_df.groupby("Month")["Weekly_Sales"].mean()
    colors = ["#f59e0b" if m == 12 else "#374151" if m == 1 else "#1e3a5f"
              for m in range(1, 13)]

    fig2 = go.Figure(go.Bar(
        x=[MONTH_NAMES[m] for m in range(1, 13)],
        y=[monthly_avg.get(m, 0) for m in range(1, 13)],
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    fig2.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        margin=dict(l=10, r=10, t=10, b=10),
        height=180,
        xaxis=dict(gridcolor="#1e2535", tickfont=dict(color="#7c8db5", size=10)),
        yaxis=dict(gridcolor="#1e2535", tickfont=dict(color="#7c8db5", size=10),
                   tickformat="$,.0f"),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🟡 Dec (peak)   |   Dark bar = Jan (lowest)")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
col_a, col_b = st.columns([3, 1])
with col_a:
    st.caption(
        "**Model:** XGBoost (tuned with GridSearchCV + TimeSeriesSplit)  ·  "
        "**Best params:** learning_rate=0.1, max_depth=5, n_estimators=200  ·  "
        "Evaluated on held-out 12-week test set (time-based split, no data leakage)"
    )
with col_b:
    st.caption("**Tareq** · Data Science Portfolio")
