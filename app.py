import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# ==========================================
# 1) Page Setup
# ==========================================
st.set_page_config(
    page_title="Group 14: SDXL Advanced Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #0E1117; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2) Data Loading + Processing
# ==========================================
def find_csv_path():
    # Streamlit Cloud usually uses current working directory; local dev may use D:
    candidates = [
        "B_simulation_results_flat.csv",
        "./B_simulation_results_flat.csv",
        "D:/B_simulation_results_flat.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner=False)
def load_and_process_data(file_path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # ---- basic columns sanity check ----
    required_cols = ["lambda", "capacity", "avg_latency_ms", "p99_latency_ms", "throughput_req_s", "avg_quality"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # ---- capacity ----
    df["capacity_int"] = df["capacity"].astype(int)
    df["capacity"] = df["capacity_int"].astype(str)

    # ---- policy mapping ----
    policy_map = {
        "high": "Static High (FCFS)",
        "fast": "Static Fast (FCFS)",
        "smart": "Adaptive (Threshold)",
        "sjf": "SJF (Shortest Job First)",
        "smart_as": "Auto-Scaling",
    }
    col_to_map = "base_policy" if "base_policy" in df.columns else ("policy_key" if "policy_key" in df.columns else None)
    if col_to_map is None:
        # fallback: keep a safe string policy name
        df["Policy Name"] = "Unknown Policy"
    else:
        df["Policy Name"] = df[col_to_map].astype(str).map(policy_map).fillna(df[col_to_map].astype(str))

    # ---- engineering metrics ----
    GPU_HOURLY_COST = 0.35
    df["hourly_cost_usd"] = df["capacity_int"] * GPU_HOURLY_COST
    df["hourly_throughput"] = df["throughput_req_s"] * 3600.0

    df["cost_per_1k_req"] = np.where(
        df["hourly_throughput"] > 0,
        (df["hourly_cost_usd"] / df["hourly_throughput"]) * 1000.0,
        np.nan
    )

    # ---- Little’s Law: L = λ * W ----
    # If lambda is req/s and avg_latency_ms is ms, convert W to seconds.
    df["est_system_load"] = df["lambda"] * (df["avg_latency_ms"] / 1000.0)

    return df


# Sidebar: reload button (clears cache)
st.sidebar.title("🎛️ Control Panel")
if st.sidebar.button("🔄 Reload data (clear cache)"):
    st.cache_data.clear()

csv_path = find_csv_path()
if not csv_path:
    st.error("⚠️ Data file not found. Put 'B_simulation_results_flat.csv' in the app folder (same directory).")
    st.stop()

mtime = os.path.getmtime(csv_path)

try:
    df = load_and_process_data(csv_path, mtime)
except Exception as e:
    st.error(f"⚠️ Failed to load/process CSV: {e}")
    st.stop()

# ==========================================
# 3) Sidebar Filters
# ==========================================
st.sidebar.subheader("1) View Settings")
use_log_scale = st.sidebar.checkbox(
    "Use Log Scale (Y-Axis)",
    value=True,
    help="Useful when values span large ranges (e.g., 10ms vs 100000ms)."
)

st.sidebar.subheader("2) Filter Configurations")

all_policies = sorted(df["Policy Name"].unique())
default_selection = [p for p in ["SJF (Shortest Job First)", "Adaptive (Threshold)", "Static High (FCFS)"] if p in all_policies]

selected_policies = st.sidebar.multiselect(
    "Select Policies:",
    options=all_policies,
    default=default_selection if default_selection else all_policies[:3]
)

all_caps = sorted(df["capacity"].unique(), key=lambda x: int(x))
selected_caps = st.sidebar.multiselect(
    "Select Server Capacity:",
    options=all_caps,
    default=all_caps
)

st.sidebar.markdown("---")
st.sidebar.caption("Note: For **Auto-Scaling**, 'capacity' typically means a GPU cap (k_max), not a fixed k.")

filtered_df = df[
    (df["Policy Name"].isin(selected_policies)) &
    (df["capacity"].isin(selected_caps))
].copy()

if filtered_df.empty:
    st.warning("Please select at least one Policy and Capacity.")
    st.stop()

# If log scale, make sure all plotted values are > 0; otherwise Plotly log axis breaks
if use_log_scale:
    for col in ["p99_latency_ms", "avg_latency_ms", "cost_per_1k_req", "est_system_load"]:
        filtered_df.loc[filtered_df[col] <= 0, col] = np.nan

# ==========================================
# 4) Colors
# ==========================================
color_map = {
    "Static High (FCFS)": "#D62728",
    "SJF (Shortest Job First)": "#2CA02C",
    "Static Fast (FCFS)": "#1F77B4",
    "Adaptive (Threshold)": "#FF7F0E",
    "Auto-Scaling": "#9467BD",
}

# ==========================================
# 5) Main Dashboard (Charts Only)
# ==========================================
st.markdown('<div class="main-header">🧠 SDXL Service Performance Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Compare policies under different arrival rates (λ) and server capacities (k)</div>', unsafe_allow_html=True)

# --- Row 1: Latency ---
st.subheader("1) System Latency & Stability")
c1, c2 = st.columns(2)

with c1:
    fig_p99 = px.line(
        filtered_df,
        x="lambda", y="p99_latency_ms",
        color="Policy Name", line_dash="capacity",
        markers=True,
        title="<b>P99 Latency</b> (Tail Latency)",
        color_discrete_map=color_map,
        log_y=use_log_scale,
        labels={"lambda": "Arrival Rate (λ)", "p99_latency_ms": "Latency (ms)", "capacity": "GPUs"}
    )
    fig_p99.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(b=90)
    )
    st.plotly_chart(fig_p99, use_container_width=True)

with c2:
    fig_avg = px.line(
        filtered_df,
        x="lambda", y="avg_latency_ms",
        color="Policy Name", line_dash="capacity",
        markers=True,
        title="<b>Average Latency</b>",
        color_discrete_map=color_map,
        log_y=use_log_scale,
        labels={"lambda": "Arrival Rate (λ)", "avg_latency_ms": "Latency (ms)", "capacity": "GPUs"}
    )
    fig_avg.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(b=90)
    )
    st.plotly_chart(fig_avg, use_container_width=True)

# --- Row 2: Cost & Congestion ---
st.markdown("---")
st.subheader("2) Advanced Analysis: Cost & Load")
c3, c4 = st.columns(2)

with c3:
    fig_cost = px.line(
        filtered_df,
        x="lambda", y="cost_per_1k_req",
        color="Policy Name", line_dash="capacity",
        markers=True,
        title="<b>Cost per 1,000 Requests ($)</b>",
        color_discrete_map=color_map,
        log_y=use_log_scale,
        labels={"lambda": "Arrival Rate (λ)", "cost_per_1k_req": "Cost ($) / 1k req", "capacity": "GPUs"}
    )
    fig_cost.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(b=90)
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    st.caption("Assumption: T4 GPU cost ≈ $0.35/hr. Lower is better.")

with c4:
    fig_load = px.line(
        filtered_df,
        x="lambda", y="est_system_load",
        color="Policy Name", line_dash="capacity",
        markers=True,
        title="<b>Estimated Requests in System</b> (Little’s Law: L = λW)",
        color_discrete_map=color_map,
        log_y=use_log_scale,
        labels={"lambda": "Arrival Rate (λ)", "est_system_load": "Avg # requests (L)", "capacity": "GPUs"}
    )
    fig_load.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(b=90)
    )
    st.plotly_chart(fig_load, use_container_width=True)
    st.caption("Computed using W = avg_latency_ms / 1000. Shows congestion level (higher = more backlog).")

# --- Row 3: Throughput & Quality ---
st.markdown("---")
st.subheader("3) Throughput & Quality")
c5, c6 = st.columns(2)

with c5:
    fig_thr = px.line(
        filtered_df,
        x="lambda", y="throughput_req_s",
        color="Policy Name", line_dash="capacity",
        markers=True,
        title="<b>System Throughput</b>",
        color_discrete_map=color_map,
        labels={"lambda": "Arrival Rate (λ)", "throughput_req_s": "Requests / sec", "capacity": "GPUs"}
    )
    fig_thr.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(b=90)
    )
    st.plotly_chart(fig_thr, use_container_width=True)

with c6:
    fig_qual = px.line(
        filtered_df,
        x="lambda", y="avg_quality",
        color="Policy Name", line_dash="capacity",
        markers=True,
        title="<b>Average Quality</b> (CLIP Score)",
        color_discrete_map=color_map,
        labels={"lambda": "Arrival Rate (λ)", "avg_quality": "CLIP Score", "capacity": "GPUs"}
    )
    # Keep a stable range for easier visual comparison
    fig_qual.update_yaxes(range=[30, 34])
    fig_qual.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(b=90)
    )
    st.plotly_chart(fig_qual, use_container_width=True)

st.markdown("---")
st.caption("© 2025 Group 14 | Data-Driven Simulation Dashboard")
