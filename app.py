import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# ==========================================
# 1. 页面基础配置 (Page Setup)
# ==========================================
st.set_page_config(
    page_title="Group 14: SDXL Advanced Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #0E1117; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .metric-container { background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; text-align: center; }
    .highlight-val { font-size: 28px; font-weight: bold; color: #FF4B4B; }
    .success-val { font-size: 28px; font-weight: bold; color: #2ECC71; }
    .info-val { font-size: 24px; font-weight: bold; color: #3498DB; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据加载与工程计算模块
# ==========================================
@st.cache_data
def load_and_process_data():
    paths = ["D:/B_simulation_results_flat.csv", "B_simulation_results_flat.csv"]
    file_path = None
    for p in paths:
        if os.path.exists(p):
            file_path = p
            break

    if not file_path:
        return None

    try:
        df = pd.read_csv(file_path)

        # 基础清洗
        df["capacity_int"] = df["capacity"].astype(int)
        df["capacity"] = df["capacity"].astype(str)

        policy_map = {
            "high": "Static High (FCFS)",
            "fast": "Static Fast (FCFS)",
            "smart": "Adaptive (Threshold)",
            "sjf": "SJF (Shortest Job First)",
            "smart_as": "Auto-Scaling",
        }
        col_to_map = "base_policy" if "base_policy" in df.columns else "policy_key"
        df["Policy Name"] = df[col_to_map].map(policy_map).fillna(df[col_to_map])

        # 工程指标计算
        GPU_HOURLY_COST = 0.35
        df["hourly_cost_usd"] = df["capacity_int"] * GPU_HOURLY_COST
        df["hourly_throughput"] = df["throughput_req_s"] * 3600
        df["cost_per_1k_req"] = df.apply(
            lambda x: (x["hourly_cost_usd"] / x["hourly_throughput"] * 1000) if x["hourly_throughput"] > 0 else np.nan,
            axis=1,
        )

        # ✅ Little's Law 单位修正：L = λ * W(seconds)
        # 如果 lambda 是 req/s，则 avg_latency_ms 必须转换为 seconds
        df["est_system_load"] = df["lambda"] * (df["avg_latency_ms"] / 1000.0)

        return df
    except Exception as e:
        st.error(f"Error parsing CSV or calculating metrics: {e}")
        return None

df = load_and_process_data()

# ==========================================
# 3. 侧边栏控制台 (Sidebar Controls)
# ==========================================
if df is not None:
    st.sidebar.title("🎛️ Control Panel")

    st.sidebar.subheader("1. View Settings")

    use_log_scale = st.sidebar.checkbox(
        "Use Log Scale (Y-Axis)",
        value=True,
        help="Turn on to see clear differences between high and low values (e.g. 10ms vs 10000ms)."
    )

    st.sidebar.subheader("2. Filter Configurations")

    all_policies = sorted(df["Policy Name"].unique())
    default_selection = ["SJF (Shortest Job First)", "Adaptive (Threshold)", "Static High (FCFS)"]
    selected_policies = st.sidebar.multiselect(
        "Select Policies:",
        options=all_policies,
        default=[p for p in default_selection if p in all_policies],
    )

    all_caps = sorted(df["capacity"].unique(), key=lambda x: int(x))
    selected_caps = st.sidebar.multiselect(
        "Select Server Capacity:",
        options=all_caps,
        default=all_caps,
    )

    filtered_df = df[
        (df["Policy Name"].isin(selected_policies)) &
        (df["capacity"].isin(selected_caps))
    ].copy()

    # ✅ log scale 防护：log 轴不接受 <= 0
    if use_log_scale:
        for c in ["p99_latency_ms", "avg_latency_ms", "cost_per_1k_req", "est_system_load"]:
            if c in filtered_df.columns:
                filtered_df.loc[filtered_df[c] <= 0, c] = np.nan

    st.sidebar.markdown("---")
    st.sidebar.info("**Group 14 Project Dashboard**")
else:
    st.error("⚠️ Data file not found. Please place 'B_simulation_results_flat.csv' in D:/ or the app folder.")
    st.stop()

# ==========================================
# 4. 配色方案定义 (Custom Colors)
# ==========================================
color_map = {
    "Static High (FCFS)": "#D62728",
    "SJF (Shortest Job First)": "#2CA02C",
    "Static Fast (FCFS)": "#1F77B4",
    "Adaptive (Threshold)": "#FF7F0E",
    "Auto-Scaling": "#9467BD",
}

# ==========================================
# 5. 主面板可视化 (Main Dashboard)
# ==========================================
st.markdown('<div class="main-header">🧠 SDXL Service Performance Analysis</div>', unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("Please select at least one Policy and Capacity.")
    st.stop()

# --- 第一行：核心延迟指标 ---
st.subheader("1. System Latency & Stability")
col1, col2 = st.columns(2)

with col1:
    fig_p99 = px.line(
        filtered_df,
        x="lambda",
        y="p99_latency_ms",
        color="Policy Name",
        line_dash="capacity",
        markers=True,
        title="<b>P99 Latency</b> (Tail Latency)",
        color_discrete_map=color_map,
        log_y=use_log_scale,
        labels={"lambda": "Arrival Rate (λ)", "p99_latency_ms": "Latency (ms)", "capacity": "GPUs"},
    )
    fig_p99.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_p99, use_container_width=True)

with col2:
    fig_avg = px.line(
        filtered_df,
        x="lambda",
        y="avg_latency_ms",
        color="Policy Name",
        line_dash="capacity",
        markers=True,
        title="<b>Average Latency</b>",
        color_discrete_map=color_map,
        log_y=use_log_scale,
        labels={"lambda": "Arrival Rate (λ)", "avg_latency_ms": "Latency (ms)", "capacity": "GPUs"},
    )
    fig_avg.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_avg, use_container_width=True)

# --- 第二行：成本与拥塞 ---
st.markdown("---")
st.subheader("2. Advanced Analysis: Cost & Load")
col3, col4 = st.columns(2)

with col3:
    st.markdown("##### 💰 Cost Efficiency Analysis")
    fig_cost = px.line(
        filtered_df,
        x="lambda",
        y="cost_per_1k_req",
        color="Policy Name",
        line_dash="capacity",
        markers=True,
        title="<b>Cost per 1,000 Requests ($)</b>",
        color_discrete_map=color_map,
        log_y=use_log_scale,
        labels={"lambda": "Arrival Rate (λ)", "cost_per_1k_req": "Cost ($) / 1k Reqs", "capacity": "GPUs"},
    )
    fig_cost.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    st.caption("Assumption: T4 GPU cost ~$0.35/hr. Lower is better.")

with col4:
    st.markdown("##### 🚦 System Congestion (Queue Depth)")
    fig_load = px.line(
        filtered_df,
        x="lambda",
        y="est_system_load",
        color="Policy Name",
        line_dash="capacity",
        markers=True,
        title="<b>Est. Requests in System</b> (Little's Law)",
        color_discrete_map=color_map,
        log_y=use_log_scale,
        labels={"lambda": "Arrival Rate (λ)", "est_system_load": "Avg Requests Count", "capacity": "GPUs"},
    )
    fig_load.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_load, use_container_width=True)
    st.caption("Calculated using Little's Law (L = λW). Indicates congestion level.")

# --- 第三行：吞吐量与质量 ---
st.markdown("---")
st.subheader("3. Throughput & Quality")
col5, col6 = st.columns(2)

with col5:
    fig_thr = px.line(
        filtered_df,
        x="lambda",
        y="throughput_req_s",
        color="Policy Name",
        line_dash="capacity",
        markers=True,
        title="<b>System Throughput</b>",
        color_discrete_map=color_map,
        labels={"lambda": "Arrival Rate (λ)", "throughput_req_s": "Reqs / sec", "capacity": "GPUs"},
    )
    fig_thr.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_thr, use_container_width=True)

with col6:
    fig_qual = px.line(
        filtered_df,
        x="lambda",
        y="avg_quality",
        color="Policy Name",
        line_dash="capacity",
        markers=True,
        title="<b>Average Quality</b> (CLIP Score)",
        color_discrete_map=color_map,
        labels={"lambda": "Arrival Rate (λ)", "avg_quality": "CLIP Score"},
    )
    fig_qual.update_yaxes(range=[30, 34])
    fig_qual.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_qual, use_container_width=True)

# ==========================================
# 6. 关键洞察卡片 (Insights)
# ==========================================
st.markdown("---")
st.subheader("💡 Key Engineering Conclusion")

try:
    # ✅ Insight 基于当前筛选（filtered_df），避免和用户筛选不一致
    max_lambda = filtered_df["lambda"].max()
    df_insight = filtered_df[(filtered_df["lambda"] == max_lambda) & (filtered_df["capacity"] == "1")]

    if not df_insight.empty:
        high_row = df_insight[df_insight["Policy Name"] == "Static High (FCFS)"]
        sjf_row = df_insight[df_insight["Policy Name"] == "SJF (Shortest Job First)"]

        if not high_row.empty and not sjf_row.empty:
            # ✅ 用 P99 作为核心结论（更符合“稳定性/尾延迟”故事线）
            high_p99 = high_row["p99_latency_ms"].values[0]
            sjf_p99 = sjf_row["p99_latency_ms"].values[0]

            high_cost = high_row["cost_per_1k_req"].values[0]
            sjf_cost = sjf_row["cost_per_1k_req"].values[0]

            speedup = (high_p99 / sjf_p99) if (sjf_p99 and sjf_p99 > 0) else np.nan
            cost_saving = ((high_cost - sjf_cost) / high_cost * 100) if (high_cost and high_cost > 0) else np.nan

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-container">
                    <div>Baseline P99 (Static High)</div>
                    <div class="highlight-val">{high_p99/1000:,.1f} s</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-container" style="border-color: #2ECC71;">
                    <div>Optimized P99 (SJF)</div>
                    <div class="success-val">{sjf_p99/1000:,.1f} s</div>
                </div>""", unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-container" style="background-color: #e8f5e9;">
                    <div>Efficiency Gain</div>
                    <div class="info-val">⚡ {speedup:,.1f}x Faster</div>
                    <div style="font-size: 0.9rem; color: #555;">💰 {cost_saving:,.1f}% Cost Reduction</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Select both 'Static High (FCFS)' and 'SJF (Shortest Job First)' to see the insight card.")
    else:
        st.info("No data available for insight card under the current filters (try including capacity=1).")
except Exception as e:
    st.info(f"Insufficient data for auto-insight. ({e})")

st.markdown("---")
st.caption("© 2025 Group 14 | Data-Driven Simulation Dashboard")
