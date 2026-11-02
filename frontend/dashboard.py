import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Vantara AI | Customer Intelligence Nexus",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Futuristic Cyberpunk / Glassmorphic Custom CSS
FUTURISTIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background canvas */
.stApp {
    background: radial-gradient(circle at 10% 10%, #0d1527 0%, #030712 100%);
    color: #e2e8f0;
}

/* Top Banner Title */
.nexus-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 22px 30px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 0 16px rgba(56, 189, 248, 0.08);
    backdrop-filter: blur(14px);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.nexus-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.nexus-sub {
    font-size: 0.9rem;
    color: #94a3b8;
    letter-spacing: 0.3px;
    margin: 0;
}

.system-status {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: #34d399;
    display: flex;
    align-items: center;
    gap: 8px;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
}

/* Futuristic KPI Metric Card */
.kpi-container {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(10px);
}
.kpi-container:hover {
    border-color: rgba(56, 189, 248, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 12px 28px -6px rgba(56, 189, 248, 0.2);
}
.kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-tag {
    font-size: 0.72rem;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-block;
    margin-top: 6px;
    font-weight: 600;
}
.tag-cyan { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
.tag-amber { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
.tag-emerald { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
.tag-violet { background: rgba(192, 132, 252, 0.15); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.3); }

/* Custom Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: transparent;
    border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    padding-bottom: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 10px;
    padding: 8px 20px;
    color: #94a3b8;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%) !important;
    border-color: #38bdf8 !important;
    color: #f8fafc !important;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.25);
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    overflow: hidden;
}
</style>
"""
st.markdown(FUTURISTIC_CSS, unsafe_allow_html=True)

ARTIFACTS_DIR = Path("models_artifacts")
DATA_PATH = Path("data/processed/customer_features.parquet")


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    with open(ARTIFACTS_DIR / "xgboost_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "features.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open(ARTIFACTS_DIR / "shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    
    if "churn_probability" not in df.columns:
        df["churn_probability"] = model.predict_proba(df[feature_cols])[:, 1]
    return df, model, feature_cols, explainer


df, model, feature_cols, explainer = load_data()

# Hero Header with Live Telemetry Badge
st.markdown("""
<div class="nexus-header">
    <div>
        <div class="nexus-title">⚡ VANTARA // CUSTOMER INTELLIGENCE NEXUS</div>
        <div class="nexus-sub">Next-Gen Predictive Behavioral Analytics • Churn Mitigation Core • TreeSHAP Explainability</div>
    </div>
    <div class="system-status">
        <div class="pulse-dot"></div>
        <span>CORE ONLINE // LATENCY 14ms</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Top Metrics Grid
total_customers = len(df)
overall_churn_rate = df["is_churn"].mean() * 100
total_historical_spend = df["monetary_value"].sum()
avg_clv = df["future_clv"].mean()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Active Profiles</div>
        <div class="kpi-value">{total_customers:,}</div>
        <span class="kpi-tag tag-cyan">Telemetry Active</span>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">System Churn Rate</div>
        <div class="kpi-value">{overall_churn_rate:.1f}%</div>
        <span class="kpi-tag tag-amber">Model Ingestion</span>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Aggregated Revenue</div>
        <div class="kpi-value">£{total_historical_spend/1e6:.2f}M</div>
        <span class="kpi-tag tag-emerald">Historical Value</span>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Expected CLV</div>
        <div class="kpi-value">£{avg_clv:,.2f}</div>
        <span class="kpi-tag tag-violet">Horizon: 90-Day</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🪐 Cluster Topography",
    "🚨 Real-time Churn Radar",
    "🧬 Explainability Engine"
])

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(15, 23, 42, 0.45)",
    plot_bgcolor="rgba(15, 23, 42, 0.45)",
    font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
    yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)")
)

# TAB 1: Clusters
with tab1:
    st.markdown("#### 🌐 Customer Cohort Behavioral Topography")
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        segment_counts = df["customer_segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Count"]
        
        fig_pie = px.pie(
            segment_counts, names="Segment", values="Count",
            hole=0.58,
            color_discrete_sequence=["#38bdf8", "#818cf8", "#f43f5e", "#fbbf24"]
        )
        fig_pie.update_layout(
            **PLOT_LAYOUT, 
            showlegend=True,
            annotations=[dict(text=f"<b>{total_customers:,}</b><br><span style='font-size:10px;color:#94a3b8;'>PROFILES</span>", x=0.5, y=0.5, font_size=16, font_color="#f8fafc", showarrow=False)]
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent', marker=dict(line=dict(color='#030712', width=2)))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b:
        sample_df = df.sample(min(1500, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample_df,
            x="recency", y="monetary_value", color="customer_segment",
            size="frequency", hover_data=["customer_id", "engagement_score"],
            log_y=True,
            color_discrete_sequence=["#38bdf8", "#818cf8", "#f43f5e", "#fbbf24"],
            labels={"recency": "Days Elapsed Since Order", "monetary_value": "Log Monetary Spend (£)"}
        )
        fig_scatter.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig_scatter, use_container_width=True)

# TAB 2: Radar
with tab2:
    st.markdown("#### 📡 At-Risk Customer Priority Matrix")
    
    selected_segment = st.selectbox(
        "Segment Filtering Protocol:",
        ["All Segments"] + list(df["customer_segment"].dropna().unique())
    )

    filtered_df = df if selected_segment == "All Segments" else df[df["customer_segment"] == selected_segment]

    leaderboard = filtered_df.sort_values(by=["churn_probability", "monetary_value"], ascending=[False, False])[
        ["customer_id", "customer_segment", "churn_probability", "monetary_value", "future_clv", "recency", "frequency"]
    ].head(50).copy()

    # Stylized risk representation
    leaderboard["churn_probability"] = leaderboard["churn_probability"].apply(
        lambda p: f"🔴 {p*100:.1f}% Critical" if p >= 0.70 else (f"🟡 {p*100:.1f}% Moderate" if p >= 0.40 else f"🟢 {p*100:.1f}% Low")
    )
    leaderboard["monetary_value"] = leaderboard["monetary_value"].apply(lambda v: f"£{v:,.2f}")
    leaderboard["future_clv"] = leaderboard["future_clv"].apply(lambda v: f"£{v:,.2f}")

    st.dataframe(leaderboard, use_container_width=True, height=450)

# TAB 3: XAI
with tab3:
    st.markdown("#### 🧬 Neural Diagnostics & Feature Attribution")
    
    col_input, col_meta = st.columns([1, 2])
    with col_input:
        search_id = st.number_input(
            "Target Customer Identifier:",
            min_value=int(df["customer_id"].min()),
            max_value=int(df["customer_id"].max()),
            value=14799
        )

    matched = df[df["customer_id"] == search_id]
    if matched.empty:
        st.warning("Customer ID not found in database.")
    else:
        cust = matched.iloc[0]
        prob = cust["churn_probability"]
        
        status_color = "#f43f5e" if prob >= 0.65 else ("#10b981" if prob <= 0.35 else "#fbbf24")
        risk_label = "HIGH RISK OF ATTRITION" if prob >= 0.65 else ("RETAINED & STABLE" if prob <= 0.35 else "MODERATE ATTENTION")

        with col_meta:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid {status_color}; border-radius: 12px; padding: 12px 20px; display: inline-block;">
                <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 600;">CLASSIFICATION:</span> 
                <b style="color: {status_color}; font-size: 1.1rem; margin-left: 8px;">{risk_label} ({prob*100:.1f}%)</b>
                <span style="margin-left: 20px; color: #94a3b8; font-size: 0.85rem;">SEGMENT:</span>
                <b style="color: #38bdf8; margin-left: 8px;">{cust['customer_segment']}</b>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        single_X = matched[feature_cols]
        shap_vals = explainer(single_X).values[0]

        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "Value": single_X.iloc[0].values,
            "SHAP_Impact": shap_vals
        }).sort_values(by="SHAP_Impact", key=abs, ascending=True)

        colors = ['#f43f5e' if v > 0 else '#10b981' for v in shap_df.tail(8)["SHAP_Impact"]]

        fig_bar = go.Figure(go.Bar(
            x=shap_df.tail(8)["SHAP_Impact"],
            y=shap_df.tail(8)["Feature"],
            orientation='h',
            marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.2)', width=1))
        ))
        
        bar_layout = dict(**PLOT_LAYOUT)
        bar_layout["title"] = "Local Feature Contributions (Red = Driving Churn, Green = Driving Retention)"
        fig_bar.update_layout(bar_layout)
        st.plotly_chart(fig_bar, use_container_width=True)

        top_feature = shap_df.iloc[-1]
        driver_type = "pushing toward churn" if top_feature['SHAP_Impact'] > 0 else "supporting customer retention"
        
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.65); border-left: 4px solid #38bdf8; padding: 14px 20px; border-radius: 8px; margin-top: 10px;">
            <span style="color: #38bdf8; font-weight: 700;">Diagnostic Insight:</span>
            Customer #{search_id} exhibits a <b>{prob*100:.1f}%</b> probability of churning. The strongest influencing variable is 
            <code>{top_feature['Feature']}</code> (Value: <b>{top_feature['Value']:.2f}</b>), which is heavily {driver_type}.
        </div>
        """, unsafe_allow_html=True)