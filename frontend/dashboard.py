import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Vantara Retail - Customer Intelligence Platform",
    page_icon="🛍️",
    layout="wide"
)

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

# Header Banner
st.title("🛍️ Vantara Customer Intelligence & Churn Platform")
st.markdown("### AI-Powered Retention, Customer Lifetime Value & Behavioral Segmentation")
st.markdown("---")

# Top KPI Summary Cards
col1, col2, col3, col4 = st.columns(4)
total_customers = len(df)
overall_churn_rate = df["is_churn"].mean() * 100
total_historical_spend = df["monetary_value"].sum()
avg_clv = df["future_clv"].mean()

col1.metric("👥 Total Active Customers", f"{total_customers:,}")
col2.metric("⚠️ Predicted Churn Rate", f"{overall_churn_rate:.1f}%")
col3.metric("💳 Historical Revenue", f"£{total_historical_spend:,.0f}")
col4.metric("📈 Avg Expected CLV", f"£{avg_clv:,.2f}")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 Customer Segments", "🚨 Churn Risk Leaderboard", "🔍 Individual Customer XAI"])

# TAB 1: Customer Segments
with tab1:
    st.subheader("Customer Behavioral Segmentation (K-Means & RFM)")
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        segment_counts = df["customer_segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Count"]
        fig_pie = px.pie(
            segment_counts, names="Segment", values="Count",
            title="Customer Base Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b:
        fig_scatter = px.scatter(
            df.sample(min(1500, len(df)), random_state=42),
            x="recency", y="monetary_value", color="customer_segment",
            size="frequency", hover_data=["customer_id", "engagement_score"],
            log_y=True,
            title="Behavioral RFM Space (Recency vs Log-Monetary Spend)",
            labels={"recency": "Days Since Last Order", "monetary_value": "Total Spend (£)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# TAB 2: Churn Risk Leaderboard
with tab2:
    st.subheader("High-Priority Retention Leaderboard")
    st.caption("Prioritizing customers with High Historical Spend and High Predicted Churn Risk (PRD Section 12).")

    selected_segment = st.selectbox(
        "Filter by Segment:",
        ["All Segments"] + list(df["customer_segment"].dropna().unique())
    )

    filtered_df = df if selected_segment == "All Segments" else df[df["customer_segment"] == selected_segment]

    leaderboard = filtered_df.sort_values(by=["churn_probability", "monetary_value"], ascending=[False, False])[
        ["customer_id", "customer_segment", "churn_probability", "monetary_value", "future_clv", "recency", "frequency"]
    ].head(50)

    # Clean display
    leaderboard_display = leaderboard.copy()
    leaderboard_display["churn_probability"] = (leaderboard_display["churn_probability"] * 100).round(1).astype(str) + "%"
    leaderboard_display["monetary_value"] = "£" + leaderboard_display["monetary_value"].round(2).astype(str)
    leaderboard_display["future_clv"] = "£" + leaderboard_display["future_clv"].round(2).astype(str)

    st.dataframe(leaderboard_display, use_container_width=True)

# TAB 3: Individual Customer XAI Search
with tab3:
    st.subheader("🔍 Individual Customer Behavioral Explainability (SHAP)")
    search_id = st.number_input("Enter Customer ID to Inspect:", min_value=int(df["customer_id"].min()), max_value=int(df["customer_id"].max()), value=14799)

    matched = df[df["customer_id"] == search_id]
    if matched.empty:
        st.warning("Customer ID not found in database.")
    else:
        cust = matched.iloc[0]
        prob = cust["churn_probability"]
        risk_color = "red" if prob >= 0.65 else ("green" if prob <= 0.35 else "orange")

        st.markdown(f"### Profile: Customer #{search_id} | Segment: **{cust['customer_segment']}**")
        st.markdown(f"**Predicted Churn Probability:** :{risk_color}[**{prob*100:.1f}%**]")

        # Run SHAP on selected customer
        single_X = matched[feature_cols]
        shap_vals = explainer(single_X).values[0]

        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "Value": single_X.iloc[0].values,
            "SHAP_Impact": shap_vals
        }).sort_values(by="SHAP_Impact", key=abs, ascending=True)

        fig_bar = px.bar(
            shap_df.tail(8),
            x="SHAP_Impact", y="Feature", orientation="h",
            title="Top Factors Influencing This Prediction (SHAP Values)",
            color="SHAP_Impact",
            color_continuous_scale="RdYlGn_r"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.info(
            f"💡 **Plain Language AI Summary:** Customer #{search_id} is classified under the **{cust['customer_segment']}** profile. "
            f"The primary driver impacting churn probability is their **{shap_df.iloc[-1]['Feature']}** (Value: {shap_df.iloc[-1]['Value']:.2f})."
        )