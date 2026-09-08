import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Telecom Customer Intelligence", layout="wide")

# 1. Load Scored Dataset
@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/scored_customers.parquet")

df = load_data()

# 2. Sidebar Filters
st.sidebar.title("Filter Controls")
selected_segment = st.sidebar.selectbox("Customer Segment", ["All"] + sorted(df["segment"].unique().tolist()))
selected_risk = st.sidebar.selectbox("Risk Band", ["All", "Critical", "High", "Medium", "Low"])

filtered_df = df.copy()
if selected_segment != "All":
    filtered_df = filtered_df[filtered_df["segment"] == selected_segment]
if selected_risk != "All":
    filtered_df = filtered_df[filtered_df["risk_band"] == selected_risk]

# 3. Main Dashboard Layout
st.title("Telecom Customer Intelligence Platform")
st.markdown("End-to-end churn prediction, risk monetization, and behavioural segmentation dashboard.")

# Top-level Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(filtered_df))
col2.metric("Overall Churn Rate", f"{filtered_df['churn_flag'].mean():.1%}")
col3.metric("Total Annual Revenue", f"${filtered_df['annual_revenue_exposure'].sum():,.0f}")
col4.metric("Revenue at Risk", f"${filtered_df['revenue_at_risk'].sum():,.0f}")

st.divider()

# 4. Visualizations
c1, c2 = st.columns(2)

with c1:
    st.subheader("Revenue at Risk by Risk Band")
    risk_summary = filtered_df.groupby("risk_band")["revenue_at_risk"].sum().reset_index()
    fig_risk = px.bar(risk_summary, x="risk_band", y="revenue_at_risk", color="risk_band", title="Revenue Exposure per Risk Tier")
    st.plotly_chart(fig_risk, use_container_width=True)

with c2:
    st.subheader("Customer Segments (PCA 2D Projection)")
    fig_pca = px.scatter(
        filtered_df, x="pca_x", y="pca_y", color=filtered_df["segment"].astype(str),
        hover_data=["customerID", "MonthlyCharges", "tenure", "churn_probability"],
        title="K-Means Customer Clusters Visualized"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

# 5. Retention Action Table
st.subheader("High-Priority Retention Target List")
st.markdown("Top customers ranked by churn probability for immediate marketing outreach.")
display_cols = ["customerID", "tenure", "MonthlyCharges", "contract", "churn_probability", "risk_band", "segment"]
available_cols = [c for c in display_cols if c in filtered_df.columns]
st.dataframe(
    filtered_df.sort_values("churn_probability", ascending=False)[available_cols].head(25),
    use_container_width=True
)