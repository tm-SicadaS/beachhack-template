import streamlit as st
import numpy as np
import pandas as pd
import time

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Silent Model Failure Detection",
    layout="wide"
)

st.title("Silent Model Failure Detection System")
st.caption("Real-time model monitoring dashboard (simulated data)")

# ===============================
# KPIs
# ===============================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Model Accuracy", "91.2%", "-2.1%")
col2.metric("Avg Confidence", "0.78", "-0.05")
col3.metric("Data Drift Score", "0.32", "+0.12")
col4.metric("Failure Risk", "MEDIUM")

st.divider()

# ===============================
# Generate fake time-series data
# ===============================
time_steps = pd.date_range("2025-01-01", periods=30)

confidence = np.clip(np.random.normal(0.8, 0.05, 30), 0, 1)
accuracy = np.clip(np.random.normal(0.9, 0.04, 30), 0, 1)
drift = np.clip(np.random.normal(0.25, 0.1, 30), 0, 1)

df = pd.DataFrame({
    "Date": time_steps,
    "Confidence": confidence,
    "Accuracy": accuracy,
    "Drift Score": drift
}).set_index("Date")

# ===============================
# Charts
# ===============================
left, right = st.columns(2)

with left:
    st.subheader("Model Confidence Over Time")
    st.line_chart(df["Confidence"])

with right:
    st.subheader("Model Accuracy Trend")
    st.line_chart(df["Accuracy"])

st.subheader("Data Drift Monitoring")
st.area_chart(df["Drift Score"])

st.divider()

# ===============================
# Anomaly Alert
# ===============================
st.subheader("Anomaly Detection")

if drift[-1] > 0.4:
    st.error("⚠️ High data drift detected! Model performance may degrade.")
else:
    st.success("✅ Model operating within safe thresholds.")

# ===============================
# Live Prediction Feed (SAFE)
# ===============================
st.subheader("Live Prediction Feed")

if st.button("Generate Live Predictions"):
    placeholder = st.empty()
    for _ in range(3):
        placeholder.info(
            f"Prediction confidence: {round(np.random.uniform(0.6, 0.95), 2)}"
        )
        time.sleep(0.5)
