import streamlit as st
import numpy as np
import pandas as pd
import time

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="API Latency & Error Monitor",
    layout="wide"
)

st.title("API Performance Monitor")
st.caption("Real-time System Latency & Error Tracking (Simulated)")

# ===============================
# 1. SIMULATE DATA (Latency & Errors)
# ===============================
# Create 50 simulated data points
steps = 50
time_index = pd.date_range("2025-01-01", periods=steps, freq="H")

# Simulate Latency (Normal distribution with occasional spikes)
base_latency = np.random.normal(120, 15, steps) # Mean 120ms, SD 15ms
spikes = np.random.choice([0, 200, 400], size=steps, p=[0.9, 0.08, 0.02])
latency_data = np.clip(base_latency + spikes, 20, 1000)

# Simulate Errors (0 = Success, 1 = Error)
# 5% chance of error per call
error_hits = np.random.choice([0, 1], size=steps, p=[0.95, 0.05])
# Calculate rolling error rate (as percentage)
error_rate_pct = pd.Series(error_hits).rolling(window=5).mean().fillna(0) * 100

df = pd.DataFrame({
    "Latency (ms)": latency_data,
    "Error Rate (%)": error_rate_pct
}, index=time_index)

# ===============================
# 2. BASELINE LEARNING (Statistics)
# ===============================
# Calculate the "Normal" baseline from historical data
latency_mean = np.mean(latency_data)
latency_std = np.std(latency_data)
current_latency = latency_data[-1]
current_error = error_rate_pct.iloc[-1]

# Dynamic Threshold: Mean + 2 Standard Deviations
threshold_high = latency_mean + (2 * latency_std)

# ===============================
# 3. KPIs
# ===============================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Current Latency", f"{int(current_latency)} ms", delta=f"{int(current_latency - latency_mean)} ms vs Avg")
col2.metric("Error Rate (5-hr avg)", f"{current_error:.1f}%")
# These metrics prove "Baseline Learning"
col3.metric("Baseline Latency (Mean)", f"{int(latency_mean)} ms")
col4.metric("Volatility (StdDev)", f"±{int(latency_std)} ms")

st.divider()

# ===============================
# 4. CHARTS
# ===============================
left, right = st.columns(2)

with left:
    st.subheader("Latency Over Time (ms)")
    # Add a visual line for the threshold
    chart_data = df[["Latency (ms)"]].copy()
    chart_data["Threshold (+2σ)"] = threshold_high
    st.line_chart(chart_data, color=["#3366cc", "#ff4b4b"]) 
    st.caption(f"Red line indicates 2-Sigma Threshold ({int(threshold_high)} ms)")

with right:
    st.subheader("Error Rate Trend (%)")
    st.area_chart(df["Error Rate (%)"], color="#ff4b4b")

st.divider()

# ===============================
# 5. ANOMALY DETECTION LOGIC
# ===============================
st.subheader("System Health Status")

# Check if current latency exceeds learned baseline threshold
if current_latency > threshold_high:
    st.error(f"⚠️ **Latency Spike Detected!** Current ({int(current_latency)}ms) is > Baseline ({int(threshold_high)}ms)")
elif current_error > 10:
    st.error("⚠️ **High Error Rate Detected!** Errors exceed 10%.")
else:
    st.success(f"✅ **System Healthy.** Latency is within normal range (Mean {int(latency_mean)} ± {int(latency_std*2)}).")

# ===============================
# 6. LIVE SIMULATION
# ===============================
st.subheader("Live Request Simulator")

if st.button("Ping API (Simulate 5 Requests)"):
    status_text = st.empty()
    bar = st.progress(0)
    
    for i in range(1, 6):
        # Simulate a single ping
        sim_latency = np.random.normal(120, 20)
        sim_error = np.random.choice([True, False], p=[0.1, 0.9])
        
        status = "❌ ERROR" if sim_error else "✅ OK"
        status_text.text(f"Request {i}/5: {status} ({int(sim_latency)} ms)")
        bar.progress(i * 20)
        time.sleep(0.3)
    
    status_text.text("Simulation Complete. Metrics updated.")