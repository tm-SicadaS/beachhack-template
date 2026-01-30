import streamlit as st
import numpy as np
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
import logging

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Silent Guard",
    layout="wide"
)

st_autorefresh(interval=700, key="monitor_tick")

# ===============================
# LOGGING
# ===============================
logging.basicConfig(
    filename="silentguard.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# SESSION STATE
# ===============================
if "confidence_log" not in st.session_state:
    st.session_state.confidence_log = []

if "last_status" not in st.session_state:
    st.session_state.last_status = "INITIALIZING"

if "stable_counter" not in st.session_state:
    st.session_state.stable_counter = 0

if "baseline_mature" not in st.session_state:
    st.session_state.baseline_mature = False

if "post_baseline_ticks" not in st.session_state:
    st.session_state.post_baseline_ticks = 0

# ===============================
# MODEL INFERENCE
# ===============================
def production_predict(step: int) -> float:
    if step < 80:
        X = np.random.normal(0.0, 1.0, (1, 5))
    elif step < 120:
        X = np.random.normal(-0.6, 1.2, (1, 5))
    else:
        X = np.random.normal(-1.2, 1.6, (1, 5))

    X = scaler.transform(X)
    prob = model.predict_proba(X)[0][1]
    return float(np.clip(prob, 0.05, 0.95))

# ===============================
# STREAM SAMPLE
# ===============================
step = len(st.session_state.confidence_log)
conf = production_predict(step)
st.session_state.confidence_log.append(conf)

df = pd.DataFrame({"confidence": st.session_state.confidence_log})

# ===============================
# WINDOWS
# ===============================
BASELINE_SIZE = 50
CURRENT_SIZE = 20
MATURITY_WINDOWS = 5

baseline = df.iloc[:BASELINE_SIZE] if len(df) >= BASELINE_SIZE else None
current = df.iloc[-CURRENT_SIZE:] if len(df) >= BASELINE_SIZE + CURRENT_SIZE else None

# ===============================
# METRICS
# ===============================
z_score = 0.0
cur_std = 0.0
base_mean = None
cur_mean = None
new_status = "INITIALIZING"

if baseline is not None and current is not None:
    base_mean = baseline["confidence"].mean()
    base_std = baseline["confidence"].std()
    cur_mean = current["confidence"].mean()
    cur_std = current["confidence"].std()

    if base_std > 0:
        z_score = abs((cur_mean - base_mean) / base_std)

    if z_score > 2.5 or cur_std > 0.22:
        new_status = "CRITICAL"
    elif z_score > 1.8 or cur_std > 0.16:
        new_status = "WARNING"
    else:
        new_status = "NORMAL"

# ===============================
# BASELINE MATURITY LOCK
# ===============================
if baseline is not None and current is not None:
    st.session_state.post_baseline_ticks += 1
    if st.session_state.post_baseline_ticks >= MATURITY_WINDOWS:
        st.session_state.baseline_mature = True

if not st.session_state.baseline_mature:
    status = "NORMAL"
else:
    last = st.session_state.last_status

    if last == "CRITICAL":
        if new_status == "NORMAL":
            st.session_state.stable_counter += 1
            if st.session_state.stable_counter >= 5:
                status = "NORMAL"
                st.session_state.last_status = "NORMAL"
                st.session_state.stable_counter = 0
            else:
                status = "CRITICAL"
        else:
            status = "CRITICAL"
            st.session_state.stable_counter = 0

    elif last == "WARNING":
        if new_status == "CRITICAL":
            status = "CRITICAL"
            st.session_state.last_status = "CRITICAL"
            st.session_state.stable_counter = 0
        elif new_status == "NORMAL":
            st.session_state.stable_counter += 1
            if st.session_state.stable_counter >= 3:
                status = "NORMAL"
                st.session_state.last_status = "NORMAL"
                st.session_state.stable_counter = 0
            else:
                status = "WARNING"
        else:
            status = "WARNING"
            st.session_state.stable_counter = 0

    else:
        status = new_status
        st.session_state.last_status = new_status
        st.session_state.stable_counter = 0

# ===============================
# LOG ENTRY
# ===============================
logging.info(
    f"step={step} | conf={conf:.3f} | z={z_score:.2f} | std={cur_std:.2f} | status={status}"
)

# ===============================
# ALERT (TOP)
# ===============================
if status == "CRITICAL":
    st.error("🚨 Silent model failure detected. Immediate action required.")
elif status == "WARNING":
    st.warning("⚠️ Early model degradation detected.")
elif status == "NORMAL":
    st.success("✅ Model behavior within expected range.")
else:
    st.info("⏳ Learning baseline behavior…")

# ===============================
# HEADER
# ===============================
st.title("Silent Guard")
st.caption("Behavior-based production monitoring without ground-truth labels")

# ===============================
# KPIs
# ===============================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Baseline Confidence", f"{base_mean:.2f}" if base_mean else "—")
c2.metric("Current Confidence", f"{cur_mean:.2f}" if cur_mean else "—")
c3.metric("Output Volatility", f"{cur_std:.2f}")
c4.metric("Model Health", status)

# ===============================
# BASELINE PROGRESS
# ===============================
REQUIRED = BASELINE_SIZE + CURRENT_SIZE
samples = len(st.session_state.confidence_log)

if not st.session_state.baseline_mature:
    st.progress(min(samples / REQUIRED, 1.0))
    st.caption(f"📊 Establishing stable baseline: {samples}/{REQUIRED}")

# ===============================
# TREND
# ===============================
st.subheader("Model Confidence Trend")
st.line_chart(df.tail(200))
