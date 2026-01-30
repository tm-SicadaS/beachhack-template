import streamlit as st
import numpy as np
import pandas as pd
import time
import psutil
from datetime import datetime

# ===============================
# CONFIGURATION
# ===============================
MAX_HISTORY_LENGTH = 100
CPU_THRESHOLD_MEDIUM = 75
CPU_THRESHOLD_HIGH = 90
MEMORY_THRESHOLD_MEDIUM = 80
MEMORY_THRESHOLD_HIGH = 90

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Silent Model Failure Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# HELPER FUNCTIONS
# ===============================
@st.cache_data(ttl=1)
def collect_metrics():
    """Collect real-time system metrics with error handling."""
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        
        # Cross-platform load average handling
        if hasattr(psutil, "getloadavg"):
            load = np.mean(psutil.getloadavg())
        else:
            load = cpu / 100
        
        # Calculate confidence score (inverse relationship with CPU usage)
        confidence = np.clip(1 - (cpu / 120), 0.4, 0.95)
        
        return cpu, memory, disk, load, confidence
    except Exception as e:
        st.error(f"Error collecting metrics: {e}")
        return 0, 0, 0, 0, 0.5

def calculate_risk_level(cpu, memory):
    """Determine system risk level based on CPU and memory usage."""
    if cpu > CPU_THRESHOLD_HIGH or memory > MEMORY_THRESHOLD_HIGH:
        return "HIGH"
    elif cpu > CPU_THRESHOLD_MEDIUM or memory > MEMORY_THRESHOLD_MEDIUM:
        return "MEDIUM"
    else:
        return "LOW"

def get_risk_color(risk):
    """Return color based on risk level."""
    colors = {
        "LOW": "🟢",
        "MEDIUM": "🟡",
        "HIGH": "🔴"
    }
    return colors.get(risk, "⚪")

# ===============================
# SESSION STATE INITIALIZATION
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = True

# ===============================
# HEADER
# ===============================
st.title("🔍 Silent Model Failure Detection System")
st.caption("Real-time AI system behavior monitoring (no ground-truth required)")

# ===============================
# COLLECT AND STORE METRICS
# ===============================
cpu, mem, disk, load, conf = collect_metrics()

# Add new metrics to history
st.session_state.history.append({
    "CPU %": cpu,
    "Memory %": mem,
    "Disk %": disk,
    "Load": load,
    "Confidence": conf,
    "Time": datetime.now().strftime("%H:%M:%S"),
    "Timestamp": datetime.now()
})

# Limit history length to prevent memory issues
if len(st.session_state.history) > MAX_HISTORY_LENGTH:
    st.session_state.history = st.session_state.history[-MAX_HISTORY_LENGTH:]

# Convert to DataFrame
df = pd.DataFrame(st.session_state.history)

# ===============================
# CONTROL BUTTONS
# ===============================
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
with col_btn1:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

with col_btn2:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.divider()

# ===============================
# KPI METRICS
# ===============================
risk = calculate_risk_level(cpu, mem)
risk_icon = get_risk_color(risk)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "CPU Usage", 
        f"{cpu:.1f}%",
        delta=f"{cpu - df['CPU %'].iloc[-2]:.1f}%" if len(df) > 1 else None
    )

with col2:
    st.metric(
        "Memory Usage", 
        f"{mem:.1f}%",
        delta=f"{mem - df['Memory %'].iloc[-2]:.1f}%" if len(df) > 1 else None
    )

with col3:
    st.metric(
        "Confidence Score", 
        f"{conf:.2f}",
        delta=f"{conf - df['Confidence'].iloc[-2]:.2f}" if len(df) > 1 else None,
        delta_color="normal"
    )

with col4:
    st.metric(
        "Failure Risk", 
        f"{risk_icon} {risk}"
    )

st.divider()

# ===============================
# CHARTS
# ===============================
if len(df) > 1:
    left, right = st.columns(2)
    
    with left:
        st.subheader("📈 Model Confidence Trend")
        st.line_chart(
            df.set_index("Time")["Confidence"],
            use_container_width=True
        )
    
    with right:
        st.subheader("💻 System Load Trend")
        st.line_chart(
            df.set_index("Time")[["CPU %", "Memory %"]],
            use_container_width=True
        )
    
    st.subheader("💾 Resource Drift (Disk Usage)")
    st.area_chart(
        df.set_index("Time")["Disk %"],
        use_container_width=True
    )
else:
    st.info("Collecting initial data... Please refresh in a moment.")

st.divider()

# ===============================
# ANOMALY ALERT
# ===============================
st.subheader("🚨 Anomaly Detection")

if risk == "HIGH":
    st.error("🚨 **HIGH ALERT:** High system stress detected. Silent model failure likely.")
    st.write("**Recommended Actions:**")
    st.write("- Check for resource-intensive processes")
    st.write("- Review recent model deployments")
    st.write("- Consider scaling resources")
elif risk == "MEDIUM":
    st.warning("⚠️ **WARNING:** Early signs of degradation detected.")
    st.write("**Recommended Actions:**")
    st.write("- Monitor system closely")
    st.write("- Prepare for potential intervention")
else:
    st.success("✅ **NORMAL:** System operating within stable baseline.")

st.divider()

# ===============================
# DETAILED METRICS TABLE
# ===============================
with st.expander("📊 Detailed Metrics History"):
    st.dataframe(
        df[["Time", "CPU %", "Memory %", "Disk %", "Confidence"]].tail(20),
        use_container_width=True,
        hide_index=True
    )

# ===============================
# SYSTEM STATISTICS
# ===============================
with st.expander("📈 Statistical Summary"):
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.metric("Avg CPU", f"{df['CPU %'].mean():.1f}%")
        st.metric("Max CPU", f"{df['CPU %'].max():.1f}%")
    
    with col_stat2:
        st.metric("Avg Memory", f"{df['Memory %'].mean():.1f}%")
        st.metric("Max Memory", f"{df['Memory %'].max():.1f}%")
    
    with col_stat3:
        st.metric("Avg Confidence", f"{df['Confidence'].mean():.2f}")
        st.metric("Min Confidence", f"{df['Confidence'].min():.2f}")

# ===============================
# AUTO-REFRESH
# ===============================
st.divider()
st.caption("💡 Tip: Click 'Refresh' to update metrics, or use Streamlit's auto-rerun feature for continuous monitoring.")
