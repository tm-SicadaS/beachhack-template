import streamlit as st
import numpy as np
import pandas as pd
import pickle
import psutil   
import requests 
import time
import plotly.express as px 

# ===============================
# 1. LOAD AI MODELS
# ===============================
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 CRITICAL ERROR: 'model.pkl' or 'scaler.pkl' not found.")
    st.stop()

# ===============================
# 2. PAGE SETUP
# ===============================
st.set_page_config(page_title="Real-Time AI Monitor", layout="wide")
st.title("Silent Model Failure Detection")

TARGET_URL = "http://www.google.com" 

# ===============================
# 3. SESSION STATE
# ===============================
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Timestamp", "CPU", "Memory", "Latency", "Error_Rate"])
if "error_count" not in st.session_state:
    st.session_state.error_count = 0
if "total_calls" not in st.session_state:
    st.session_state.total_calls = 0

# ===============================
# 4. SIDEBAR
# ===============================
st.sidebar.header("⚙️ Controls")
window_size = st.sidebar.slider("History Window", 10, 200, 50)

if st.sidebar.button("🗑️ Clear Data"):
    st.session_state.history = pd.DataFrame(columns=["Timestamp", "CPU", "Memory", "Latency", "Error_Rate"])
    st.session_state.error_count = 0
    st.session_state.total_calls = 0
    st.sidebar.success("Cleared!")
    st.rerun()

# ===============================
# 5. UI LAYOUT (THE FIX IS HERE)
# ===============================
# We create columns first
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# IMPORTANT: Create empty placeholders inside these columns.
# We will only write to these specific variables inside the loop.
with kpi1:
    cpu_placeholder = st.empty()
with kpi2:
    mem_placeholder = st.empty()
with kpi3:
    lat_placeholder = st.empty()
with kpi4:
    err_placeholder = st.empty()

st.divider()

# Placeholders for the Chart and the Alert message
chart_placeholder = st.empty()
alert_placeholder = st.empty()

start_btn = st.button("🔴 Start Live Monitoring")

# ===============================
# 6. DATA FUNCTION
# ===============================
def fetch_real_data():
    cpu = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory().percent
    
    start = time.time()
    is_error = 0
    try:
        requests.get(TARGET_URL, timeout=1) 
    except:
        is_error = 1 
    latency = (time.time() - start) * 1000
    
    st.session_state.total_calls += 1
    if is_error:
        st.session_state.error_count += 1
    
    error_rate = 0
    if st.session_state.total_calls > 0:
        error_rate = (st.session_state.error_count / st.session_state.total_calls) * 100

    return cpu, memory, latency, error_rate

# ===============================
# 7. MAIN LOOP
# ===============================
if start_btn:
    while True:
        # A. Fetch Data
        cpu, mem, lat, err = fetch_real_data()
        
        # B. Update History
        new_row = pd.DataFrame({
            "Timestamp": [pd.Timestamp.now()],
            "CPU": [cpu],
            "Memory": [mem],
            "Latency": [lat],
            "Error_Rate": [err]
        })
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
        if len(st.session_state.history) > window_size:
            st.session_state.history = st.session_state.history.tail(window_size)
        
        # C. UPDATE UI (Use placeholders to overwrite)
        # Instead of creating new metrics, we overwrite the existing slots
        cpu_placeholder.metric("Real CPU", f"{cpu:.1f}%")
        mem_placeholder.metric("Real RAM", f"{mem:.1f}%")
        lat_placeholder.metric("Real Latency", f"{int(lat)} ms")
        err_placeholder.metric("Error Rate", f"{err:.1f}%")
        
        # D. Update Graph
        fig = px.line(
            st.session_state.history, 
            x="Timestamp", 
            y=["CPU", "Memory"], 
            title=f"System Resources (Last {window_size} points)",
            markers=True
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # E. AI Prediction
        input_data = np.array([[cpu, mem, lat, err]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        
        if prediction[0] == -1:
            alert_placeholder.error(f"🚨 **AI ALERT:** Silent Failure Detected! {input_data}")
        else:
            alert_placeholder.success("✅ **SYSTEM HEALTHY:** AI Model predicts normal operations.")
            
        time.sleep(1)