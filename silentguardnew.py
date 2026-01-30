import streamlit as st
import numpy as np
import pandas as pd
import time
import pickle  # <--- NEW: Required to load your .pkl files

# ===============================
# 1. LOAD YOUR AI MODELS
# ===============================
# We try to load the files. If they aren't found, we handle the error gracefully.
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.warning("⚠️ 'model.pkl' or 'scaler.pkl' not found. Using simple threshold logic instead.")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI-Powered Monitor", layout="wide")
st.title("Silent Model Failure Detection System")
st.caption("Monitoring via AI Model (model.pkl)")

# ===============================
# 2. SIMULATE LIVE DATA
# ===============================
# We generate data points that your model will analyze
# NOTE: These inputs MUST match the order you used in 'train_model.py'
# Assumption: You trained on [CPU, Memory, Latency, Error_Rate]
current_cpu = np.random.uniform(20, 90)       # Simulated CPU %
current_memory = np.random.uniform(30, 80)    # Simulated RAM %
current_latency = np.random.normal(120, 30)   # Simulated Latency ms
current_error_rate = np.random.uniform(0, 5)  # Simulated Error %

# Create a DataFrame for the metrics (Visuals)
st.metric("CPU Usage", f"{current_cpu:.1f}%")
st.metric("Memory Usage", f"{current_memory:.1f}%")

col1, col2 = st.columns(2)
col1.metric("Latency", f"{int(current_latency)} ms")
col2.metric("Error Rate", f"{current_error_rate:.1f}%")

st.divider()

# ===============================
# 3. AI PREDICTION LOGIC
# ===============================
st.subheader("🤖 AI Model Diagnosis")

if model_loaded:
    # A. Prepare Input: Organize data into the exact shape the model expects
    # IMPORTANT: Check 'train_model.py' to see the order of features!
    # Here I assume: [CPU, Memory, Latency, Error Rate]
    input_features = np.array([[current_cpu, current_memory, current_latency, current_error_rate]])
    
    # B. Scale Input: Use the loaded scaler to normalize the data
    scaled_features = scaler.transform(input_features)
    
    # C. Predict: Ask the model "Is this a failure?" (1 = Yes, 0 = No)
    prediction = model.predict(scaled_features)
    
    # D. Display Result
    if prediction[0] == 1:
        st.error("🚨 CRITICAL ALERT: The AI Model detected a system failure!")
        st.write(f"The model analyzed the input: {input_features}")
    else:
        st.success("✅ SYSTEM NORMAL: AI Model predicts safe operations.")

else:
    # Fallback if model.pkl is missing (Simple math check)
    if current_latency > 200:
        st.error("⚠️ High Latency Detected (Threshold Rule)")
    else:
        st.success("✅ System Normal (Threshold Rule)")