
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import yaml
import time
import os
from collections import deque
import psutil
import smtplib
from email.mime.text import MIMEText

# Custom CSS for styling
st.markdown("""
<style>
    /* Main dashboard title and headers - light orange */
    h1, h2, h3, h4, h5, h6 {
        color: #FF9F66 !important;
    }
    
    /* Streamlit metric labels - light orange */
    [data-testid="stMetricLabel"] {
        color: #FF9F66 !important;
    }
    
    /* Streamlit metric values - black for data */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    /* Background - white */
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #F5F5F5 !important;
    }
    
    /* Alert boxes - critical (red) */
    .alert-box {
        padding: 12px;
        margin: 8px 0;
        border-radius: 5px;
        font-weight: 500;
    }
    
    .alert-critical {
        background-color: #FFE5E5;
        border-left: 5px solid #FF0000;
        color: #CC0000;
    }
    
    /* Alert boxes - warning (orange-red) */
    .alert-warning {
        background-color: #FFF3E5;
        border-left: 5px solid #FF6B6B;
        color: #D04545;
    }
    
    /* Buttons - light orange */
    .stButton button {
        background-color: #FF9F66 !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton button:hover {
        background-color: #FF8844 !important;
    }
    
    /* Data tables - black text */
    .dataframe {
        color: #000000 !important;
    }
    
    /* Chart titles - light orange */
    .plot-container .gtitle {
        fill: #FF9F66 !important;
    }
    
    /* Input fields border - light orange on focus */
    input:focus {
        border-color: #FF9F66 !important;
    }
    
    /* Expander headers - light orange */
    [data-testid="stExpander"] summary {
        color: #FF9F66 !important;
    }
</style>
""", unsafe_allow_html=True)
import smtplib
from email.mime.text import MIMEText

# --- 1. CONFIGURATION LOADING ---
def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    return {
        "model_path": "nasa_rul_failure_prediction_model.pkl",
        "monitoring": {
            "latency_threshold_ms": 100.0,
            "drift_sigma_threshold": 3.0,
            "health_warning_threshold": 70,
            "health_warning_threshold": 70, "health_critical_threshold": 40
        },
        "email": {
            "enabled": True,
            "host": "sandbox.smtp.mailtrap.io",
            "port": 2525,
            "username": "8c1255d22724ad",
            "password": "1347fbc2c88a1b",
            "sender": "AI Monitor <alerts@demo.ai>"
        },

    }

CONFIG = load_config()

# --- 2. THE MONITOR (CORE LOGIC) ---
class ModelMonitor:
    def __init__(self, model_path, warmup_target=100):
        self.model = self._load_model_robust(model_path)
        # Per-metric baseline statistics will be filled after warmup
        self.baseline_stats = {}
        # Rolling history for UI charts
        self.history = {
            "predictions": deque(maxlen=200),
            "latency": deque(maxlen=200),
            "cpu_delta": deque(maxlen=200),
            "mem_delta": deque(maxlen=200),
            "health": deque(maxlen=200)
        }
        self.alerts = []
        # Baseline warmup/learning
        self.warmup_target = warmup_target
        self.warmup_data = []
        self.baseline_learned = False
        # metadata from model package (optional)
        self.feature_names = None
        self._model_package = None
        
    def _load_model_robust(self, path):
        if not os.path.exists(path):
            st.warning(f"Model file '{path}' not found. Using Mock Model for Demo.")
            return None
        try:
            obj = joblib.load(path)
            # If a package dict was saved, keep it for metadata
            if isinstance(obj, dict):
                self._model_package = obj
                # Extract feature names if provided
                self.feature_names = obj.get('feature_names', None)
                for key in ['model', 'estimator', 'pipeline']:
                    if key in obj:
                        return obj[key]
                # Fallback to 'model' key if present
                if 'model' in obj:
                    return obj['model']
                # Otherwise return the dict itself
                return obj
            # If model object has feature names (scikit-learn), extract them
            if hasattr(obj, 'feature_names_in_'):
                self.feature_names = list(obj.feature_names_in_)
            return obj
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None

    def predict(self, input_data):
        """Returns a metrics dict measured around the model inference.
        Measures CPU and memory deltas specifically during prediction, and converts
        numpy input into a pandas DataFrame with feature names when available."""
        # Prepare input as a pandas DataFrame with feature names when possible
        try:
            arr = np.array(input_data)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if self.feature_names is not None and len(self.feature_names) == arr.shape[1]:
                df = pd.DataFrame(arr, columns=self.feature_names)
            elif hasattr(self.model, 'feature_names_in_'):
                df = pd.DataFrame(arr, columns=list(self.model.feature_names_in_))
            else:
                df = pd.DataFrame(arr, columns=[f'f{i}' for i in range(arr.shape[1])])
        except Exception:
            # Fallback to DataFrame conversion
            df = pd.DataFrame(input_data)

        # Measure resource usage around the model call
        if self.model:
            try:
                start_cpu = psutil.cpu_percent(interval=0.1)
                start_mem = psutil.Process().memory_info().rss
                start = time.time()
                pred = self.model.predict(df)
                latency = (time.time() - start) * 1000
                end_cpu = psutil.cpu_percent(interval=None)
                end_mem = psutil.Process().memory_info().rss

                cpu_delta = end_cpu - start_cpu
                mem_delta = (end_mem - start_mem) / (1024.0 * 1024.0)  # bytes -> MB

                val = float(pred[0]) if hasattr(pred, "__getitem__") else float(pred)
            except Exception as e:
                # Fallback synthetic metrics when prediction fails
                val = 0.0
                latency = (time.time() - start) * 1000 if 'start' in locals() else 0.0
                cpu_delta = 0.0
                mem_delta = 0.0
        else:
            # Mock prediction for demo if no model loaded
            val = float(np.random.normal(100, 10))
            latency = float(np.random.uniform(20, 200))
            cpu_delta = float(np.random.uniform(0, 5))
            mem_delta = float(np.random.uniform(0, 5))

        metrics = {
            'value': float(val),
            'latency': float(latency),
            'cpu_delta': float(cpu_delta),
            'mem_delta': float(mem_delta)
        }
        return metrics

    def append_history(self, metrics):
        """Append latest metrics into rolling history used for charts."""
        self.history['predictions'].append(metrics['value'])
        self.history['latency'].append(metrics['latency'])
        self.history['cpu_delta'].append(metrics.get('cpu_delta', 0.0))
        self.history['mem_delta'].append(metrics.get('mem_delta', 0.0))

    def collect_baseline(self, metrics):
        """Collect initial metrics to learn a baseline (mean/std) for z-score alerts."""
        if self.baseline_learned:
            return
        self.warmup_data.append(metrics)
        if len(self.warmup_data) >= self.warmup_target:
            df = pd.DataFrame(self.warmup_data)
            for col in df.columns:
                mu = float(df[col].mean())
                sigma = float(df[col].std()) if df[col].std() > 0 else 1e-6
                self.baseline_stats[col] = {'mean': mu, 'std': sigma}
            self.baseline_learned = True

    def check_health(self, metrics):
        alerts = []
        health_score = 100

        # 1. Latency hard-threshold check
        if metrics['latency'] > CONFIG['monitoring']['latency_threshold_ms']:
            health_score -= 20
            alerts.append(f"High Latency: {metrics['latency']:.1f}ms")

        # 2. Use z-scores against baseline once learned
        if self.baseline_learned:
            warning = False
            critical = False
            for key in ['value', 'latency', 'cpu_delta', 'mem_delta']:
                if key not in self.baseline_stats:
                    continue
                mean = self.baseline_stats[key]['mean']
                std = self.baseline_stats[key]['std'] if self.baseline_stats[key]['std'] > 0 else 1e-6
                z = abs(metrics.get(key, 0.0) - mean) / std
                if z > 3:
                    alerts.append(f"CRITICAL: {key} {z:.1f}σ from baseline")
                    critical = True
                elif z > 2:
                    alerts.append(f"WARNING: {key} {z:.1f}σ from baseline")
                    warning = True
            if critical:
                health_score -= 50
            elif warning:
                health_score -= 25

        # Clamp score
        health_score = max(0, min(100, health_score))
        return health_score, alerts

# --- EMAIL FUNCTION ---
def send_alert_email(to_email, subject, body, config):
    email_config = config.get('email', {})
    if not email_config.get('enabled', True):  # Default to enabled if not set
        return False

    try:
        port = email_config.get('port', 2525)
        print(f"Connecting to {email_config['host']}:{port}")
        if port == 465:
            server = smtplib.SMTP_SSL(email_config['host'], port)
        else:
            server = smtplib.SMTP(email_config['host'], port)
            if port == 587:
                server.starttls()
        print("Logging in...")
        server.login(email_config['username'], email_config['password'])
        print("Logged in, sending email...")
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = email_config['sender']
        msg['To'] = to_email
        server.sendmail(email_config['sender'], to_email, msg.as_string())
        print("Email sent successfully")
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        st.error(f"Failed to send email: {e}")
        return False


# --- 3. DASHBOARD UI ---
st.set_page_config(page_title="AI Model Monitor", layout="wide")

# Load local background image (if present) and use as data URI; fallback to path otherwise
bg_path = "background.png"
bg_url = "app/static/background.png"
if os.path.exists(bg_path):
    try:
        import base64
        with open(bg_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        bg_url = f"data:image/png;base64,{b64}"
    except Exception:
        # keep fallback
        bg_url = "app/static/background.png"

st.markdown(f"""
st.set_page_config(page_title="SilentGuard", layout="wide", page_icon="🤖")

st.markdown("""
<style>
    /* Set the background image */
    .stApp {{
        background-image: url("{bg_url}");
        background-attachment: fixed;
        background-size: cover;
    }}

    /* Make the main content containers transparent */
    [data-testid="stHeader"], [data-testid="stAppViewContainer"] {{
        background-color: rgba(0,0,0,0) !important;
    }}

    /* Sidebar transparency */
    [data-testid="stSidebar"] {{
        background-color: rgba(245, 245, 245, 0.8) !important; /* Semi-transparent gray */
        backdrop-filter: blur(10px);
    }}
    
    /* Headers and Labels - keeping your light orange theme */
    h1, h2, h3, h4, h5, h6, [data-testid="stMetricLabel"] {{
        color: #FF9F66 !important;
    }}
    
    /* Improve readability of metric values over the image */
    [data-testid="stMetricValue"] {{
        color: #000000 !important;
        background-color: rgba(255, 255, 255, 0.4);
        padding: 5px;
        border-radius: 5px;
    }}

    /* Alert boxes - keeping functional colors */
    .alert-box {{
        padding: 12px;
        margin: 8px 0;
        border-radius: 5px;
        font-weight: 500;
        backdrop-filter: blur(5px);
    }}
    
    .alert-critical {{
        background-color: rgba(255, 229, 229, 0.9);
        border-left: 5px solid #FF0000;
        color: #CC0000;
    }}
    
    .alert-warning {{
        background-color: rgba(255, 243, 229, 0.9);
        border-left: 5px solid #FF6B6B;
        color: #D04545;
    }}
    
    /* Buttons */
    .stButton button {{
        background-color: #FF9F66 !important;
        color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ SilentGuard")

# Sidebar Controls
st.sidebar.header("⚙️ Controls")

if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False

def start_monitoring():
    st.session_state.monitoring_active = True

def stop_monitoring():
    st.session_state.monitoring_active = False

st.sidebar.button("▶ Start Monitoring", type="primary", on_click=start_monitoring)
st.sidebar.button("⏹ Stop Monitoring", on_click=stop_monitoring)

user_email = st.sidebar.text_input("Enter your email for alerts", placeholder="user@example.com")

if st.sidebar.button("📧 Send Report"):
    if user_email:
        # Generate report
        total_steps = 0
        avg_health = 0
        total_alerts = 0
        if 'monitor' in locals() and monitor is not None:
            total_steps = len(monitor.history['predictions'])
            avg_health = np.mean(list(monitor.history['health'])) if monitor.history['health'] else 0
        if 'alerts_history' in locals():
            total_alerts = len(alerts_history)
        report_body = f"""
AI Model Monitoring Report

Total Monitoring Steps: {total_steps}
Average Health Score: {avg_health:.1f}/100
Total Alerts: {total_alerts}

Recent Alerts:
""" + "\n".join(list(alerts_history)[:10] if 'alerts_history' in locals() else []) + """

Thank you for using the AI Monitor.
"""
        send_alert_email(user_email, "AI Monitor Report", report_body, CONFIG)
        st.sidebar.success("Report sent!")
    else:
        st.sidebar.error("Please enter your email first.")

st.sidebar.markdown("---")
config_view = st.sidebar.expander("View Config")
config_view.json(CONFIG)

# Layout: Top Banner (Health), Middle (Charts), Bottom (Alerts)
col_health, col_ttf, col_status = st.columns(3)
health_ph = col_health.empty()
ttf_ph = col_ttf.empty()
status_ph = col_status.empty()

chart_col1, chart_col2 = st.columns(2)
chart_pred_ph = chart_col1.empty()
chart_lat_ph = chart_col2.empty()

st.markdown("###  Active Alerts")
alert_log_ph = st.empty()

# --- 4. DATA SIMULATION (DEMO) ---
def get_synthetic_input(step):
    # Matches NASA CMAPSS feature count (24) roughly
    # Normal behavior for first 50 steps
    # Then Drift + Latency spikes
    
    base_feats = np.random.normal(0, 1, 24)
    
    if step > 50:
        # Inject Drift
        base_feats[0] += (step - 50) * 0.1 
        
    return base_feats.reshape(1, -1)

# --- 5. MAIN LOOP ---
if st.session_state.monitoring_active:
    monitor = ModelMonitor(CONFIG['model_path'], warmup_target=100)

    # State
    alerts_history = deque(maxlen=200)
    step = 0
    last_status = "NORMAL"
    max_steps = 500  # Limit to prevent infinite loop blocking UI

    # Additional placeholders for CPU and Memory charts (clean UI: one chart per metric)
    chart_col3, chart_col4 = st.columns(2)
    chart_cpu_ph = chart_col3.empty()
    chart_mem_ph = chart_col4.empty()

    while True:
        # 1. Get Data
        input_data = get_synthetic_input(step)

        # 2. Predict & Monitor (CPU/memory deltas measured inside predict())
        metrics = monitor.predict(input_data)

        # Artificial Latency Injection for Demo (kept for demo spikes)
        if step > 80 and step % 5 == 0:
            metrics['latency'] += 150  # Spike > 100ms threshold
            time.sleep(0.15)

        # 3. Baseline warmup
        if not monitor.baseline_learned:
            monitor.collect_baseline(metrics)
        else:
            pass

        # 4. Append to history used for charts
        monitor.append_history(metrics)

        # 5. Health checks (only meaningful after baseline learned)
        if monitor.baseline_learned:
            score, current_alerts = monitor.check_health(metrics)
        else:
            score, current_alerts = 100, []

        # Determine current status
        current_status = "CRITICAL" if score < 40 else "WARNING" if score < 70 else "NORMAL"

        # Send email on status change to WARNING or CRITICAL
        if user_email and current_status != last_status and current_status in ["WARNING", "CRITICAL"]:
            subject = f"AI Monitor Alert: {current_status}"
            body = f"System status changed to {current_status}.\n\nAlerts:\n" + "\n".join(current_alerts) + f"\n\nHealth Score: {score}/100\nStep: {step}"
            send_alert_email(user_email, subject, body, CONFIG)
            last_status = current_status

        # Add alerts to log
        if current_alerts:
            for a in current_alerts:
                alerts_history.appendleft(f"[Step {step}] {a}")

        # 6. Render UI
        # Health Gauge with custom light orange color
        health_color = "#28A745" if score > 70 else "#FF9F66" if score > 40 else "#FF0000"
        health_ph.markdown(f"""
            <div style="text-align: center;">
                <h3 style="margin:0; color:#FF9F66;">System Health</h3>
                <h1 style="color: {health_color}; font-size: 3em; margin:0;">{score}/100</h1>
            </div>
        """, unsafe_allow_html=True)

        # Learning status or Current Status
        if not monitor.baseline_learned:
            status_ph.metric("Status", f"Learning Baseline: {len(monitor.warmup_data)}/{monitor.warmup_target}")
            ttf_ph.metric("Est. Time to Failure", "Learning...")
        else:
            status_text = "CRITICAL" if score < 40 else "WARNING" if score < 70 else "NORMAL"
            status_ph.metric("Current Status", status_text)

            # Simple TTF logic using health trend
            if len(monitor.history['health']) > 10:
                h_recent = list(monitor.history['health'])[-10:]
                diff = h_recent[0] - h_recent[-1]
                if diff > 0:
                    steps_left = int(score / (diff / 10)) if diff != 0 else 0
                    ttf_text = f"{steps_left} cycles"
                else:
                    ttf_text = "Stable"
            else:
                ttf_text = "Calibrating..."
            ttf_ph.metric("Est. Time to Failure", ttf_text)

        monitor.history['health'].append(score)

        # Charts: Prediction, Latency, CPU Delta, Memory Delta with custom colors
        fig_pred = px.line(y=list(monitor.history['predictions']), title="Prediction Stream (Drift Detection)")
        fig_pred.update_traces(line_color='#FF9F66')
        if monitor.baseline_learned and 'value' in monitor.baseline_stats:
            mean = monitor.baseline_stats['value']['mean']
            std = monitor.baseline_stats['value']['std']
            fig_pred.add_hline(y=mean, line_dash="dash", line_color="green")
            fig_pred.add_hline(y=mean + 3*std, line_dash="dot", line_color="#FF0000")
            fig_pred.add_hline(y=mean - 3*std, line_dash="dot", line_color="#FF0000")
        fig_pred.update_layout(
            height=300, 
            margin=dict(l=0, r=0, t=30, b=0),
            title_font=dict(color='#FF9F66'),
            font=dict(color='#000000'),
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF'
        )
        try:
            chart_pred_ph.plotly_chart(fig_pred, width="stretch")
        except:
            chart_pred_ph.plotly_chart(fig_pred, use_container_width=True)

        fig_lat = px.line(y=list(monitor.history['latency']), title="Inference Latency (ms)")
        fig_lat.update_traces(line_color='#FF9F66')
        fig_lat.add_hline(y=CONFIG['monitoring']['latency_threshold_ms'], line_color="#FF0000")
        fig_lat.update_layout(
            height=250, 
            margin=dict(l=0, r=0, t=30, b=0),
            title_font=dict(color='#FF9F66'),
            font=dict(color='#000000'),
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF'
        )
        try:
            chart_lat_ph.plotly_chart(fig_lat, width="stretch")
        except:
            chart_lat_ph.plotly_chart(fig_lat, use_container_width=True)

        fig_cpu = px.line(y=list(monitor.history['cpu_delta']), title="CPU Delta (%)")
        fig_cpu.update_traces(line_color='#FF9F66')
        fig_cpu.update_layout(
            height=250, 
            margin=dict(l=0, r=0, t=30, b=0),
            title_font=dict(color='#FF9F66'),
            font=dict(color='#000000'),
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF'
        )
        try:
            chart_cpu_ph.plotly_chart(fig_cpu, width="stretch")
        except:
            chart_cpu_ph.plotly_chart(fig_cpu, use_container_width=True)

        fig_mem = px.line(y=list(monitor.history['mem_delta']), title="Memory Delta (MB)")
        fig_mem.update_traces(line_color='#FF9F66')
        fig_mem.update_layout(
            height=250, 
            margin=dict(l=0, r=0, t=30, b=0),
            title_font=dict(color='#FF9F66'),
            font=dict(color='#000000'),
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF'
        )
        try:
            chart_mem_ph.plotly_chart(fig_mem, width="stretch")
        except:
            chart_mem_ph.plotly_chart(fig_mem, use_container_width=True)

        # Alert Log (show last 10) with red styling for alerts
        alert_html = ""
        for msg in list(alerts_history)[:10]:
            color = "alert-critical" if "CRITICAL" in msg.upper() else ("alert-warning" if "WARNING" in msg.upper() else "alert-warning")
            alert_html += f"<div class='alert-box {color}'>{msg}</div>"
        alert_log_ph.markdown(alert_html, unsafe_allow_html=True)

        # Pause between iterations to make it truly real-time (2.5s)
        time.sleep(2.5)
        step += 1
        if step >= max_steps:
            break
