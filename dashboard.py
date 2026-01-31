
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import yaml
import time
import os
import datetime
from collections import deque
import psutil
import psutil
import mailtrap as mt
import base64
import io
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION LOADING ---
def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    return {
        "model_path": "nasa_rul_failure_prediction_model.pkl",
        "monitoring": {
            "baseline_window_size": 50,
            "latency_threshold_ms": 100.0,
            "sensitivity_threshold": 0.1,
            "variance_threshold": 10.0,
            "uncertainty_threshold": 5.0,
            "min_safety_margin": 10.0,
            "health_warning_threshold": 70,
            "health_critical_threshold": 40
        },
        "email": {"enabled": False}
    }

CONFIG = load_config()

# Helper to load optional background image and apply a lighter overlay + UI accent colors

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Get background image
bg_image = get_base64_image("background.png")

# Custom CSS for styling with background image
bg_style = f"""
    background-image: url("data:image/png;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
""" if bg_image else "background-color: transparent;"

st.markdown(f"""
<style>
    
    /* Main app background with image */
    .stApp {{
        {bg_style}
    }}
    
    /* Overlay kept transparent so background is visible */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.0);
        z-index: 0;
        pointer-events: none;
    }}
    
    
    /* Ensure content is above overlay */
    .main .block-container {{
        position: relative;
        z-index: 1;
    }}
    
    /* Dashboard headers - light orange */
    h1, h2, h3, h4, h5, h6 {{
        color: #FF9F66 !important;
    }}
    
    /* Metric labels - light orange */
    [data-testid="stMetricLabel"] {{
        color: #FF9F66 !important;
        font-weight: 600 !important;
    }}
    
    /* Metric values and all numericals - black */
    [data-testid="stMetricValue"] {{
        color: #000000 !important;
        font-weight: 700 !important;
    }}
    
    /* All text - black */
    .stMarkdown, p, span, div {{
        color: #000000 !important;
    }}
    
    /* Sidebar background - light with transparency */
    [data-testid="stSidebar"] {{
        background-color: rgba(245, 245, 245, 0.95) !important;
    }}
    
    [data-testid="stSidebar"] * {{
        color: #000000 !important;
    }}
    
    /* Alert boxes - RED for all alerts */
    .alert-box {{
        padding: 12px;
        margin: 8px 0;
        border-radius: 5px;
        font-weight: 600;
        background-color: #FFE5E5;
        border-left: 5px solid #FF0000;
        color: #FF0000 !important;
    }}
    
    .alert-critical {{
        background-color: #FFE5E5;
        border-left: 5px solid #FF0000;
        color: #FF0000 !important;
    }}
    
    .alert-warning {{
        background-color: #FFE5E5;
        border-left: 5px solid #FF0000;
        color: #FF0000 !important;
    }}
    
    /* Buttons - light orange */
    .stButton button {{
        background-color: #FF9F66 !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }}
    
    .stButton button:hover {{
        background-color: #FF8844 !important;
    }}
    
    /* Data tables - black text */
    .dataframe {{
        color: #000000 !important;
    }}
    
    /* Input fields */
    input {{
        color: #000000 !important;
    }}
    
    input:focus {{
        border-color: #FF9F66 !important;
    }}
    
    /* Expander headers - light orange */
    [data-testid="stExpander"] summary {{
        color: #FF9F66 !important;
        font-weight: 600 !important;
    }}
    
    /* Select boxes and other inputs */
    [data-baseweb="select"] {{
        color: #000000 !important;
    }}
    
    /* Text input */
    [data-testid="stTextInput"] input {{
        color: #000000 !important;
    }}
    
    /* Ensure chart backgrounds are transparent */
    .js-plotly-plot .plotly {{
        background: rgba(255, 255, 255, 0.0) !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- 3. THE MONITOR (CORE LOGIC) ---
class ModelMonitor:
    def __init__(self, model_path):
        self.model = self._load_model_robust(model_path)
        self.warmup_target = CONFIG['monitoring'].get('baseline_window_size', 50)
        
        # Per-metric baseline statistics
        self.baseline_stats = {}
        
        # Rolling history for UI charts
        maxlen = 200
        self.history = {
            "predictions": deque(maxlen=maxlen),
            "latency": deque(maxlen=maxlen),
            "cpu_delta": deque(maxlen=maxlen),
            "mem_delta": deque(maxlen=maxlen),
            "health": deque(maxlen=maxlen),
            "variance": deque(maxlen=maxlen),
            "sensitivity": deque(maxlen=maxlen),
            "uncertainty": deque(maxlen=maxlen),
            "margin": deque(maxlen=maxlen),
            "steps": deque(maxlen=maxlen) # Track X-axis (steps) explicitly
        }
        
        self.recent_predictions = deque(maxlen=20) 
        self.alerts = []
        self.warmup_data = [] 
        self.baseline_learned = False
        
        self.model_info = {
            "type": str(type(self.model).__name__) if self.model else "Mock",
            "features": "Auto-detected"
        }
        if hasattr(self.model, 'n_features_in_'):
            self.model_info["features"] = str(self.model.n_features_in_)

        # Persistence
        self.model_path = model_path
        self.baseline_source = "Not learned"
        self._load_baseline()

    def _get_baseline_path(self):
        # Ensure baselines directory exists
        base_dir = os.path.join(os.path.dirname(self.model_path) if os.path.isabs(self.model_path) else ".", "baselines")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        # Generate filename based on model path to avoid conflicts
        base_name = os.path.splitext(os.path.basename(self.model_path))[0]
        return os.path.join(base_dir, f"{base_name}_baseline.json")

    def _save_baseline(self):
        import json
        try:
            data = {
                "metadata": {
                    "model_path": self.model_path,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "window_size": self.warmup_target
                },
                "stats": self.baseline_stats,
                "raw_data": list(self.warmup_data) # Save all raw steps
            }
            with open(self._get_baseline_path(), "w") as f:
                json.dump(data, f, indent=4)
            print(f"✅ Baseline data saved to {self._get_baseline_path()}")
            self.baseline_source = f"Saved {data['metadata']['timestamp']}"
        except Exception as e:
            print(f"Failed to save baseline: {e}")

    def _load_baseline(self):
        import json
        path = self._get_baseline_path()
        if not os.path.exists(path):
            return
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            # Validation
            saved_model_path = data.get("metadata", {}).get("model_path", "")
            if saved_model_path != self.model_path:
                print(f"⚠️ Baseline mismatch: Saved for {saved_model_path}, current is {self.model_path}")
                return

            self.baseline_stats = data["stats"]
            self.warmup_data = data.get("raw_data", []) # Restore raw data
            self.baseline_learned = True
            self.baseline_source = f"Loaded from file ({data['metadata']['timestamp']})"
            print(f"✅ Baseline loaded from {path}")
        except Exception as e:
            print(f"Failed to load baseline: {e}")

    def _load_model_robust(self, path):
        if not os.path.exists(path):
            return None
        try:
            obj = joblib.load(path)
            if isinstance(obj, dict):
                return obj.get('model', obj.get('estimator', obj))
            return obj
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict(self, input_data):
        try:
            arr = np.array(input_data)
            if arr.ndim == 1: arr = arr.reshape(1, -1)
            df = arr 
        except:
            df = input_data

        if not self.model:
            return self._mock_metrics()

        try:
            # Resource Measurement
            process = psutil.Process()
            start_cpu = process.cpu_percent(interval=None) 
            start_mem = process.memory_info().rss
            start_time = time.time()
            
            # Inference
            pred_raw = self.model.predict(df)
            
            end_time = time.time()
            end_mem = process.memory_info().rss
            end_cpu = process.cpu_percent(interval=None) # Interval=None gets usage since last call
            
            latency_ms = (end_time - start_time) * 1000
            cpu_delta = max(0.0, end_cpu) 
            mem_delta = (end_mem - start_mem) / (1024 * 1024) 
            
            val = float(pred_raw[0]) if hasattr(pred_raw, "__getitem__") else float(pred_raw)

            # Advanced Metrics
            self.recent_predictions.append(val)
            pred_variance =  np.var(self.recent_predictions) if len(self.recent_predictions) > 2 else 0.0
            
            try:
                # Sensitivity: 1% Noise
                noise = np.random.normal(0, 0.01, arr.shape) * arr
                perturbed_input = arr + noise
                pred_perturbed = float(self.model.predict(perturbed_input)[0])
                sensitivity = abs((pred_perturbed - val) / (val + 1e-9))
            except:
                sensitivity = 0.0

            uncertainty = 0.0
            if hasattr(self.model, 'estimators_'):
                try:
                    tree_preds = [tree.predict(df)[0] for tree in self.model.estimators_]
                    uncertainty = np.std(tree_preds)
                except:
                    uncertainty = 0.0
            else:
                if len(self.recent_predictions) > 2:
                    uncertainty = abs(val - np.mean(self.recent_predictions))
            
            margin = max(0.0, val) 

            return {
                'value': val,
                'latency': latency_ms,
                'cpu_delta': cpu_delta,
                'mem_delta': mem_delta,
                'variance': float(pred_variance),
                'sensitivity': float(sensitivity),
                'uncertainty': float(uncertainty),
                'margin': float(margin)
            }

        except Exception as e:
            print(f"Inference Error: {e}")
            return self._mock_metrics()

    def _mock_metrics(self):
        val = np.random.normal(100, 10)
        return {
            'value': val,
            'latency': np.random.uniform(50, 150),
            'cpu_delta': np.random.uniform(0, 5),
            'mem_delta': np.random.uniform(0, 2),
            'variance': np.random.uniform(0, 5),
            'sensitivity': np.random.uniform(0, 0.05),
            'uncertainty': np.random.uniform(0, 2),
            'margin': max(0, val)
        }

    def update(self, input_data, step_num):
        metrics = self.predict(input_data)
        
        self.history['steps'].append(step_num)
        self.history['predictions'].append(metrics['value'])
        self.history['latency'].append(metrics['latency'])
        self.history['cpu_delta'].append(metrics['cpu_delta'])
        self.history['mem_delta'].append(metrics['mem_delta'])
        self.history['variance'].append(metrics['variance'])
        self.history['sensitivity'].append(metrics['sensitivity'])
        self.history['uncertainty'].append(metrics['uncertainty'])
        self.history['margin'].append(metrics['margin'])
        
        if not self.baseline_learned:
            self.warmup_data.append(metrics)
            if len(self.warmup_data) >= self.warmup_target:
                self._finalize_baseline()
        
        score, alerts = self._check_health(metrics)
        self.history['health'].append(score)
        
        return metrics, score, alerts
        
    def _finalize_baseline(self):
        df = pd.DataFrame(self.warmup_data)
        for col in df.columns:
            mu = df[col].mean()
            sigma = df[col].std() if df[col].std() > 0 else 1e-6
            self.baseline_stats[col] = {'mean': mu, 'std': sigma}
        self.baseline_learned = True
        self._save_baseline()

    def _check_health(self, metrics):
        if not self.baseline_learned:
            return 100, []

        alerts = []
        demerits = 0
        
        # 1. Latency Hard Threshold
        if metrics['latency'] > CONFIG['monitoring']['latency_threshold_ms']:
            demerits += 20
            alerts.append(f"High Latency: {metrics['latency']:.0f}ms")
            
        # 2. Z-Score Checks 
        problematic_metrics = ['variance', 'sensitivity', 'uncertainty']
        for key in problematic_metrics:
            if key not in self.baseline_stats: continue
            stats = self.baseline_stats[key]
            z = (metrics[key] - stats['mean']) / stats['std']
            
            if z > 3:
                alerts.append(f"CRITICAL {key.upper()}: +{z:.1f}σ")
                demerits += 30
            elif z > 2:
                alerts.append(f"WARNING {key.upper()}: +{z:.1f}σ")
                demerits += 15
        
        if 'margin' in self.baseline_stats:
            if metrics['margin'] < CONFIG['monitoring']['min_safety_margin']:
                 alerts.append(f"LOW MARGIN: {metrics['margin']:.1f}")
                 demerits += 50
            
        score = max(0, 100 - demerits)
        return score, alerts

# --- 4. EMAILER ---
# --- 4. EMAILER ---
def generate_sparkline(data, title, color='red'):
    """Generate a small sparkline chart as base64 string"""
    if not data or len(data) < 2: return None
    
    plt.figure(figsize=(4, 1.5))
    plt.plot(list(data), color=color, linewidth=2)
    plt.title(title, fontsize=10, pad=3)
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def send_email_alert(alert_list, score, step, current_metrics, monitor, force=False):
    # Check Session State Toggle & Cooldown
    if not st.session_state.get('email_enabled', False) and not force: return
    
    # Simple Cooldown (e.g., 5 minutes to prevent spamming)
    last_sent = st.session_state.get('last_email_time', 0)
    if not force and time.time() - last_sent < 300: 
        return
    
    try:
        sender = CONFIG['email'].get('sender', 'alerts@silentguard.ai')
        recipients = st.session_state.get('email_recipients', [])
        if not recipients: return
        
        # --- PREPARE DATA ---
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        baseline_learned = monitor.baseline_learned
        baseline_stats = monitor.baseline_stats
        
        # 1. SEVERITY & COLOR
        severity = "CRITICAL" if score < CONFIG['monitoring']['health_critical_threshold'] else "WARNING"
        color = "#DC3545" if severity == "CRITICAL" else "#FFC107"
        
        # 2. ANALYSIS
        root_causes = []
        rec_actions = []
        
        # Analyze specific failures
        if any("Latency" in a for a in alert_list):
            root_causes.append("System overload or network congestion")
            rec_actions.append("Check load balancer and scaling scaling policies")
        if any("VARIANCE" in a for a in alert_list) or any("SENSITIVITY" in a for a in alert_list):
            root_causes.append("Data Drift / Out-of-Distribution Input detected")
            rec_actions.append("Verify input data quality and schema")
        if any("MARGIN" in a for a in alert_list):
            root_causes.append("Model confidence degrading due to shifting patterns")
            rec_actions.append("Schedule model retraining pipeline")
        if any("CPU" in a for a in alert_list):
            root_causes.append("Resource exhaustion")
            rec_actions.append("Check container memory limits")
            
        if not root_causes: root_causes.append("Composite degradation of multiple factors")
        if not rec_actions: rec_actions.append("Review full system logs")

        # Create Trigger HTML
        trigger_html = "<ul>" + "".join([f"<li>{a}</li>" for a in alert_list]) + "</ul>"

        # 3. METRICS TABLE HTML
        rows = ""
        metric_display_names = {
            'value': 'Prediction', 'latency': 'Latency (ms)', 'cpu_delta': 'CPU (%)', 
            'mem_delta': 'Memory (MB)', 'variance': 'Variance', 
            'sensitivity': 'Sensitivity', 'uncertainty': 'Uncertainty', 'margin': 'Safety Margin'
        }
        
        for key, val in current_metrics.items():
            name = metric_display_names.get(key, key)
            base_val = "N/A"
            dev_str = "-"
            status_icon = "🟢"
            row_style = ""
            
            if baseline_learned and key in baseline_stats:
                b_mean = baseline_stats[key]['mean']
                base_val = f"{b_mean:.2f}"
                
                if b_mean != 0:
                    deviation = ((val - b_mean) / b_mean) * 100
                    dev_str = f"{deviation:+.1f}%"
                
                # Highlight logic (simplified based on alerts)
                is_alerted = any(key.upper() in a.upper() for a in alert_list)
                if is_alerted:
                    status_icon = "🔴"
                    row_style = "background-color: #ffe6e6;"
                elif key == 'latency' and val > CONFIG['monitoring']['latency_threshold_ms']:
                    status_icon = "🔴"
                    row_style = "background-color: #ffe6e6;"
            
            rows += f"""
            <tr style="{row_style}">
                <td style="padding: 8px; border: 1px solid #ddd;">{name}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{val:.4f}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{base_val}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{dev_str}</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{status_icon}</td>
            </tr>
            """

        # 4. PREDICTION HISTORY
        last_preds = list(monitor.recent_predictions)[-8:]
        pred_str = ", ".join([f"{p:.2f}" for p in last_preds])

        # 5. CHARTS (Base64)
        chart_html = ""
        # Check what to plot based on alerts or default to Health + worst metric
        chart_imgs = []
        
        # Always plot Health
        if monitor.history['health']:
            img = generate_sparkline(monitor.history['health'], "System Health Score", color='green')
            if img: chart_imgs.append(f'<div style="flex:1; min-width:200px;"><img src="data:image/png;base64,{img}" style="width:100%;"/><p style="text-align:center; font-size:12px;">Health Trend</p></div>')

        # Plot the first alerting metric found
        for key in ['latency', 'variance', 'sensitivity', 'margin']:
            if any(key.upper() in a.upper() for a in alert_list):
                 if monitor.history[key]:
                    img = generate_sparkline(monitor.history[key], f"{key.title()} Trend", color='red')
                    if img: chart_imgs.append(f'<div style="flex:1; min-width:200px;"><img src="data:image/png;base64,{img}" style="width:100%;"/><p style="text-align:center; font-size:12px;">{key.title()} Spike</p></div>')
                    break # Just one metric chart to keep it clean

        chart_html = '<div style="display:flex; gap:10px; flex-wrap:wrap;">' + "".join(chart_imgs) + '</div>'

        # --- HTML BODY ---
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: auto; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;">
                <!-- HEADER -->
                <div style="background-color: {color}; color: #000; padding: 20px; text-align: center;">
                    <h2 style="margin:0;">⚠️ {severity} ALERT</h2>
                    <p style="margin:5px 0 0;">Score: <strong>{score}%</strong> | Step: {step}</p>
                </div>
                
                <div style="padding: 20px;">
                    <!-- 1. REASON -->
                    <div style="margin-bottom: 20px; background: #fff3cd; padding: 15px; border-left: 5px solid #ffc107; border-radius: 4px;">
                        <strong style="color: #856404;">📝 Alert Triggers:</strong>
                        <div style="margin-top: 5px; color: #856404;">{trigger_html}</div>
                    </div>

                    <!-- 2. ANALYSIS -->
                    <h3 style="border-bottom: 2px solid #eee; padding-bottom: 5px;">🔍 Root Cause Analysis</h3>
                    <p><strong>Possible Cause:</strong> {", ".join(root_causes)}</p>
                    <p><strong>Severity Explanation:</strong> {severity} status indicates parameters have crossed safety thresholds by >3σ or critical limits.</p>

                    <!-- 3. TIMELINE -->
                    <h3 style="border-bottom: 2px solid #eee; padding-bottom: 5px;">⏱️ Incident Timeline</h3>
                    <ul style="list-style: none; padding: 0;">
                        <li><strong>Detected:</strong> {timestamp}</li>
                        <li><strong>Baseline Age:</strong> {monitor.warmup_target} steps (Established)</li>
                        <li><strong>Current Session:</strong> Step {step}</li>
                    </ul>

                    <!-- 4. METRICS -->
                    <h3 style="border-bottom: 2px solid #eee; padding-bottom: 5px;">📊 Affected Metrics</h3>
                    <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                        <tr style="background-color: #f8f9fa;">
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Metric</th>
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Current</th>
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Baseline</th>
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Dev %</th>
                            <th style="padding: 8px; border: 1px solid #ddd; text-align: center;">Status</th>
                        </tr>
                        {rows}
                    </table>

                    <!-- 5. CHARTS -->
                    <h3 style="border-bottom: 2px solid #eee; padding-bottom: 5px; margin-top: 20px;">📈 Trend Analysis</h3>
                    {chart_html}
                    
                    <p style="font-size: 12px; color: #666; margin-top: 10px;">
                        <strong>Recent Predictions:</strong> [{pred_str}]
                    </p>

                    <!-- 6. ACTIONS -->
                    <div style="background-color: #e2e3e5; padding: 15px; border-radius: 4px; margin-top: 20px;">
                        <h4 style="margin-top: 0;">🛡️ Recommended Actions</h4>
                        <ul style="margin-bottom: 0; padding-left: 20px;">
                            {''.join([f'<li>{a}</li>' for a in rec_actions])}
                        </ul>
                    </div>
                </div>
                
                <div style="background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #888;">
                    SilentGuard Automated Monitoring System<br>
                    <a href="#">View Dashboard</a>
                </div>
            </div>
        </body>
        </html>
        """

        # Mailtrap SDK Sending Logic
        token = CONFIG['email'].get('api_token')
        if not token or token == "your_mailtrap_api_token_here":
             ts = datetime.datetime.now().strftime("%H:%M:%S")
             st.session_state.email_logs.appendleft(f"[{ts}] ⚠️ Mailtrap Token Missing (Simulated Send)")
             st.session_state['last_email_time'] = time.time()
             return

        # Prepare Mail Object
        sender_email = CONFIG['email'].get('sender_email', 'hello@demomailtrap.co')
        
        # Create recipient objects
        to_addresses = [mt.Address(email=r) for r in recipients]
        
        mail = mt.Mail(
            sender=mt.Address(email=sender_email, name="SilentGuard AI Monitor"),
            to=to_addresses,
            subject=f"🚨 {severity}: SilentGuard Alert (Score: {score})",
            html=html_content,
            category="System Alert"
        )

        # Create Client and Send
        client = mt.MailtrapClient(token=token)
        response = client.send(mail)
        
        print(f"Mailtrap Response: {response}")
            
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.email_logs.appendleft(f"[{ts}] 📧 Alert Sent via Mailtrap SDK")
        st.session_state['last_email_time'] = time.time()

    except Exception as e:
        st.session_state.email_logs.appendleft(f"❌ Email Error: {str(e)}")
        print(f"Email failed: {e}")

# --- 5. HELPER: INPUT SIMULATION  ---
def get_synthetic_input(step):
    feats = np.random.normal(0, 1, 24)
    if step > 100: feats[0] += (step - 100) * 0.05 # Drift
    return feats.reshape(1, -1)

# --- 6. STREAMLIT UI ---
st.set_page_config(page_title="AI Model Monitoring Dashboard", layout="wide")

st.title("AI Model Monitoring Dashboard")

# Sidebar
st.sidebar.header("Control Panel")

# Initialize Monitor in Session State (Persistence Fix)
if 'monitor' not in st.session_state:
    st.session_state.monitor = ModelMonitor(CONFIG['model_path'])

# Persistent monitoring run flag
if 'monitoring_run' not in st.session_state:
    st.session_state.monitoring_run = False

# Persistent calibration & baseline state (fixed to avoid recalibration on reruns)
if 'baseline_learned' not in st.session_state:
    st.session_state.baseline_learned = False
if 'calibration_count' not in st.session_state:
    st.session_state.calibration_count = 0
if 'baseline_stats' not in st.session_state:
    st.session_state.baseline_stats = {}

# If the baseline was already learned in session state, ensure the Monitor reflects that
if st.session_state.baseline_learned:
    st.session_state.monitor.baseline_learned = True
    st.session_state.monitor.baseline_stats = st.session_state.baseline_stats
    # Ensure the calibration counter shows completed progress so UI doesn't re-run calibration
    st.session_state.calibration_count = st.session_state.monitor.warmup_target

def toggle_run(): st.session_state.monitoring_run = not st.session_state.monitoring_run

# Control Buttons
if st.sidebar.button("Start Monitoring"):
    st.session_state.monitoring_run = True
    st.rerun()

if st.sidebar.button("Stop Monitoring"):
    st.session_state.monitoring_run = False
    st.rerun()




# Reset Button
if st.sidebar.button("🔄 RESET SYSTEM"):
    # Full system reset
    st.session_state.monitor = ModelMonitor(CONFIG['model_path'])
    st.session_state.monitoring_run = False
    # Reset persistent baseline state as well
    st.session_state.baseline_learned = False
    st.session_state.calibration_count = 0
    st.session_state.baseline_stats = {}
    st.rerun()

# Reset baseline only (do not restart whole system)
# Reset baseline only (do not restart whole system)
if st.sidebar.button("🔁 RESET BASELINE"):
    st.session_state.baseline_learned = False
    st.session_state.calibration_count = 0
    st.session_state.baseline_stats = {}
    
    # Delete the persistent file
    try:
        baseline_file = st.session_state.monitor._get_baseline_path()
        if os.path.exists(baseline_file):
            os.remove(baseline_file)
            st.toast("Baseline file deleted", icon="🗑️")
    except:
        pass

    # Clear internal monitor warmup to restart calibration cleanly
    st.session_state.monitor.warmup_data = []
    st.session_state.monitor.baseline_stats = {}
    st.session_state.monitor.baseline_learned = False
    st.rerun()


# Documentation
with st.sidebar.expander("ℹ️ How it Works"):
    st.markdown("""
    **1. Performance**: Checks if the model output is drifting (shifting values) or getting slow (latency).
    **2. Stability (The "Jitter"):** High variance or sensitivity means model instability.
    **3. Safety:** Uncertainty and Margin to failure.
    """)

# --- EMAIL CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.header("📧 Email Alerts")

# 1. State Init
if 'email_enabled' not in st.session_state:
    st.session_state.email_enabled = CONFIG['email'].get('enabled', False)
if 'email_recipients' not in st.session_state:
    st.session_state.email_recipients = CONFIG['email'].get('recipients', ['admin@example.com'])
if 'email_logs' not in st.session_state:
    st.session_state.email_logs = deque(maxlen=5)

# 2. Toggle
enable_email = st.sidebar.toggle("Enable Email Alerts", value=st.session_state.email_enabled)
st.session_state.email_enabled = enable_email

status_icon = "✅" if enable_email else "⏸️"
st.sidebar.caption(f"Status: {status_icon} {'Enabled' if enable_email else 'Disabled'}")

# 3. Settings Expander
with st.sidebar.expander("⚙️ Email Configuration"):
    st.markdown(f"**API Token:** `{'*' * 10 if CONFIG['email'].get('api_token') else 'Missing'}`")
    st.markdown(f"**Sender:** `{CONFIG['email'].get('sender_email')}`")
    
    # Recipient Management (State-only override for session)
    recipients_str = st.text_input("Recipients (comma separated)", 
                                  value=",".join(st.session_state.email_recipients))
    if recipients_str:
        st.session_state.email_recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]

# 4. Test Email Logic
def send_test_email():
    try:
        # Use AUTHENTIC DATA from Session State
        if 'monitor' not in st.session_state:
             return False, "System not initialized. Please run monitor first."
             
        monitor = st.session_state.monitor
        
        # Construct current snapshot
        # If we have no history, providing empty metrics which will show as 0s in email
        if not monitor.history['predictions']:
             metrics = {k: 0.0 for k in ['value', 'latency', 'cpu_delta', 'mem_delta', 'variance', 'sensitivity', 'uncertainty', 'margin']}
             step = 0
             score = 100
             alerts = ["TEST ALERT: System is IDLE (Run monitor for real data)"]
        else:
             # Grab the very last frame of data from history
             metrics = {
                 'value': monitor.history['predictions'][-1],
                 'latency': monitor.history['latency'][-1],
                 'cpu_delta': monitor.history['cpu_delta'][-1],
                 'mem_delta': monitor.history['mem_delta'][-1],
                 'variance': monitor.history['variance'][-1],
                 'sensitivity': monitor.history['sensitivity'][-1],
                 'uncertainty': monitor.history['uncertainty'][-1],
                 'margin': monitor.history['margin'][-1],
             }
             step = monitor.history['steps'][-1]
             score = monitor.history['health'][-1]
             alerts = ["TEST ALERT: Snapshot of CURRENT system state"]

        # Send using force=True to bypass cooldown
        send_email_alert(alerts, score, step, metrics, monitor, force=True)
        return True, "Rich Alert Sent (Authentic Data)"
    except Exception as e:
        return False, str(e)

if st.sidebar.button("📨 Send Test Email"):
    with st.spinner("Sending Rich Alert..."):
        success, msg = send_test_email()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        if success:
            st.sidebar.success(f"Sent! ({ts})")
        else:
            st.sidebar.error("Failed")
            st.session_state.email_logs.appendleft(f"[{ts}] ❌ Test Failed: {msg}")

# 5. Email Logs
if st.session_state.email_logs:
    st.sidebar.markdown("**Recent Logs:**")
    for log in st.session_state.email_logs:
        st.sidebar.caption(log)



# Layout - Define Placeholders OUTSIDE the loop
c1, c2 = st.columns([3, 1])
info_ph = c1.empty() # Main info placeholder
gauge_ph = c2.empty()


# Calibration UI Placeholders (Dynamic) - Use columns/container to stack them
calibration_container = st.container()
with calibration_container:
    cal_banner_ph = st.empty()
    cal_progress_ph = st.empty()
    cal_stats_ph = st.empty()


tabs = st.tabs(["📊 PERFORMANCE", "📉 STABILITY", "🛡️ SAFETY", "💻 RESOURCES"])

# Tab 1: Performance
t1c1, t1c2 = tabs[0].columns(2)
chart_ph_pred = t1c1.empty()
chart_ph_lat = t1c2.empty()

# Tab 2: Stability
t2c1, t2c2 = tabs[1].columns(2)
chart_ph_var = t2c1.empty()
chart_ph_sens = t2c2.empty()

# Tab 3: Safety
t3c1, t3c2 = tabs[2].columns(2)
chart_ph_unc = t3c1.empty()
chart_ph_mar = t3c2.empty()

# Tab 4: Resources
t4c1, t4c2 = tabs[3].columns(2)
chart_ph_cpu = t4c1.empty()
chart_ph_mem = t4c2.empty()


alert_log = st.empty()

# Chart Helper (Dark Theme & High Visibility)
def plot_chart(key, title, color, placeholder, monitor, step, y_label):
    if not monitor.history[key]: return
    
    # Use step count for X axis if available
    steps = list(monitor.history['steps']) if monitor.history['steps'] else list(range(len(monitor.history[key])))
    df = pd.DataFrame({'Step': steps, 'Value': list(monitor.history[key])})
    
    # Add "(Learning...)" to title if calibrating
    final_title = title
    if not monitor.baseline_learned:
        final_title += " <span style='color:#FFC107; font-size:0.8em;'>(Learning...)</span>"
    
    fig = px.line(df, x='Step', y='Value', title=f"<b>{final_title}</b>")
    fig.update_traces(line=dict(color=color, width=3)) # 3px Bold Line
    
    # Dark Theme Layout
    fig.update_layout(
        font=dict(family="Arial", size=14, color="#FAFAFA"), # White text
        title_font=dict(size=18, family="Arial Black"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
        showlegend=False,
        plot_bgcolor='rgba(14, 17, 23, 1)', # Match app background
        paper_bgcolor='rgba(14, 17, 23, 1)',
        xaxis=dict(
            title="<b>Simulation Step</b>", 
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)', # White transparent grid
            gridwidth=1
        ),
        yaxis=dict(
            title=f"<b>{y_label}</b>", 
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)', 
            gridwidth=1
        )
    )
    
    if monitor.baseline_learned and key in monitor.baseline_stats:
        m = monitor.baseline_stats[key]['mean']
        s = monitor.baseline_stats[key]['std']
        fig.add_hline(y=m, line_dash="dash", line_width=2, line_color="#00FF00", opacity=0.7, annotation_text="Base")
        fig.add_hline(y=m+3*s, line_dash="dot", line_width=2, line_color="#FF4B4B", annotation_text="Limit")

    # Use unique key per step to prevent DuplicateKey error (Streamlit requires unique keys in a loop)
    placeholder.plotly_chart(fig, use_container_width=True, key=f"{key}_chart_{step}")

# Main Loop
if st.session_state.monitoring_run:
    monitor = st.session_state.monitor # USE PERSISTED MONITOR
    
    # Ensure step is tracked in session too or derived from history
    step = monitor.history['steps'][-1] + 1 if monitor.history['steps'] else 0
    
    full_alert_history = deque(maxlen=50) # Alert log local display
    
    while st.session_state.monitoring_run:
        # Run one monitoring step and then sync calibration progress from monitor
        metrics, score, alerts = monitor.update(get_synthetic_input(step), step)

        # Keep session-level calibration counter strictly in sync with monitor's warmup data
        if not monitor.baseline_learned:
            st.session_state.calibration_count = min(len(monitor.warmup_data), monitor.warmup_target)
        else:
            # if monitor finalized baseline ensure session shows full progress
            st.session_state.calibration_count = monitor.warmup_target

        # When the Monitor finalizes baseline, persist it into session_state (once)
        if monitor.baseline_learned and not st.session_state.baseline_learned:
            st.session_state.baseline_learned = True
            st.session_state.baseline_stats = monitor.baseline_stats

        # --- CALIBRATION UI ---
        # Use session_state as authoritative source for whether baseline is learned
        if not st.session_state.baseline_learned:
            curr = st.session_state.calibration_count
            target = monitor.warmup_target
            prog = min(1.0, curr / target)

            # Update banner and progress in fixed placeholders (do not create new elements)
            cal_banner_ph.markdown(f"""
            <div style="background-color:#2c2f38; padding:15px; border-radius:10px; border-left:6px solid #6610f2; margin-bottom:15px;">
                <h3 style="margin:0; color:#b366ff;">🛠️ SYSTEM CALIBRATION IN PROGRESS</h3>
                <p style="margin:0; color:#ddd;">Establishing baseline behavior for anomaly detection...</p>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar (update in-place)
            cal_progress_ph.progress(prog)
            cal_stats_ph.empty()

            # Real-time Stats Grid (use warmup samples collected so far)
            if monitor.warmup_data:
                df_w = pd.DataFrame(monitor.warmup_data)
                # Build a compact stats block and update the stats placeholder
                s = "**📉 Real-time Learning Statistics**\n"
                s += "\n"
                stat_cols = cal_stats_ph.columns(4)

                # Helper to show stats
                def show_stat(col, label, key, fmt="{:.2f}"):
                    if key in df_w:
                        avg = df_w[key].mean()
                        std = df_w[key].std()
                        col.metric(label, fmt.format(avg), f"±{std:.2f}")

                show_stat(stat_cols[0], "Avg Prediction", "value")
                show_stat(stat_cols[1], "Avg Latency", "latency", "{:.0f} ms")
                show_stat(stat_cols[2], "Avg CPU", "cpu_delta", "{:.1f}%")
                show_stat(stat_cols[3], "Avg Margin", "margin")
        else:
            # Baseline Established Message (Persistent and driven by session state)
            # Baseline Established Message (Persistent and driven by session state)
            source_info = getattr(monitor, 'baseline_source', 'Session')
            if "Loaded" in source_info:
                ts = source_info.split('(')[-1].strip(')')
                msg = f"✅ BASELINE ACTIVE (Loaded: {ts})"
            elif "Saved" in source_info:
                ts = source_info.replace("Saved ", "")
                msg = f"✅ BASELINE ACTIVE (Saved: {ts})"
            else:
                msg = f"✅ BASELINE ESTABLISHED ({source_info})"
                
            cal_banner_ph.success(msg)
            cal_progress_ph.empty()
            cal_stats_ph.empty()


        # --- NORMAL METRICS ---
        with info_ph.container():
            # Display Model Name & Type
            model_name = os.path.basename(CONFIG['model_path'])
            st.markdown(f"**🤖 Model:** `{model_name}` | **Type:** `{monitor.model_info['type']}`")
            
            st.markdown(f"### ⏱️ Last Prediction: **{metrics['value']:.2f} cycles**")
            status = "🟢 OPERATIONAL" if score > 70 else "🔴 DEGRADED"
            st.markdown(f"**System Status:** {status} | **Step:** {step}")

        # Gauge — color scheme: white for healthy, faded-orange for warning, red for critical
        if score > 70:
            border_color = "#FFFFFF"
            value_color = "#FFFFFF"
        elif score > 40:
            border_color = "#FFDCC8"  # faded orange (light)
            value_color = "#F6AD83"   # orange value for emphasis
        else:
            border_color = "#DC3545"
            value_color = "#DC3545"

        gauge_ph.markdown(f"""
        <div style="text-align:center; background:#262730; border:3px solid {border_color}; border-radius:12px; padding:15px; box-shadow:0 4px 8px rgba(0,0,0,0.2);">
            <h3 style="margin:0; color:#FAFAFA; font-weight:900;">HEALTH</h3>
            <h1 style="font-size:4em; margin:0; color:{value_color}; font-weight:900;">{int(score)}%</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Update Charts (Pass Placeholders & Labels)
        plot_chart('predictions', 'RUL PREDICTION', '#4da6ff', chart_ph_pred, monitor, step, "RUL (Cycles)")
        plot_chart('latency', 'INFERENCE LATENCY', '#b366ff', chart_ph_lat, monitor, step, "Time (ms)")
        
        plot_chart('variance', 'PREDICTION VARIANCE', '#ffa64d', chart_ph_var, monitor, step, "Variance (σ²)")
        plot_chart('sensitivity', 'INPUT SENSITIVITY', '#ff66b3', chart_ph_sens, monitor, step, "Sensitivity Score")
        
        plot_chart('uncertainty', 'MODEL UNCERTAINTY', '#a64dff', chart_ph_unc, monitor, step, "Uncertainty (StdDev)")
        plot_chart('margin', 'SAFETY MARGIN', '#4dffb3', chart_ph_mar, monitor, step, "Time to Failure")
        
        plot_chart('cpu_delta', 'CPU USAGE', '#4dffff', chart_ph_cpu, monitor, step, "Utilization (%)")
        plot_chart('mem_delta', 'MEMORY USAGE', '#cccccc', chart_ph_mem, monitor, step, "Usage (MB)")

        # Alerts
        if alerts:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            for a in alerts: full_alert_history.appendleft(f"[{ts}] {a}")
            # Trigger Real Rich Email Alert
            send_email_alert(alerts, score, step, metrics, monitor)
            
        h = ""
        for m in list(full_alert_history)[:5]:
            s = "alert-critical" if "CRITICAL" in m else "alert-warning"
            h += f"<div class='alert-box {s}'>{m}</div>"
        alert_log.markdown(f"### 🚨 INCIDENT LOG\n{h}", unsafe_allow_html=True)
        
        step += 1
        time.sleep(1.5)
        
        if not st.session_state.monitoring_run: break

