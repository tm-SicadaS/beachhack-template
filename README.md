# Pre Incident Failure Detection System by SicadaS

## Selected Problem Statement

Modern AI/ML systems and production applications often fail silently. Instead of crashing, they continue running while performance degrades due to latency spikes, memory issues, error rate increase, or unseen data drift. Existing monitoring tools rely on ground-truth labels or manual thresholds, which leads to delayed detection and business impact.

---

## Project Overview

Silent Model Failure Detection System is a multi-metric anomaly detection platform that identifies early signs of system degradation before complete failure occurs.

Instead of waiting for accuracy reports or manual alerts, our system:

* Learns what “normal system behaviour” looks like
* Monitors latency, memory usage, CPU usage, and error rate
* Detects deviations using unsupervised machine learning
* Classifies health status as:

  * 🟢 Normal
  * 🟡 Warning
  * 🔴 Critical

The system works using batch statistical monitoring and anomaly detection models trained on baseline (healthy) data.

---

## Technical Approach

### 1. Baseline Learning

* Healthy system metrics are extracted from dataset
* Isolation Forest (unsupervised ML) is trained on normal behaviour
* The model learns multi-dimensional patterns instead of static thresholds

### 2. Multi-Metric Monitoring

The system monitors:

* Latency
* Memory Usage
* CPU Usage
* Error Rate

Instead of checking individual thresholds, it detects abnormal combinations of these metrics.

### 3. Batch Monitoring Engine

* Data is processed in batches (50–100 samples)
* Each batch is evaluated using the trained model
* If anomaly percentage crosses threshold → system triggers alert

### 4. Severity Classification

| Condition            | Status   |
| -------------------- | -------- |
| No anomalies         | Normal   |
| Few anomalies        | Warning  |
| High anomaly density | Critical |

---

## Tech Stack

* Python
* Pandas & NumPy (data handling)
* Scikit-learn (Isolation Forest model)
* Matplotlib (visualization)
* Joblib (model saving/loading)
* VS Code / Google Colab (development environment)

---

## Setup Instructions

### 1. Clone Repository

```
git clone <your-repo-link>
cd Beachhack-template
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

```
/models/baseline_model.pkl
```

### 5. Run Monitoring System

```
python run.py
```

The system will:

* Process metrics in batches
* Detect anomalies
* Display system health status

---

## Demo

Live Demo : TO BE ADDED

Example Output:

* Normal Batch → Status: NORMAL
* Failure Injected Batch → Status: CRITICAL

---

## Screenshots

<img width="1882" height="855" alt="Screenshot 2026-01-31 200015" src="https://github.com/user-attachments/assets/b67072b9-541a-40ff-9b88-a89048504a74" />

<img width="2963" height="2364" alt="feature_importance_rul" src="https://github.com/user-attachments/assets/67eb4a40-4477-4a86-b133-cd522775d2e7" />
<img width="4170" height="3014" alt="rul_prediction_analysis" src="https://github.com/user-attachments/assets/3c123e73-9e70-4cee-ba7d-e4d39a603873" />

---

## Why This Matters

Silent degradation causes:

* Increased cloud costs
* SLA violations
* Poor user experience
* Revenue loss

Our solution detects early degradation before catastrophic failure, making it suitable for AI pipelines, cloud systems, and production services.


## Docker Deployment

### Pull the image
Pull the published image:
```bash
docker pull r0xh4n/silentguard:v1.0
```

### Run (recommended)
Mount the client's model folder read-only and set MODEL_PATH:
```bash
docker run -d --name silentguard_monitor \
  -p 8501:8501 \
  -v /opt/secure/models:/app/models:ro \
  -e MODEL_PATH=/app/models/my_model.pkl \
  r0xh4n/silentguard:v1.0
```
Access the dashboard at: http://localhost:8501

### Docker Compose (client example)
```yaml
version: '3.8'
services:
  silentguard:
    image: r0xh4n/silentguard:v1.0
    container_name: silentguard_monitor
    ports:
      - "8501:8501"
    volumes:
      - /opt/secure/models:/app/models:ro
    environment:
      - MODEL_PATH=/app/models/my_model.pkl
    restart: unless-stopped
```

### Offline distribution
Export and load a tar if the client is air-gapped:
```bash
docker save -o silentguard_v1.tar r0xh4n/silentguard:v1.0
# On client:
docker load -i silentguard_v1.tar
```

### Security notes
- Do **NOT** include `.pkl`/`.joblib` files or secrets inside the image.
- Mount model artifacts at runtime with `:ro`.
- Use a `.dockerignore` to exclude local models and secrets when building.

---
