# Silent Model Failure Detection System (SMFDS)

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
cd SilentFailureDetection
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Add Dataset

Place your dataset inside:

```
/data/metrics.csv
```

Ensure dataset includes:

* latency
* memory
* cpu_usage
* error_rate

### 4. Train Baseline Model

Run:

```
python train_baseline.py
```

This generates:

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

Live Demo (if applicable):
Add your demo link here

Example Output:

* Normal Batch → Status: NORMAL
* Failure Injected Batch → Status: CRITICAL

---

## Screenshots

(Add screenshots of:)

* Dataset preview
* Training output
* Monitoring console output
* Graph showing anomaly detection

---

## Why This Matters

Silent degradation causes:

* Increased cloud costs
* SLA violations
* Poor user experience
* Revenue loss

Our solution detects early degradation before catastrophic failure, making it suitable for AI pipelines, cloud systems, and production services.

---

## Future Improvements

* Real-time streaming integration
* Slack / Email alert integration
* Dashboard using Streamlit or React
* Drift detection using KL Divergence
* Cloud deployment with Docker

---