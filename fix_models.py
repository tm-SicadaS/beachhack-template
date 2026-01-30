import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Generate Robust Training Data (Uniform Distribution)
# "Uniform" means any value between the low and high is EQUALLY normal.
# This prevents the model from thinking 80% CPU is "weird" just because the average is 30%.
np.random.seed(42)
n_samples = 2000

# CPU: Anything between 5% and 90% is completely normal
cpu = np.random.uniform(low=5, high=90, size=n_samples)

# Memory: Anything between 20% and 90% is completely normal
memory = np.random.uniform(low=20, high=90, size=n_samples)

# Latency: Normal operations can range widely from 10ms to 400ms
latency = np.random.uniform(low=10, high=400, size=n_samples)

# Error Rate: Still heavily biased towards 0, but allow small blips
# 95% of data has 0 error, 5% has small error (0-3%)
error_rate = np.zeros(n_samples)
n_errors = int(0.05 * n_samples)
error_indices = np.random.choice(n_samples, n_errors, replace=False)
error_rate[error_indices] = np.random.uniform(0, 3, size=n_errors) # Small errors are normal-ish

X_train = np.column_stack((cpu, memory, latency, error_rate))

# 2. Train the Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# 3. Train the Model (Isolation Forest)
# We set contamination very low because our training data is DEFINED as normal.
clf = IsolationForest(contamination=0.001, random_state=42) # 0.1% outlier rate
clf.fit(X_scaled)

# 4. Save
print("Saving robust model files...")
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ SUCCESS! Robust model regenerated with UNIFORM distributions.")
print(f"Ranges trained: CPU=[{cpu.min():.1f}-{cpu.max():.1f}], Lat=[{latency.min():.1f}-{latency.max():.1f}]")