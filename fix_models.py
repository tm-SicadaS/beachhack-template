import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Generate Dummy Training Data
# We mimic your system metrics: [CPU, Memory, Latency, Error_Rate]
rng = np.random.RandomState(42)
X_train = 0.3 * rng.randn(100, 4)
X_train = np.r_[X_train + 2, X_train - 2]  # Create two clusters of "normal" data

# 2. Train the Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# 3. Train the Model (Isolation Forest is great for anomaly detection)
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X_scaled)

# 4. Save them properly using Pickle
print("Saving new model files...")
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ SUCCESS! 'model.pkl' and 'scaler.pkl' have been regenerated.")