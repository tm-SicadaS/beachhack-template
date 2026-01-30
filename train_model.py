import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------
# Generate REALISTIC dataset
# ---------------------------
np.random.seed(42)

X = np.random.normal(0, 1, (3000, 5))   # 5 features
y = (X.sum(axis=1) > 0).astype(int)     # binary target

# ---------------------------
# Train / Test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# ---------------------------
# Train model
# ---------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------
# Save model
# ---------------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained and saved successfully.")
