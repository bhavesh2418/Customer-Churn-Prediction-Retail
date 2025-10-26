# ===============================================================
# 03_Model_Prediction_Example.py
# ===============================================================

import os
import pandas as pd
import joblib

# ===============================================================
# Set Project Paths
# ===============================================================
project_dir = r"F:\CustomerChurnProject"
os.chdir(project_dir)

data_path = os.path.join("data", "cleaned_churn_data.csv")   # Test / new data
model_path = os.path.join("models", "LogisticRegression_model.pkl")  # Change as needed

# ===============================================================
# Load Data
# ===============================================================
data = pd.read_csv(data_path)
X = data.drop("Target_Churn", axis=1)
y_true = data["Target_Churn"]

# ===============================================================
# Load Saved Model
# ===============================================================
model = joblib.load(model_path)

# ===============================================================
# Load Scaler (if features were scaled during training)
# ===============================================================
scaler_path = os.path.join("models", "scaler.pkl")
scaler = joblib.load(scaler_path)
X_scaled = scaler.transform(X)

# ===============================================================
# Make Predictions
# ===============================================================
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None

# ===============================================================
# Combine Results in a DataFrame
# ===============================================================
results = X.copy()
results["Actual"] = y_true
results["Predicted"] = y_pred

if y_proba is not None:
    results["Predicted_Probability"] = y_proba

# ===============================================================
# Save Predictions
# ===============================================================
output_path = os.path.join("reports", "Predictions_LogisticRegression.csv")
results.to_csv(output_path, index=False)

print(f"Predictions saved to: {output_path}")
print("\nSample Predictions:")
print(results.head(10))
