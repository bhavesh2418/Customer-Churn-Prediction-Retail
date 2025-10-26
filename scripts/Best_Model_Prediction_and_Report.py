# ===============================================================
# 04_Best_Model_Prediction_and_Report.py
# ===============================================================

import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from fpdf import FPDF
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================================
# Set Project Paths
# ===============================================================
project_dir = r"F:\CustomerChurnProject"
os.chdir(project_dir)

data_path = os.path.join("data", "cleaned_churn_data.csv")
models_dir = os.path.join("models")
report_dir = os.path.join("reports")
os.makedirs(report_dir, exist_ok=True)

# ===============================================================
# Load Data
# ===============================================================
data = pd.read_csv(data_path)
X = data.drop("Target_Churn", axis=1)
y_true = data["Target_Churn"]

# ===============================================================
# Identify Best Model (highest accuracy .pkl in models/)
# ===============================================================
model_files = [f for f in os.listdir(models_dir) if f.endswith("_model.pkl")]
if not model_files:
    raise FileNotFoundError("No model .pkl files found in 'models/' folder.")

# For simplicity, pick the first model (or implement logic to choose best)
best_model_file = model_files[0]
model = joblib.load(os.path.join(models_dir, best_model_file))

# Load Scaler
scaler_path = os.path.join(models_dir, "scaler.pkl")
scaler = joblib.load(scaler_path)
X_scaled = scaler.transform(X)

# ===============================================================
# Make Predictions
# ===============================================================
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None

# ===============================================================
# Evaluation Metrics
# ===============================================================
accuracy = accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
cm = confusion_matrix(y_true, y_pred)
report_text = classification_report(y_true, y_pred)

# ===============================================================
# Save Predictions CSV
# ===============================================================
results = X.copy()
results["Actual"] = y_true
results["Predicted"] = y_pred
if y_proba is not None:
    results["Predicted_Probability"] = y_proba

pred_csv_path = os.path.join(report_dir, f"{best_model_file.replace('.pkl','')}_Predictions.csv")
results.to_csv(pred_csv_path, index=False)

# ===============================================================
# Plot Confusion Matrix
# ===============================================================
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_file.replace('_model.pkl','')}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
cm_image_path = os.path.join(report_dir, f"{best_model_file.replace('.pkl','')}_Confusion_Matrix.png")
plt.savefig(cm_image_path)
plt.close()

# ===============================================================
# Generate PDF Summary Report
# ===============================================================
pdf_path = os.path.join(report_dir, f"{best_model_file.replace('.pkl','')}_Prediction_Report.pdf")
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, f"Prediction Report - {best_model_file.replace('_model.pkl','')}", ln=True, align="C")
pdf.ln(5)

pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 6, f"Accuracy: {accuracy:.4f}")
if roc_auc is not None:
    pdf.multi_cell(0, 6, f"ROC-AUC: {roc_auc:.4f}")
pdf.ln(5)
pdf.multi_cell(0, 6, "Classification Report:")
pdf.multi_cell(0, 6, report_text)

# Add Confusion Matrix Image
if os.path.exists(cm_image_path):
    pdf.ln(5)
    pdf.image(cm_image_path, w=pdf.w - 40)

# Add Sample Predictions Table
pdf.ln(5)
pdf.multi_cell(0, 6, "Sample Predictions (first 10 rows):")
sample_table = results.head(10).to_string(index=False)
pdf.set_font("Courier", '', 10)
pdf.multi_cell(0, 5, sample_table)

pdf.output(pdf_path)
print(f"✅ PDF report saved at: {pdf_path}")
print(f"✅ Predictions CSV saved at: {pred_csv_path}")
