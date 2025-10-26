import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# Step 1: Load Feature-Engineered Data
# ===============================
print("🚀 Loading feature-engineered data...")
data = pd.read_csv("data/cleaned_churn_data.csv")
print(f"Initial shape: {data.shape}")

# Split into features (X) and target (y)
X = data.drop("Target_Churn", axis=1)
y = data["Target_Churn"]

# ===============================
# Step 2: Split into Train & Test Sets
# ===============================
print("\n📊 Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ===============================
# Step 3: Scale the Features
# ===============================
print("\n⚙️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, "models/scaler.pkl")

# ===============================
# Step 4: Initialize Models
# ===============================
print("\n🧠 Training models...")
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
}

results = {}

# ===============================
# Step 5: Train and Evaluate Each Model
# ===============================
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ {name} Performance:")
    print("Accuracy:", round(acc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    results[name] = acc

# ===============================
# Step 6: Select and Save the Best Model
# ===============================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\n🏆 Best Model:", best_model_name, "with Accuracy:", round(results[best_model_name], 4))

# Save model to file
joblib.dump(best_model, f"models/{best_model_name.replace(' ', '_').lower()}_model.pkl")

print(f"✅ Saved best model to models/{best_model_name.replace(' ', '_').lower()}_model.pkl")
print("\n🎯 Model training and evaluation complete!")
