import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Load Dataset
data = pd.read_csv("data/loan_dataset.csv")

# Convert object columns to category
categorical_cols = data.select_dtypes(include=["object", "string"]).columns.tolist()
categorical_cols.append("BAD")

for col in categorical_cols:
    data[col] = data[col].astype("category")

# Data Preparation

# Target variable
y = data["BAD"]

# Feature matrix
X = data.drop(columns=["BAD"])

# One-Hot Encoding
X = pd.get_dummies(
    X,
    columns=["REASON", "JOB"],
    drop_first=True
)

# Final Tuned Random Forest

rf_model = RandomForestClassifier(
    n_estimators=250,
    max_features=0.9,
    min_samples_leaf=1,
    random_state=1
)

# Train model on complete dataset

rf_model.fit(X, y)

# Save trained model

joblib.dump(rf_model, "model/loan_model.pkl")

# Save metadata for Streamlit

metadata = {
    "feature_columns": X.columns.tolist(),
    "reason_categories": data["REASON"].cat.categories.tolist(),
    "job_categories": data["JOB"].cat.categories.tolist()
}

joblib.dump(metadata, "model/metadata.pkl")

print("Model trained successfully!")
print("Model saved to: model/loan_model.pkl")
print("Metadata saved to: model/metadata.pkl")