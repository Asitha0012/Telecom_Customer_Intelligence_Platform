import pandas as pd
import numpy as np
import joblib
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from src.features.build_features import make_features

# 1. Load Data and Model
df_raw = pd.read_parquet("data/processed/telco_clean.parquet")
df = make_features(df_raw)

for col in df.select_dtypes(exclude="number").columns:
    df[col] = df[col].astype(object)
    df.loc[df[col].isna(), col] = np.nan

y = (df.pop("Churn") == "Yes").astype(int)
leakage_cols = ["customerID", "Churn Value", "Churn Score", "Churn Reason"]
cols_to_drop = [c for c in leakage_cols if c in df.columns]
df = df.drop(columns=cols_to_drop)

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, stratify=y, random_state=42
)

model = joblib.load("models/churn_xgb.joblib")

# 2. Global Explainability (Permutation Importance)
print("Calculating permutation importance (this takes a moment)...")
r = permutation_importance(
    model, X_test, y_test, scoring="average_precision",
    n_repeats=5, random_state=42, n_jobs=1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": r.importances_mean
}).sort_values(by="Importance", ascending=False)

print("\n--- Top 5 Global Churn Drivers ---")
print(importance_df.head(5).to_string(index=False))

# 3. Individual Case Explanation
# Select the highest risk customer from the test set
probs = model.predict_proba(X_test)[:, 1]
high_risk_idx = np.argmax(probs)
customer_data = X_test.iloc[high_risk_idx]
customer_prob = probs[high_risk_idx]

def risk_band(p):
    if p >= 0.75: return "Critical"
    if p >= 0.50: return "High"
    if p >= 0.25: return "Medium"
    return "Low"

print("\n--- Individual Case Explanation ---")
print(f"Customer ID (Index): {X_test.index[high_risk_idx]}")
print(f"Churn probability: {customer_prob:.2f}")
print(f"Risk band: {risk_band(customer_prob)}")
print("\nTop drivers for this customer (based on global importance):")
for feature in importance_df.head(4)["Feature"]:
    print(f" - {feature}: {customer_data[feature]}")

print("\nRecommended retention action: review contract upgrade offer + support onboarding")