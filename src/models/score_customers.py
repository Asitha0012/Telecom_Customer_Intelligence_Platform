from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.features.build_features import make_features

# 1. Load Data
df_raw = pd.read_parquet("data/processed/telco_clean.parquet")
df = make_features(df_raw)

# Fix pandas object types for scikit-learn
for col in df.select_dtypes(exclude="number").columns:
    df[col] = df[col].astype(object)
    df.loc[df[col].isna(), col] = np.nan

y = (df["Churn"] == "Yes").astype(int)
leakage_cols = ["Churn Value", "Churn Score", "Churn Reason"]
cols_to_drop = [c for c in leakage_cols if c in df.columns]
df = df.drop(columns=cols_to_drop)

# 2. Reproduce Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["customerID", "Churn"]), y, test_size=0.2, stratify=y, random_state=42
)

model = joblib.load("models/churn_xgb.joblib")

# 3. Evaluate Top-k on Held-Out Test Set
test_prob = model.predict_proba(X_test)[:, 1]

test_out = df_raw.loc[X_test.index, ["customerID", "MonthlyCharges"]].copy()
test_out["churn_flag"] = y_test.values
test_out["churn_probability"] = test_prob
test_out = test_out.sort_values("churn_probability", ascending=False)

k = int(len(test_out) * 0.10)
targeted = test_out.head(k)

print("--- Test Set Business Metrics ---")
print(f"10% targeted capacity: {k} customers")
print(f"Actual churners captured: {targeted['churn_flag'].sum()}")
print(f"Capture rate in target group: {targeted['churn_flag'].mean():.1%}")

# 4. Calculate Revenue at Risk
test_out["annual_revenue_exposure"] = test_out["MonthlyCharges"] * 12
test_out["revenue_at_risk"] = test_out["annual_revenue_exposure"] * test_out["churn_probability"]
print(f"Test Set Annualized Revenue at Risk Proxy: ${test_out['revenue_at_risk'].sum():,.0f}")

def risk_band(p):
    if p >= 0.75: return "Critical"
    if p >= 0.50: return "High"
    if p >= 0.25: return "Medium"
    return "Low"

test_out["risk_band"] = test_out["churn_probability"].map(risk_band)

print("\nRevenue at Risk by Band (Test Set):")
print(test_out.groupby("risk_band")["revenue_at_risk"].sum().sort_values(ascending=False).apply(lambda x: f"${x:,.0f}"))

# 5. Save Artifacts
Path("reports").mkdir(exist_ok=True)
test_out.to_csv("reports/churn_risk_scores.csv", index=False)

# Score Full Dataset for Dashboard
X_all = df.drop(columns=["customerID", "Churn"])
df_raw["churn_probability"] = model.predict_proba(X_all)[:, 1]
df_raw["churn_flag"] = y.values
df_raw["annual_revenue_exposure"] = df_raw["MonthlyCharges"] * 12
df_raw["revenue_at_risk"] = df_raw["annual_revenue_exposure"] * df_raw["churn_probability"]
df_raw["risk_band"] = df_raw["churn_probability"].map(risk_band)

df_raw.to_parquet("data/processed/scored_customers.parquet", index=False)
print("\nSaved targeting list to reports/churn_risk_scores.csv")
print("Saved full dataset to data/processed/scored_customers.parquet")