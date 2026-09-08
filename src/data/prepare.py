from pathlib import Path
import pandas as pd

RAW = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUT = Path("data/processed/telco_clean.parquet")

df = pd.read_csv(RAW)

# Map the 33-column dataset names to the guide's exact expected schema
rename_map = {
    "CustomerID": "customerID",
    "Tenure Months": "tenure",
    "Churn Label": "Churn",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Senior Citizen": "SeniorCitizen"
}
df = df.rename(columns=rename_map)

# Correct known numeric field
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove impossible duplicate customer records
df = df.drop_duplicates(subset=["customerID"]).copy()

# Treat blank/whitespace categorical values as missing
obj_cols = df.select_dtypes(include="object").columns
for c in obj_cols:
    df[c] = df[c].astype("string").str.strip()

# Use 0 only after verifying tenure == 0; otherwise keep NaN for audit.
mask = df["TotalCharges"].isna() & (df["tenure"] == 0)
df.loc[mask, "TotalCharges"] = 0.0

# Validation checks
assert df["customerID"].is_unique
assert set(df["Churn"].dropna().unique()) <= {"Yes", "No"}
assert (df["MonthlyCharges"] >= 0).all()
assert (df["TotalCharges"] >= 0).all()

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print("saved", OUT, df.shape)