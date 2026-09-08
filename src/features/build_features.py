import pandas as pd

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["tenure_years"] = x["tenure"] / 12.0
    x["avg_monthly_spend"] = x["TotalCharges"] / x["tenure"].clip(lower=1)
    x["service_count"] = (x[SERVICE_COLS].eq("Yes")).sum(axis=1)
    x["is_month_to_month"] = (x["Contract"] == "Month-to-month").astype(int)
    x["has_support"] = (x["TechSupport"] == "Yes").astype(int)
    x["has_security"] = (x["OnlineSecurity"] == "Yes").astype(int)
    x["auto_pay"] = x["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
    x["charge_per_tenure"] = x["MonthlyCharges"] / x["tenure"].clip(lower=1)
    return x