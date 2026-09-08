from pathlib import Path
import pandas as pd

DATA = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(DATA)

print("shape:", df.shape)
print("columns:", list(df.columns))
print("duplicate rows:", df.duplicated().sum())

print("missing values:")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("target distribution (Churn Value):")
print(df["Churn Value"].value_counts(dropna=False, normalize=True))
print("\ntarget distribution (Churn Label):")
print(df["Churn Label"].value_counts(dropna=False, normalize=True))