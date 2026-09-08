from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.features.build_features import make_features

app = FastAPI(title="Telecom Customer Intelligence API")
MODEL = joblib.load(Path("models/churn_xgb.joblib"))

# Define the expected input payload based on raw data columns
class Customer(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 840.0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(c: Customer):
    # 1. Convert input to DataFrame
    df = pd.DataFrame([c.model_dump()])
    
    # 2. Apply feature engineering
    X = make_features(df)
    
    # 3. Predict using the loaded XGBoost model
    prob = float(MODEL.predict_proba(X)[:,1][0])
    
    return {
        "churn_probability": round(prob, 4), 
        "risk_band": "Critical" if prob >= 0.75 else "High" if prob >= 0.5 else "Medium" if prob >= 0.25 else "Low"
    }