# Telecom Customer Intelligence & Churn Analytics Platform

An end-to-end professional telecom analytics platform engineered to predict customer churn, monetize revenue-at-risk, uncover behavioral segments, and serve real-time predictions via an interactive web interface.

## Business Value & Objectives
Customer churn erodes recurring revenue in subscription-based businesses. This platform moves beyond simple classification by translating raw tabular data into actionable retention strategies:
- **Proactive Churn Prediction:** Identifies high-risk customers before cancellation using advanced machine learning.
- **Revenue Monetization:** Quantifies exact financial exposure using predicted probabilities and monthly billing data.
- **Behavioral Segmentation:** Discovers underlying customer personas via unsupervised learning to tailor retention campaigns.

## Tech Stack & Architecture
- **Data Processing & Storage:** Python, DuckDB, Pandas, Parquet
- **Machine Learning:** Scikit-learn (Pipelines, Logistic Regression, Random Forest), XGBoost, Permutation Importance
- **API & Backend:** FastAPI, Uvicorn, Pydantic
- **Dashboard UI:** Streamlit, Plotly

## Key Results & Modeling Performance
Models were evaluated using rigorous cross-validation with leakage-safe preprocessing pipelines:
- **XGBoost (Selected Model):** ROC-AUC: **0.858**, PR-AUC: **0.670**
- **Random Forest:** ROC-AUC: **0.838**, PR-AUC: **0.631**
- **Logistic Regression (Baseline):** ROC-AUC: **0.836**, PR-AUC: **0.649**
- **Targeting Efficiency:** The top 10% high-risk target list successfully captured **75.0%** of actual churners, protecting over **$332,000** in annualized revenue exposure.

## Unsupervised Segmentation (K-Means + PCA)
Clustering analysis identified three distinct customer segments:
- **Segment 0 (The Flight Risk):** High-risk, short-tenure (13 months) accounts with moderate spend and low service adoption.
- **Segment 1 (The Power User):** Loyal, long-tenure (57 months) accounts generating high monthly revenue ($91) with deep service engagement.
- **Segment 2 (The Budget Utility):** Highly stable, low-spend ($31) accounts using basic services.

## Repository Structure
```text
telecom-customer-intelligence/
├── app/
│   ├── api.py           # FastAPI prediction backend
│   └── dashboard.py     # Streamlit analytics interface
├── data/
│   ├── processed/       # Cleaned and scored Parquet datasets
│   └── raw/             # Original raw data sources
├── models/              # Serialized joblib model binaries
├── reports/             # Exported targeting lists & metrics CSVs
└── src/
    ├── features/        # Leakage-safe feature engineering pipelines
    └── models/          # Training, tuning, scoring, and explainability scripts
