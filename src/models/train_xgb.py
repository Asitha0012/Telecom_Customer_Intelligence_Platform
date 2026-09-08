from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from src.features.build_features import make_features

# 1. Load and prepare data (same as baseline)
df = pd.read_parquet("data/processed/telco_clean.parquet")
df = make_features(df)

for col in df.select_dtypes(exclude="number").columns:
    df[col] = df[col].astype(object)
    df.loc[df[col].isna(), col] = np.nan

y = (df.pop("Churn") == "Yes").astype(int)
leakage_cols = ["customerID", "Churn Value", "Churn Score", "Churn Reason"]
cols_to_drop = [c for c in leakage_cols if c in df.columns]
df = df.drop(columns=cols_to_drop)

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Define Preprocessing Pipelines
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

prep = ColumnTransformer([
    ("num", num_pipe, num_cols), 
    ("cat", cat_pipe, cat_cols)
])

# 4. Build XGBoost Pipeline
model = Pipeline([
    ("prep", prep),
    ("clf", XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",  # Laptop-friendly fast training
        random_state=42
    ))
])

# 5. Hyperparameter Tuning (Laptop-friendly grid)
param_grid = {
    "clf__n_estimators": [150, 250, 350],
    "clf__max_depth": [3, 4, 5],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__subsample": [0.7, 0.8, 1.0],
    "clf__colsample_bytree": [0.7, 0.8, 1.0]
}

print("Starting RandomizedSearchCV (this may take a minute...)")
search = RandomizedSearchCV(
    model, param_grid, n_iter=12, scoring="average_precision",
    cv=3, random_state=42, n_jobs=1, verbose=1
)

search.fit(X_train, y_train)
print("\nBest parameters found:", search.best_params_)

# 6. Evaluate Final Model
final_model = search.best_estimator_
prob = final_model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print("\n--- Final XGBoost Evaluation ---")
print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
print("PR-AUC:", average_precision_score(y_test, prob))

# 7. Save Final Artifact
Path("models").mkdir(exist_ok=True)
joblib.dump(final_model, "models/churn_xgb.joblib")
print("\nSaved final XGBoost model to models/churn_xgb.joblib")