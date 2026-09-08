from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from src.features.build_features import make_features

# 1. Load and prepare data
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

# 4. Build and Train Random Forest Pipeline
model = Pipeline([
    ("prep", prep),
    ("clf", RandomForestClassifier(
        n_estimators=200, 
        max_depth=6, 
        class_weight="balanced", 
        random_state=42, 
        n_jobs=1
    ))
])

print("Training Random Forest Classifier...")
model.fit(X_train, y_train)

# 5. Evaluate the Model
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print("\n--- Random Forest Evaluation ---")
print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
print("PR-AUC:", average_precision_score(y_test, prob))

# 6. Save the Artifact
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/churn_rf.joblib")
print("\nSaved Random Forest model to models/churn_rf.joblib")