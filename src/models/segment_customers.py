import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from src.features.build_features import make_features

# 1. Load Scored Data
df_raw = pd.read_parquet("data/processed/scored_customers.parquet")

# Re-apply feature engineering just to extract the required clustering columns
df_features = make_features(df_raw)

# 2. Prepare clustering features
cluster_cols = ["tenure", "MonthlyCharges", "TotalCharges", "service_count", "is_month_to_month"]
Z = df_features[cluster_cols].copy()

# Handle any missing values (e.g., TotalCharges for brand-new customers)
Z = Z.fillna(Z.median())

# Scale features to mean=0, variance=1 for K-Means
Zs = StandardScaler().fit_transform(Z)

# 3. Find optimal K using Silhouette Score
print("Evaluating cluster numbers...")
models = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Zs)
    models[k] = (km, labels)
    print(f"k={k}, silhouette_score={silhouette_score(Zs, labels):.4f}")

# Select the k with the highest silhouette score
best_k = max(models, key=lambda k: silhouette_score(Zs, models[k][1]))
print(f"\nSelected best k={best_k}")

km, labels = models[best_k]
df_raw["segment"] = labels

# 4. PCA for 2D visualization in the dashboard
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(Zs)
df_raw["pca_x"] = coords[:, 0]
df_raw["pca_y"] = coords[:, 1]

# 5. Profile the clusters
print("\nCluster Profiles (sorted by churn rate):")
profile = df_raw.assign(
    service_count=df_features["service_count"]
).groupby("segment").agg(
    customers=("customerID", "count"),
    churn_rate=("churn_flag", "mean"),
    avg_tenure=("tenure", "mean"),
    avg_monthly_charges=("MonthlyCharges", "mean"),
    avg_services=("service_count", "mean")
).sort_values("churn_rate", ascending=False).round(2)

print(profile)

# 6. Save back to parquet with the new segment labels
df_raw.to_parquet("data/processed/scored_customers.parquet", index=False)
print("\nSaved segmented customers to data/processed/scored_customers.parquet")