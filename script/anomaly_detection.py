import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import hdbscan
from umap import UMAP
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================
# 1. Load Data
# =====================
df = pd.read_csv("claim_anomalies/data/claims.csv")

# =====================
# 2. Configuration
# =====================
features = [
    "insured_amount", "claim_amount", "months_since_policy_start",
    "claim_hour", "previous_claim_count", "customer_seniority_years"
]
output_path = "claim_anomalies/data"
os.makedirs(output_path, exist_ok=True)

# =====================
# 3. Loop over coverages
# =====================
resultados = []

for cobertura in df["coverage_type"].unique():
    df_cov = df[df["coverage_type"] == cobertura].copy()

    if len(df_cov) < 50:
        continue

    print(f"Procesando cobertura: {cobertura} ({len(df_cov)} registros)")

    # Preprocessing
    X = df_cov[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    # Isolation Forest
    iso_model = IsolationForest(contamination=0.015, random_state=42)
    df_cov["anomaly_score"] = iso_model.fit_predict(X_scaled)
    df_cov["suspicion_score"] = iso_model.decision_function(X_scaled) * -1
    df_cov["is_suspicious"] = df_cov["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

    # UMAP
    reducer = UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    df_cov["UMAP_1"] = embedding[:, 0]
    df_cov["UMAP_2"] = embedding[:, 1]

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, prediction_data=True)
    df_cov["cluster"] = clusterer.fit_predict(embedding)

    # SHAP
    explainer = shap.Explainer(iso_model, X_scaled_df)
    df_cov = df_cov.reset_index(drop=True)
    X_scaled_df = X_scaled_df.reset_index(drop=True)

    # Top 100 (more suspicious)
    top_100_idx = df_cov.sort_values("suspicion_score", ascending=False).head(100).index
    X_top100 = X_scaled_df.loc[top_100_idx]

    shap_values_top100 = explainer(X_top100)

    # SHAP Beeswarm
    fig1 = plt.figure(figsize=(10, 5))
    shap.plots.beeswarm(shap_values_top100)
    plt.title(f"SHAP Beeswarm - {cobertura}")
    plt.show()
    plt.savefig(os.path.join(output_path, f"shap_beeswarm_{cobertura}.png"))
    plt.close(fig1)

    # SHAP Waterfall individual
    df_cov_reset = df_cov.reset_index(drop=True)
    idx_most_suspicious = df_cov_reset["suspicion_score"].idxmax()
    X_one = X_scaled_df.loc[[idx_most_suspicious]]
    shap_value_one = explainer(X_one)

    fig2 = plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_value_one[0])
    plt.title(f"SHAP Waterfall - {cobertura}")
    plt.show()
    plt.savefig(os.path.join(output_path, f"shap_waterfall_{cobertura}.png"))
    plt.close(fig2)

    # Score histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df_cov, x="suspicion_score", bins=50, kde=True, color="red")
    plt.title(f"Distribution of Suspicion Score - {cobertura}")
    plt.xlabel("Suspicion Score (higher = more atypical)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"hist_score_{cobertura}.png"))
    plt.close()

    # UMAP + Clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_cov, x="UMAP_1", y="UMAP_2", hue="cluster", palette="tab10", s=15)
    plt.title(f"Clusters Detected - {cobertura}")
    plt.savefig(os.path.join(output_path, f"umap_clusters_{cobertura}.png"))
    plt.close()

    resultados.append(df_cov)

# =====================
# 4. Export Full Dataset
# =====================
final_df = pd.concat(resultados).reset_index(drop=True)
final_df.to_csv(os.path.join(output_path, "claims_scores.csv"), index=False)
print("\n✅ File generated correctly: claims_scores.csv")
