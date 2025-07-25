import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import hdbscan
from umap import UMAP
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# 1. Load Data
# =====================
df = pd.read_csv("claim_anomalies/data/claims.csv")

# =====================
# 2. Preprocessing
# =====================
features = [
    "insured_amount", "claim_amount", "months_since_policy_start",
    "claim_hour", "previous_claim_count", "customer_seniority_years"
]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================
# 3. Isolation Forest Model
# =====================
iso_model = IsolationForest(contamination=0.015, random_state=42)
df["anomaly_score"] = iso_model.fit_predict(X_scaled)
df["suspicion_score"] = iso_model.decision_function(X_scaled) * -1
df["is_suspicious"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

# =====================
# 4. UMAP
# =====================
reducer = UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_scaled)

# Confirmamos que embedding es array
embedding = np.array(embedding)

df["UMAP_1"] = embedding[:, 0]
df["UMAP_2"] = embedding[:, 1]

# =====================
# 5. HDBSCAN clustering
# =====================
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, prediction_data=True)
df["cluster"] = clusterer.fit_predict(embedding)

# =====================
# 6. UMAP + Clusters Visualization
# =====================
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="UMAP_1", y="UMAP_2", hue="cluster", palette="tab10", s=15)
plt.title("Clusters Detected with HDBSCAN + UMAP")
plt.close()

# =====================
# 7. Suspicion Score Visualization
# =====================
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="suspicion_score", bins=50, kde=True, color="red")
plt.title("Distribution of Suspicion Score")
plt.xlabel("Suspicion Score (higher = more atypical)")
plt.ylabel("Frequency")
plt.grid(True)
plt.close()

# =====================
# 8. SHAP - Explainability
# =====================

# 8.0 Convert scaled features into a DataFrame with column names
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# 8.1 Create SHAP explainer with feature names
explainer = shap.Explainer(iso_model, X_scaled_df)

# 8.2 SHAP global: beeswarm plot for top 100 most suspicious claims
top_100_idx = df.sort_values("suspicion_score", ascending=False).index[:100]
X_top100 = X_scaled_df.iloc[top_100_idx.to_list()]
shap_values_top100 = explainer(X_top100)

print("📊 SHAP Summary Plot (Top 100 most suspicious claims):")
shap.plots.beeswarm(shap_values_top100)
plt.close()

# 8.3 SHAP individual: waterfall plot for the most suspicious claim
idx_most_suspicious = df["suspicion_score"].idxmax()
X_one = X_scaled_df.iloc[[idx_most_suspicious]]
shap_value_one = explainer(X_one)

print("📉 SHAP Waterfall Plot (Most suspicious claim):")
shap.plots.waterfall(shap_value_one[0])
plt.close()

# =====================
# 9. Exportar dataset con scores 
# =====================
df.to_csv("Claim_anomalies/data/claims_scores.csv")
