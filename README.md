# Claim_anomalies

This project detects atypical insurance claims using Isolation Forest, an unsupervised algorithm that scores anomalies based on behavioral patterns.

It features interactive filters (province, coverage, risk zone), applies UMAP for dimensionality reduction and HDBSCAN for clustering. SHAP values explain the key drivers behind each flagged claim.

Logic & modeling: Python (Pandas, Scikit-learn, Seaborn), Isolation Forest, HDBSCAN, UMAP
Visualization & explainability: Matplotlib, Streamlit, SHAP
Deployment: Streamlit cloud
