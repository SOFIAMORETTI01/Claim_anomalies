# 🔍 Detecting atypical insurance claims

This project detects atypical insurance claims using Isolation Forest, an unsupervised algorithm that scores anomalies based on behavioral patterns.  

It features interactive filters, applies UMAP for dimensionality reduction and HDBSCAN for clustering. SHAP values explain the key drivers behind each flagged claim.

[![View App Online](https://img.shields.io/badge/🚀%20View%20Online-Streamlit-green?style=for-the-badge)](https://claimanomalies-kjdxxq5bse8b3axfpopagj.streamlit.app/)

---

## ⚙️ Tech Stack

- **Logic & modeling:** Python (Pandas, Scikit-learn, Seaborn), Isolation Forest, HDBSCAN, UMAP
- **Visualization & explainability:** Matplotlib, Streamlit, SHAP
- **Deployment:** Streamlit cloud

---

## 🎯 Objective

The goal is to assign a **suspicion score** to each claim and **flag outliers** that may indicate:
- Fraudulent behavior  
- Data entry errors  
- Unusual patterns not covered by business rules

---

## 🧠 Methods

This project uses **unsupervised learning** due to the lack of labeled data.  
Implemented techniques include:

- **Isolation Forest:** Detects outliers by randomly isolating observations.  
- **Local Outlier Factor:** Compares local density of each observation to its neighbors.  
- **DBSCAN:** Clusters dense regions and identifies noise points.

Each model generates an **anomaly score** used to rank claims by suspicion level.

---

## 🛠️ How to run locally

1. **Clone this repository:**
   ```bash
   git clone https://github.com/SOFIAMORETTI01/Claim_anomalies.git
   cd Claim_anomalies

2. **Install dependencies:**
   ```bash
   pip install -r script/requirements.txt

4. **Run the scripts:**
   ```bash
   python script/anomaly_detection.py
   streamlit run script/anomaly_detection_streamlit.py
