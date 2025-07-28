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

### 🧠 Methods

This project applies **unsupervised learning** techniques due to the lack of labeled (fraud vs. non-fraud) claims data. The goal is to detect atypical or suspicious patterns without requiring predefined outcomes.

The following anomaly detection models were implemented:

- **Isolation Forest**: Identifies outliers by randomly partitioning data and measuring how quickly an observation becomes isolated. Points that are isolated in fewer steps are more likely to be anomalies.
- **Local Outlier Factor (LOF)**: Calculates the local density deviation of a given data point with respect to its neighbors. Observations in sparse regions relative to their surroundings are flagged as potential anomalies.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Groups nearby points into clusters based on density and labels sparse observations as noise (i.e., anomalies).

Each of these models generates an **anomaly score** per claim, which is used to rank records by their level of suspicion.

---

#### 🧩 Model Explainability

To enhance model transparency and provide interpretable results, **SHAP (SHapley Additive exPlanations)** was used to explain the outputs of the **Isolation Forest** model.

Although SHAP is primarily designed for supervised models, it can be adapted to unsupervised settings when the model produces a meaningful continuous output—such as the anomaly score in Isolation Forest.

> In this context, SHAP explains **which input features most contributed to a claim being considered suspicious**, helping analysts better understand and trust the model’s reasoning.



---

## 🛠️ How to run locally

1. **Clone this repository:**
   ```bash
   git clone https://github.com/SOFIAMORETTI01/Claim_anomalies.git
   cd Claim_anomalies

2. **Install dependencies:**
   ```bash
   pip install -r script/requirements.txt

3. **Run the scripts:**
   ```bash
   python script/anomaly_detection.py
   streamlit run script/anomaly_detection_streamlit.py
