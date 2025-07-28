import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =====================
# 1. Data
# =====================
st.set_page_config(page_title="Analysis of atypical claims", layout="wide")

st.markdown("""
<div style="background-color:#1f77b4; padding: 15px 20px; border-radius: 8px;">
    <h1 style="color:white;text-align:center;  font-size: 35px; margin:0;">Anomaly detection in claims using Machine Learning techniques</h1>
</div>
""", unsafe_allow_html=True)


# Introduction
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
st.markdown(
    '''
    <div style="background-color:#f8f9fa; padding: 25px; border-radius: 10px;
                border: 1px solid #ccc; box-shadow: 0px 2px 6px rgba(0,0,0,0.05);">
        <h5 style="color:#0a58ca; font-weight:700; margin-top:0; margin-bottom:10px;">
            What is the purpose of this analysis?
        </h5>
        <hr style="border-top: 1px solid #dee2e6; margin-top: 0; margin-bottom: 20px;">
        <p style="color:#333333; font-size:16px;">
            This dashboard allows actuarial and audit teams to explore potentially atypical claims using unsupervised Machine Learning techniques.
        </p>
        <p style="color:#333333; font-size:15px;">
            <strong>Model used:</strong> Isolation Forest applied to selected claim variables.<br>
            <strong>Objetive:</strong> Support the early identification of cases that require review due to behavior outside of expected patterns.
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)



@st.cache_data
def load_data():
    df = pd.read_csv("data/claims_scores.csv")

    return df

df = load_data()

# =====================
# 2. Filters
# =====================
st.markdown("""
    <style>
    [data-baseweb="tag"] {
        background-color: #2c3e50 !important; 
        color: white !important;
        border: 1px solid #1a252f !important;
        border-radius: 0.5rem !important;
        padding: 0.25rem 0.75rem !important;
        font-size: 14px !important;
    }
    [data-baseweb="tag"] svg {
        fill: white !important;
    }
    [data-baseweb="tag"]:hover {
        background-color: #1a252f !important;
    }
    </style>
""", unsafe_allow_html=True)


print(df.columns.tolist())

st.sidebar.header("🎛️ Filters")

# Filtro de cobertura único (modelo por cobertura)
selected_coverage = st.sidebar.selectbox("Select a coverage type to analyze", df["coverage_type"].unique())

# Otros filtros secundarios
Province = st.sidebar.multiselect("Province", options=df["province"].unique(), default=df["province"].unique())
risk_zone = st.sidebar.multiselect("Risk Zone", options=df["risk_zone"].unique(), default=df["risk_zone"].unique())

# Aplicar todos los filtros juntos
filter = (
    (df["coverage_type"] == selected_coverage) &
    (df["province"].isin(Province)) &
    (df["risk_zone"].isin(risk_zone))
)

df_filtered = df[filter]


# =====================
# 3. KPIs
# =====================
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#2c3e50; padding: 10px 15px; border-radius: 5px;">
  <h3 style="color:white; margin:0;">Key risk indicators</h3>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '''
    <div style="background-color:#f8f9fa; padding: 15px; border-radius: 8px;
                border: 1px solid #ccc; margin-top: 10px; margin-bottom: 20px;
                box-shadow: 0px 2px 6px rgba(0,0,0,0.03);">
        <p style="color:#333333; font-size:15px; margin:0;">
            General summary of the volume of processed claims and the proportion of atypical events detected.
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)



col1, col2, col3 = st.columns(3)

# 1. Number of claims
with col1:
    st.markdown(f"""
    <div style="background-color: #f8f9fa; border: 1px solid #ccc; border-radius: 8px; padding: 12px 10px; text-align: center; height: 90px;">
        <div style="font-size: 15px; color: #0a58ca;">Number of claims analyzed</div>
        <div style="font-size: 22px; font-weight: 600; color: #222; margin-top: 4px;">{len(df_filtered):,}</div>
    </div>
    """, unsafe_allow_html=True)

# 2. Atypical cases
with col2:
    st.markdown(f"""
    <div style="background-color: #f8f9fa; border: 1px solid #ccc; border-radius: 8px; padding: 12px 10px; text-align: center; height: 90px;">
        <div style="font-size: 15px; color: #0a58ca;">Atypical cases detected</div>
        <div style="font-size: 22px; font-weight: 600; color: #222; margin-top: 4px;">{df_filtered['is_suspicious'].sum():,}</div>
    </div>
    """, unsafe_allow_html=True)

# 3. Percentage
with col3:
    st.markdown(f"""
    <div style="background-color: #f8f9fa; border: 1px solid #ccc; border-radius: 8px; padding: 12px 10px; text-align: center; height: 90px;">
        <div style="font-size: 15px; color: #0a58ca;">Anomaly percentage</div>
        <div style="font-size: 22px; font-weight: 600; color: #222; margin-top: 4px;">{100 * df_filtered['is_suspicious'].mean():.2f}%</div>
    </div>
    """, unsafe_allow_html=True)



# =====================
# 4. Chart: Anomaly Score
# =====================
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#2c3e50; padding: 10px 15px; border-radius: 5px;">
    <h3 style="color:white; margin:0;">Anomaly score distribution</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

st.markdown(
    '''
    <div style="background-color: #f1f6fb; border-left: 4px solid #4a90e2; padding: 10px; border-radius: 6px;">
        Distribution of score values generated by the Isolation Forest model. Higher values indicate greater atypicality.
    </div>
    ''',
    unsafe_allow_html=True
)

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.histplot(data=np.array(df_filtered["suspicion_score"]), bins=50, kde=True, color="steelblue")
ax1.set_xlabel("Anomaly score — higher values indicate greater atypicality", fontsize=10)
ax1.set_ylabel("Claim frequency", fontsize=10)
st.pyplot(fig1)


# =====================
# 5. Clustering
# =====================
st.markdown("""
<div style="background-color:#2c3e50; padding: 10px 15px; border-radius: 5px;">
    <h3 style="color:white; margin:0;">Cluster visualization (UMAP + HDBSCAN)</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

st.markdown(
    '''
    <div style="background-color: #f1f6fb; border-left: 4px solid #4a90e2; padding: 10px; border-radius: 6px;">
        Visual representation of claims grouped based on behavioral patterns, using dimensionality reduction and unsupervised clustering.
    </div>
    ''',
    unsafe_allow_html=True
)

fig2, ax2 = plt.subplots(figsize=(10, 5))

blue_palette = ["#A6A6A6","#4a90e2", "#a7c7e7"]  

sns.scatterplot(
    data=df_filtered,
    x="UMAP_1",
    y="UMAP_2",
    hue="cluster",
    palette=blue_palette,
    s=20,
    ax=ax2
)

ax2.set_xlabel("UMAP Component 1", fontsize=10)
ax2.set_ylabel("UMAP Component 2", fontsize=10)

st.pyplot(fig2)


# ======================
# 6. Table: Top Atypical Claims
# ======================
st.markdown("""
<div style="background-color:#2c3e50; padding: 10px 15px; border-radius: 5px;">
    <h3 style="color:white; margin:0;">Top 10: Claims with the highest anomaly scores</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

st.markdown(
    '''
    <div style="background-color: #f1f6fb; border-left: 4px solid #4a90e2; padding: 10px; border-radius: 6px;">
        List of the most extreme cases identified by the model, sorted by descending anomaly score.
    </div>
    ''',
    unsafe_allow_html=True
)
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
top_atypical = df_filtered.sort_values("suspicion_score", ascending=False).head(10)
st.dataframe(top_atypical.reset_index(drop=True), use_container_width=True)

# =====================
# 7. Export to csv
# =====================
st.download_button(
    label="📥 Download Top 10 atypical claims (CSV)",
    data=top_atypical.to_csv(index=False).encode("utf-8"),
    file_name="top_atypical_claims.csv",
    mime="text/csv"
)

# =====================
# 10. Explainability of anomaly detection (SHAP)
# =====================

import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Título general
st.markdown("""
<div style="background-color:#2c3e50; padding: 10px 15px; border-radius: 5px;">
    <h3 style="color:white; margin:0;">Explainability of anomaly detection (SHAP)</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Caja explicativa
st.markdown("""
<div style="background-color: #f1f6fb; border-left: 4px solid #4a90e2; padding: 10px; border-radius: 6px;">
    We use SHAP (SHapley Additive exPlanations) to interpret how the model determines whether a claim is atypical.<br><br>
    🔹 The left plot shows which variables are most important across the top 100 suspicious claims.<br>
    🔹 The right one explains the most suspicious individual claim, breaking down the exact variable contributions.
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
# Preprocesamiento
features = [
    "insured_amount", "claim_amount", "months_since_policy_start",
    "claim_hour", "previous_claim_count", "customer_seniority_years"
]
X = df_filtered[features].reset_index(drop=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)


# Modelo y SHAP
iso_model = IsolationForest(contamination=0.015, random_state=42)
iso_model.fit(X_scaled_df)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})
shap.plots.colors.red_blue = plt.get_cmap("Blues")
# =====================
# SHAP sobre top 100 más sospechosos del subconjunto filtrado
# =====================

# 1. Obtener scores del subconjunto filtrado (más bajos = más anómalos)
scores_filtered = iso_model.decision_function(X_scaled_df)

# 2. Índices de top 100 del subconjunto (los más anómalos)
top_100_idx = np.argsort(scores_filtered)[:100]  # los menores scores indican mayor anomalía

# 3. Subset de los 100 casos más sospechosos
X_top100 = X_scaled_df.iloc[top_100_idx]

# 4. Calcular SHAP values sobre los 100 más sospechosos
explainer = shap.Explainer(iso_model, X_top100)
shap_values_top100 = explainer(X_top100)

# 6. Explicación individual del caso más sospechoso del subconjunto filtrado
df_filtered_reset = df_filtered.reset_index(drop=True)
idx_most_suspicious = df_filtered_reset["suspicion_score"].idxmax()

X_one = X_scaled_df.iloc[[idx_most_suspicious]]
shap_value_one = explainer(X_one)

# Columnas y títulos estilizados
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background-color:#2c3e50; width: 100, padding: 6px 10px; border-radius: 5px; text-align:center;">
        <h4 style="color:white; font-size:16px; margin:0;"> Global Feature Importance</h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    fig_beeswarm = plt.figure(figsize=(6, 4))
    shap.plots.beeswarm(shap_values_top100, show=False)
    st.pyplot(fig_beeswarm)
    plt.close(fig_beeswarm)

with col2:
    st.markdown("""
    <div style="background-color:#2c3e50; width: 100,padding: 6px 10px; border-radius: 5px; text-align:center;">
        <h4 style="color:white; font-size:16px; margin:0;"> Individual Explanation</h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    fig_waterfall = plt.figure(figsize=(6, 4))
    shap.plots.waterfall(shap_value_one[0], show=False)
    st.pyplot(fig_waterfall)
    plt.close(fig_waterfall)


# =====================
# 9. Anomaly Frequency by Time of Day
# =====================
st.markdown("""
<div style="background-color:#2c3e50; padding: 10px 15px; border-radius: 5px;">
    <h3 style="color:white; margin:0;">Anomaly frequency by time of day</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

st.markdown(
    '''
    <div style="background-color: #f1f6fb; border-left: 4px solid #4a90e2; padding: 10px; border-radius: 6px;">
        Objective: detect suspicious time-based patterns in claim occurrences.
    </div>
    ''',
    unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(10, 5))  
sns.histplot(data=df_filtered[df_filtered["is_suspicious"] == 1], x="claim_hour", bins=24, ax=ax, color="steelblue")
ax.set_title("Frequency of anomalies by time of claim")
ax.set_xlabel("Hour of the Day")
ax.set_ylabel("Number of Atypical Claims")
st.pyplot(fig)

# =====================
# 9. Time Since Policy Start vs. Anomalies
# =====================
st.markdown("""
<div style="background-color:#2c3e50; padding: 10px 15px; border-radius: 5px;">
    <h3 style="color:white; margin:0;">Do new customers generate more atypical claims</h3>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="background-color: #f1f6fb; border-left: 4px solid #4a90e2; padding: 10px; border-radius: 6px;">
We explore whether atypical claims tend to occur among customers who have had their policy for only a few months.
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)


# Density Plot (KDE)
fig, ax = plt.subplots(figsize=(10, 5))
sns.kdeplot(
    data=df_filtered[df_filtered["is_suspicious"] == 1],
    x="months_since_policy_start",
    fill=True,
    label="Atypical claims",
    color="#4a90e2"
)
sns.kdeplot(
    data=df_filtered[df_filtered["is_suspicious"] == 0],
    x="months_since_policy_start",
    fill=True,
    label="Non-atypical claims",
    color="#a7c7e7"
)
ax.set_title("Distribution of Time Since Policy Start at the Time of Claim")
ax.set_xlabel("Months Since Policy Start")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)
