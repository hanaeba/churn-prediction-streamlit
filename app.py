# ==========================================
# 🚨 Churn Prediction System - FINAL PRO VERSION
# ==========================================

# ==========================================
# Streamlit App - Churn Prediction PRO
# ==========================================
# ==========================================
# 🚨 Churn Prediction System - PRO VERSION
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Churn Prediction System",
    layout="centered"
)

# =========================
# CSS (Professional UI)
# =========================
st.markdown("""
<style>
.block-container {max-width: 900px;}
.card {
    background: #ffffff;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.high {color:#d63031;font-weight:bold;}
.medium {color:#e17055;font-weight:bold;}
.low {color:#00b894;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# =========================
# UTILS
# =========================
def risk_segment(p):
    if p >= 0.7:
        return "🔥 HIGH RISK"
    elif p >= 0.4:
        return "⚠️ MEDIUM RISK"
    else:
        return "✅ LOW RISK"

# =========================
# SIDEBAR MENU
# =========================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Upload Data", "Prediction", "Dashboard", "Explainability"]
)

# =========================
# UPLOAD DATA
# =========================
if page == "Upload Data":
    st.title("📤 Upload Telco Churn Dataset")

    file = st.file_uploader("Upload CSV file", type="csv")

    if file:
        df = pd.read_csv(file)

        # Cleaning
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

        df["Churn"] = df["Churn"].astype(str).str.strip()
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        df = df.dropna(subset=["Churn"])

        st.session_state["raw_df"] = df
        st.success("✅ Dataset loaded successfully")
        st.dataframe(df.head())

# =========================
# PREDICTION & TRAINING
# =========================
if page == "Prediction" and "raw_df" in st.session_state:
    st.title("🔮 Churn Prediction")

    df = st.session_state["raw_df"].copy()

    X = df.drop(columns=["customerID", "Churn"])
    X = pd.get_dummies(X, drop_first=True)
    y = df["Churn"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Save
    joblib.dump(model, "churn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    # Predict ALL customers
    df["Churn_Probability"] = model.predict_proba(X_scaled)[:, 1]
    df["Risk_Level"] = df["Churn_Probability"].apply(risk_segment)
    df["Predicted_Churn"] = (df["Churn_Probability"] >= 0.5).astype(int)

    st.session_state["df"] = df
    st.session_state["model"] = model
    st.session_state["X"] = X

    # =========================
    # SINGLE CUSTOMER
    # =========================
    idx = st.selectbox("Select customer index", range(len(df)))
    prob = df.iloc[idx]["Churn_Probability"]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔥 Risk Prediction")
    st.markdown(
        f"<h3>{df.iloc[idx]['Risk_Level']} ({prob:.2f})</h3>",
        unsafe_allow_html=True
    )
    st.progress(int(prob * 100))
    st.caption("Model confidence level")
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # MODEL METRICS
    # =========================
    with st.expander("📈 Model Performance"):
        st.text(classification_report(y_test, y_pred))
        st.write("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

# =========================
# DASHBOARD
# =========================
if page == "Dashboard" and "df" in st.session_state:
    st.title("📊 Business Dashboard")

    df = st.session_state["df"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("High Risk Customers", (df["Risk_Level"] == "🔥 HIGH RISK").sum())
    col3.metric("Avg Churn Probability", round(df["Churn_Probability"].mean(), 2))

    st.subheader("🔥 Top High Risk Customers")
    st.dataframe(
        df[["customerID", "Churn_Probability", "Risk_Level"]]
        .sort_values("Churn_Probability", ascending=False)
        .head(10)
    )

    st.subheader("📊 Risk Distribution")
    st.bar_chart(df["Risk_Level"].value_counts())

    st.download_button(
        "⬇️ Download Predictions",
        df.to_csv(index=False),
        "churn_predictions.csv"
    )

# =========================
# EXPLAINABILITY (SHAP)
# =========================
if page == "Explainability" and "model" in st.session_state:
    st.title("🧠 Explainability (SHAP)")

    model = st.session_state["model"]
    X = st.session_state["X"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.subheader("Global Feature Importance")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    st.pyplot(fig)
