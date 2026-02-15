import sys
import os
sys.path.append(os.path.abspath("."))
from model.data_loader import load_and_preprocess_data
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

from model.data_loader import load_and_preprocess_data
from model.logistic_regression import train_logistic_regression
from model.decision_tree import train_decision_tree
from model.knn import train_knn
from model.naive_bayes import train_naive_bayes
from model.random_forest import train_random_forest
from model.xgboost_model import train_xgboost

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="ML Assignment 2 – Classification Models",
    layout="wide"
)

st.title("2024dc04255_Machine Learning Assignment – 2")
st.subheader("Classification Models Comparison & Evaluation")

# -----------------------------
# Load and Train Models
# -----------------------------
st.info("Loading dataset and training models...")

X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data()

models = {
    "Logistic Regression": train_logistic_regression(
        X_train_scaled, X_test_scaled, y_train, y_test
    ),
    "Decision Tree": train_decision_tree(
        X_train, X_test, y_train, y_test
    ),
    "kNN": train_knn(
        X_train_scaled, X_test_scaled, y_train, y_test
    ),
    "Naive Bayes": train_naive_bayes(
        X_train_scaled, X_test_scaled, y_train, y_test
    ),
    "Random Forest": train_random_forest(
        X_train, X_test, y_train, y_test
    ),
    "XGBoost": train_xgboost(
        X_train, X_test, y_train, y_test
    )
}

st.success("Models trained successfully!")

# -----------------------------
# Model Selection
# -----------------------------
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a Classification Model",
    list(models.keys())
)

model, metrics = models[selected_model]

# -----------------------------
# Display Metrics
# -----------------------------
st.subheader(f"Evaluation Metrics – {selected_model}")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
col1.metric("AUC", f"{metrics['AUC']:.3f}")

col2.metric("Precision", f"{metrics['Precision']:.3f}")
col2.metric("Recall", f"{metrics['Recall']:.3f}")

col3.metric("F1 Score", f"{metrics['F1']:.3f}")
col3.metric("MCC", f"{metrics['MCC']:.3f}")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")

# Get predictions again for CM
if selected_model in ["Logistic Regression", "kNN", "Naive Bayes"]:
    y_pred = model.predict(X_test_scaled)
else:
    y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

# -----------------------------
# CSV Upload Section (Optional)
# -----------------------------
st.subheader("Upload Test Dataset (Optional)")

uploaded_file = st.file_uploader(
    "Upload CSV file (same feature format, without target column)",
    type=["csv"]
)

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Test Data Preview:")
    st.dataframe(input_df.head())

    if selected_model in ["Logistic Regression", "kNN", "Naive Bayes"]:
        predictions = model.predict(input_df)
    else:
        predictions = model.predict(input_df)

    st.write("Predictions:")
    st.write(predictions)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "**BITS Pilani – M.Tech (AIML / DSE)**  \n"
    "**2024DC04255-Yazhini R**  \n"
    "Machine Learning Assignment – 2"
)
