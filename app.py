import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Adult Income Classification App")

# ===============================
# Feature Columns (Training Order)
# ===============================

feature_columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race",
    "sex", "capital_gain", "capital_loss", "hours_per_week",
    "native_country"
]

# ===============================
# Model Selection
# ===============================

model_name = st.selectbox(
    "Select Model",
    [
        "Logistic_Regression",
        "Decision_Tree",
        "KNN",
        "Naive_Bayes",
        "Random_Forest",
        "XGBoost"
    ]
)

# ===============================
# Upload Test Data
# ===============================

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if len(data.columns) == 15:
        data.columns = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race",
            "sex", "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income"
        ]

    if "income" in data.columns:
        data = data.drop("income", axis=1)

    # 🔥 CLEAN DATA
    data.replace("?", pd.NA, inplace=True)

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip()

    data.dropna(inplace=True)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    model = joblib.load(f"model/{model_name}.pkl")

    predictions = model.predict(data)

    st.subheader("Predictions")
    st.write(predictions)

# ===============================
# Display Evaluation Metrics
# ===============================

st.subheader("Model Evaluation Metrics (Training Phase)")
results_df = pd.read_csv("model_results.csv", index_col=0)
st.dataframe(results_df)
