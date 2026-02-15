# =====================================================
# Machine Learning Assignment - 2 (Final Clean Version)
# =====================================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# =====================================================
# Create model directory
# =====================================================

os.makedirs("model", exist_ok=True)


# =====================================================
# Load Dataset
# =====================================================

column_names = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race",
    "sex", "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]

df = pd.read_csv(
    "adult.csv",
    names=column_names,
    sep=",",
    skipinitialspace=True
)

df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Strip whitespace everywhere
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()


# =====================================================
# Split Features and Target
# =====================================================

X = df.drop("income", axis=1)

# Convert target to binary
y = df["income"].apply(lambda x: 1 if x == ">50K" else 0)



# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns


# =====================================================
# Preprocessing Pipeline
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ]
)



# =====================================================
# Define Models
# =====================================================

models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive_Bayes": GaussianNB(),
    "Random_Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}


# =====================================================
# Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =====================================================
# Train Pipelines
# =====================================================

results = {}

for name, model in models.items():

    print(f"Training {name}...")

    # Full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    }


    # Save full pipeline
    joblib.dump(pipeline, f"model/{name}.pkl")


# =====================================================
# Save Results
# =====================================================

results_df = pd.DataFrame(results).T
results_df.to_csv("model_results.csv")

print("\nTraining Complete ✅")
print(results_df)
