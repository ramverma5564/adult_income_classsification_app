# Machine Learning Assignment -- 2

## Income Classification Using Multiple ML Models

------------------------------------------------------------------------

## a. Problem Statement

The objective of this project is to develop and compare multiple machine
learning classification models to predict whether an individual's annual
income exceeds \$50,000 based on demographic and employment-related
attributes.

This project demonstrates an end-to-end machine learning workflow
including:

-   Data preprocessing\
-   Model training\
-   Model evaluation using multiple performance metrics\
-   Comparative analysis of classification models\
-   Development of an interactive Streamlit web application\
-   Deployment on Streamlit Community Cloud

------------------------------------------------------------------------

## b. Dataset Description

**Dataset Used:** Adult Income Dataset (UCI Machine Learning Repository)

The dataset is used to predict whether a person earns more than \$50K
per year.

### Dataset Details:

-   Total Instances: 48,842\
-   Total Features: 14\
-   Target Variable: Income (\>50K or \<=50K)\
-   Problem Type: Binary Classification

### Important Features:

-   Age\
-   Workclass\
-   Education\
-   Marital Status\
-   Occupation\
-   Relationship\
-   Race\
-   Sex\
-   Capital Gain\
-   Capital Loss\
-   Hours per Week\
-   Native Country

### Data Preprocessing Steps:

-   Removed missing values ("?")\
-   Trimmed whitespace from categorical features\
-   Applied One-Hot Encoding for categorical variables\
-   Applied Standard Scaling for numerical features\
-   Used Pipeline and ColumnTransformer for robust preprocessing\
-   Performed Stratified Train-Test Split (80:20)

------------------------------------------------------------------------

## c. Models Used and Evaluation Metrics

The following six classification models were implemented on the same
dataset:

1.  Logistic Regression\
2.  Decision Tree Classifier\
3.  K-Nearest Neighbor (KNN)\
4.  Naive Bayes (GaussianNB)\
5.  Random Forest (Ensemble Model)\
6.  XGBoost (Ensemble Model)

For each model, the following evaluation metrics were calculated:

-   Accuracy\
-   AUC Score\
-   Precision\
-   Recall\
-   F1 Score\
-   Matthews Correlation Coefficient (MCC)

------------------------------------------------------------------------

## Model Comparison Table

  ------------------------------------------------------------------------------------
  ML Model Name         Accuracy   AUC      Precision    Recall   F1 Score    MCC
  --------------------- ---------- -------- ------------ -------- ----------- --------
  Logistic_Regression   0.8475     0.9022   0.7354       0.6052   0.6640      0.5711

  Decision_Tree         0.8087     0.7478   0.6134       0.6265   0.6199      0.4922

  KNN                   0.8270     0.8595   0.6664       0.6105   0.6372      0.5248

  Naive_Bayes           0.5826     0.8018   0.3678       0.9414   0.5290      0.3643

  Random_Forest         0.8488     0.9007   0.7287       0.6258   0.6734      0.5786

  XGBoost               0.8633     0.9227   0.7667       0.6478   0.7023      0.6180
  ------------------------------------------------------------------------------------

------------------------------------------------------------------------

## Observations on Model Performance

  -----------------------------------------------------------------------
  ML Model                          Observation
  --------------------------------- -------------------------------------
  Logistic Regression               Provides strong baseline performance
                                    with good generalization capability.

  Decision Tree                     Slightly lower AUC; prone to
                                    overfitting compared to ensemble
                                    methods.

  KNN                               Moderate performance; sensitive to
                                    feature scaling.

  Naive Bayes                       High recall but lower precision and
                                    accuracy due to independence
                                    assumption.

  Random Forest                     Strong overall performance due to
                                    ensemble averaging.

  XGBoost                           Best performing model with highest
                                    Accuracy, AUC, and MCC.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Streamlit Application Features

The deployed Streamlit application includes:

-   CSV Dataset Upload (Test Data)\
-   Model Selection Dropdown\
-   Display of Model Evaluation Metrics\
-   Prediction Output\
-   Robust preprocessing using saved pipeline

------------------------------------------------------------------------

## Project Repository Structure

    project-folder/
    │-- app.py
    │-- requirements.txt
    │-- README.md
    │-- model/
    │   ├── Logistic_Regression.pkl
    │   ├── Decision_Tree.pkl
    │   ├── KNN.pkl
    │   ├── Naive_Bayes.pkl
    │   ├── Random_Forest.pkl
    │   ├── XGBoost.pkl
    │-- model_results.csv
    │-- adult.csv

------------------------------------------------------------------------

## Deployment

The application has been deployed using Streamlit Community Cloud.

-   GitHub Repository Link: (Add your repository link here)
-   Live Streamlit App Link: (Add your deployed app link here)

------------------------------------------------------------------------

## Conclusion

This project demonstrates the comparative analysis of six different
classification models on a real-world dataset. Ensemble models such as
Random Forest and XGBoost outperform individual classifiers in most
evaluation metrics.

The implementation successfully integrates machine learning modeling
with interactive web deployment, showcasing a complete end-to-end
machine learning pipeline.
