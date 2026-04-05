# Customer Churn Prediction System

**Live App:** https://customer-churn-prediction.streamlit.app

An end-to-end machine learning system for predicting customer churn, combining model development, evaluation, threshold optimization, and explainability. The project is deployed as an interactive web application using Streamlit.

---

## Overview

Customer churn prediction enables businesses to identify users who are likely to leave and take preventive action. This project builds a complete pipeline starting from raw data and ending with a deployed application that provides both predictions and explanations.

---

## Features

* Predicts churn probability for individual customers
* Uses a decision threshold tuned for business objectives
* Provides model explanations using SHAP
* Interactive web interface for real-time predictions
* Outputs actionable insights for retention strategies

---

## Machine Learning Pipeline

### Data Processing

* Missing value handling
* Binary and one-hot encoding
* Feature scaling using StandardScaler
* Structured preprocessing using ColumnTransformer

### Models

* Logistic Regression (baseline and tuned)
* Random Forest
* XGBoost

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Key Design Choice

The model prioritizes recall for the churn class to ensure high-risk customers are identified, even at the cost of additional false positives. A threshold of 0.40 was selected based on this trade-off.

---

## Model Performance

* ROC-AUC: ~0.83
* Recall (churn class): ~0.80
* F1 Score: ~0.60

The model performs well in identifying potential churners while maintaining reasonable overall performance.

---

## Explainability

SHAP is used to interpret model predictions at both global and local levels.

Key factors influencing churn:

* Tenure
* Monthly charges
* Contract type

This allows the model’s behavior to be understood and validated against domain intuition.

---

## Web Application

The application is built using Streamlit and allows users to:

* Input customer details
* View predicted churn probability
* Classify customers based on risk
* Understand predictions through SHAP explanations
* Access simple business recommendations

---

## Project Structure

```id="4fxp5c"
customer-churn/
│
├── artifacts/          # Saved models and preprocessing pipeline
├── data/               # Raw and processed datasets
├── notebooks/          # EDA, modeling, tuning, explainability
├── app.py              # Streamlit application
├── requirements.txt    # Dependencies
└── README.md
```

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit

---

## Running Locally

```bash id="qwy36r"
git clone <[repo-url](https://github.com/Sindhu-github1106/customer-churn)>
cd customer-churn
pip install -r requirements.txt
streamlit run app.py
```

---

## Summary

This project demonstrates a complete machine learning workflow, including data preprocessing, model development, evaluation, explainability, and deployment in a user-facing application.
