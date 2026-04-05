import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# load artifacts


@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/logreg_tuned_model.pkl")
    preprocessor = joblib.load("artifacts/preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_artifacts()

THRESHOLD = 0.40   # from notebook 06

INTERNET_MAP = {"No internet service": 0, "No": 1, "Yes": 2}
YES_NO_MAP = {"Yes": 1, "No": 0}

# helperr


def build_input_df(inputs: dict) -> pd.DataFrame:
    row = {
        "gender": inputs["gender"],
        "SeniorCitizen": inputs["SeniorCitizen"],
        "Partner": YES_NO_MAP[inputs["Partner"]],
        "Dependents": YES_NO_MAP[inputs["Dependents"]],
        "tenure": inputs["tenure"],
        "PhoneService": YES_NO_MAP[inputs["PhoneService"]],
        "MultipleLines": inputs["MultipleLines"],
        "InternetService": inputs["InternetService"],
        "OnlineSecurity": INTERNET_MAP[inputs["OnlineSecurity"]],
        "OnlineBackup": INTERNET_MAP[inputs["OnlineBackup"]],
        "DeviceProtection": INTERNET_MAP[inputs["DeviceProtection"]],
        "TechSupport": INTERNET_MAP[inputs["TechSupport"]],
        "StreamingTV": INTERNET_MAP[inputs["StreamingTV"]],
        "StreamingMovies": INTERNET_MAP[inputs["StreamingMovies"]],
        "Contract": inputs["Contract"],
        "PaperlessBilling": YES_NO_MAP[inputs["PaperlessBilling"]],
        "PaymentMethod": inputs["PaymentMethod"],
        "MonthlyCharges": inputs["MonthlyCharges"],
        "TotalCharges": inputs["TotalCharges"],
    }
    return pd.DataFrame([row])


with st.sidebar:
    st.title("📡 Churn Predictor")
    st.caption("Enter customer details to predict churn risk")
    st.divider()

    st.subheader("👤 Demographics")
    gender = st.selectbox("Gender",         ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner",        ["Yes", "No"])
    dependents = st.selectbox("Dependents",     ["Yes", "No"])

    st.divider()
    st.subheader("📋 Account")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input(
        "Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)",   0.0, 10000.0,
                                    float(tenure * monthly_charges), step=1.0)

    st.divider()
    st.subheader("📞 Phone Service")
    phone = st.selectbox("Phone Service",    ["Yes", "No"])
    multi = st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])

    st.divider()
    st.subheader("🌐 Internet Service")
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

    internet_opts = ["No internet service", "No", "Yes"] if internet == "No" else [
        "No", "Yes", "No internet service"]

    online_sec = st.selectbox("Online Security",   internet_opts)
    online_bkp = st.selectbox("Online Backup",     internet_opts)
    device_prot = st.selectbox("Device Protection", internet_opts)
    tech_sup = st.selectbox("Tech Support",      internet_opts)
    stream_tv = st.selectbox("Streaming TV",      internet_opts)
    stream_mov = st.selectbox("Streaming Movies",  internet_opts)

    st.divider()
    predict_btn = st.button("🔍 Predict Churn Risk", use_container_width=True)


# results
st.title("Customer Churn Prediction")
st.caption("Logistic Regression · Threshold 0.40 · SHAP explanations")

if not predict_btn:
    st.info(
        "👈 Fill in the customer details in the sidebar and click **Predict Churn Risk**.")
    st.stop()


inputs = {
    "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": multi,
    "InternetService": internet, "OnlineSecurity": online_sec,
    "OnlineBackup": online_bkp, "DeviceProtection": device_prot,
    "TechSupport": tech_sup, "StreamingTV": stream_tv,
    "StreamingMovies": stream_mov, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
}

input_df = build_input_df(inputs)
X_processed = preprocessor.transform(input_df)
churn_prob = model.predict_proba(X_processed)[0, 1]
churn_pred = int(churn_prob >= THRESHOLD)

# risk level
if churn_prob >= 0.70:
    risk_label, risk_color, risk_emoji = "High Risk",   "#E24B4A", "🔴"
elif churn_prob >= 0.40:
    risk_label, risk_color, risk_emoji = "Medium Risk", "#EF9F27", "🟡"
else:
    risk_label, risk_color, risk_emoji = "Low Risk",    "#1D9E75", "🟢"

# results row
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Probability", f"{churn_prob:.1%}")

with col2:
    st.metric("Risk Level", f"{risk_emoji} {risk_label}")

with col3:
    st.metric("Prediction", "⚠️ Will Churn" if churn_pred else "Will Stay")

# progress bar
st.divider()
st.subheader("Churn Probability Meter")
st.progress(float(churn_prob))
st.caption(
    f"Threshold: {THRESHOLD} — predictions above this are classified as churn")

# business recommendation
st.divider()
st.subheader("💼 Business Recommendation")

if churn_pred == 1:
    if churn_prob >= 0.70:
        st.error(
            "**High priority — immediate action needed.**  \n"
            "This customer is very likely to churn. Consider offering a personalised retention "
            "deal: a discount, contract upgrade, or a direct call from the retention team."
        )
    else:
        st.warning(
            "**Medium priority — proactive outreach recommended.**  \n"
            "This customer shows moderate churn signals. A targeted email offer or service "
            "upgrade could reduce their risk significantly."
        )
else:
    st.success(
        "**Low risk — no immediate action needed.**  \n"
        "This customer is likely to stay. Continue regular engagement and monitor their "
        "usage patterns."
    )

# shap explaination
st.divider()
st.subheader("🔍 Why this prediction? (SHAP)")

try:
    explainer = shap.LinearExplainer(
        model, X_processed, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_processed)

    # get feature names from preprocessor
    cat_cols = ["gender", "MultipleLines",
                "InternetService", "Contract", "PaymentMethod"]
    num_cols = [
        "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaperlessBilling",
        "MonthlyCharges", "TotalCharges"
    ]

    try:
        ohe_names = preprocessor.named_transformers_[
            'cat'].get_feature_names_out(cat_cols).tolist()
    except Exception:
        ohe_names = [f"cat_{i}" for i in range(
            X_processed.shape[1] - len(num_cols))]

    feature_names = num_cols + ohe_names

    shap_df = pd.DataFrame({
        "Feature": feature_names[:len(shap_values[0])],
        "SHAP": shap_values[0][:len(feature_names)]
    }).reindex(columns=["Feature", "SHAP"])

    shap_df["abs"] = shap_df["SHAP"].abs()
    shap_df = shap_df.sort_values("abs", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in shap_df["SHAP"]]
    ax.barh(shap_df["Feature"][::-1], shap_df["SHAP"]
            [::-1], color=colors[::-1])
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on churn probability)")
    ax.set_title("Top 10 features driving this prediction")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "Red bars push the prediction toward churn · Green bars push toward staying")

except Exception as e:
    st.info(f"SHAP explanation unavailable: {e}")

st.divider()
st.caption("Model: Logistic Regression · Best params: C=1, penalty=l2, class_weight=balanced · CV Recall: 0.8033")
