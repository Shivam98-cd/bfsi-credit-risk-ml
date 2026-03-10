import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import get_full_metrics

# Set Page Config
st.set_page_config(page_title="BFSI Credit Risk", layout="wide")


# --- 1. Load Assets Safely ---
@st.cache_resource
def load_all_assets():
    path = "model"
    try:
        features  = joblib.load(os.path.join(path, "features.pkl"))
        scaler    = joblib.load(os.path.join(path, "scaler.pkl"))
        X_test    = joblib.load(os.path.join(path, "X_test.pkl"))
        y_test    = joblib.load(os.path.join(path, "y_test.pkl"))
        # Load per-column encoders saved by the fixed train_model.py
        encoders  = joblib.load(os.path.join(path, "encoders.pkl"))

        models = {
            "XGBoost":             joblib.load(os.path.join(path, "XGBoost.pkl")),
            "Random Forest":       joblib.load(os.path.join(path, "Random_Forest.pkl")),
            "Decision Tree":       joblib.load(os.path.join(path, "Decision_Tree.pkl")),
            "Logistic Regression": joblib.load(os.path.join(path, "Logistic_Regression.pkl"))
        }
        return models, scaler, features, X_test, y_test, encoders
    except Exception as e:
        st.error(f"⚠️ Error loading model files: {e}")
        st.info("Ensure you have run 'train_model.py' successfully.")
        return None, None, None, None, None, None


models_dict, scaler, feature_names, X_test, y_test, encoders = load_all_assets()

# --- 2. Dashboard Header ---
st.title("🏦 BFSI Credit Risk Dashboard")
st.markdown("---")

if models_dict is None:
    st.stop()

# --- 3. Tabs ---
tab1, tab2 = st.tabs(["🔍 Individual Prediction", "📊 Model Performance Analytics"])


# --- TAB 1: PREDICTION (all 10 features) ---
with tab1:
    st.subheader("Applicant Information")

    # Row 1 — Personal details
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        income = st.number_input("Annual Income (₹)", min_value=200000, max_value=2000000,
                                  value=800000, step=10000)
    with col3:
        employment = st.selectbox("Employment Status",
                                   ["Salaried", "Business", "Freelancer", "Self-Employed"])

    # Row 2 — Credit profile
    col4, col5, col6 = st.columns(3)
    with col4:
        credit_score = st.slider("Credit History Score", 550, 849, 700)
    with col5:
        dti = st.slider("Debt to Income Ratio", 0.1, 0.6, 0.3, step=0.01)
    with col6:
        past_loans = st.selectbox("Number of Past Loans", [0, 1, 2, 3, 4, 5], index=2)

    # Row 3 — Loan details
    col7, col8, col9 = st.columns(3)
    with col7:
        loan_amount = st.number_input("Loan Amount (₹)", min_value=50000, max_value=1000000,
                                       value=300000, step=10000)
    with col8:
        loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 60], index=2)
    with col9:
        interest_rate = st.slider("Interest Rate (%)", 8.0, 18.0, 13.0, step=0.1)

    # Row 4 — Category fields
    col10, col11 = st.columns(2)
    with col10:
        loan_purpose = st.selectbox("Loan Purpose",
                                     ["Home Loan", "Car Loan", "Education",
                                      "Personal Use", "Medical Emergency", "Business Expansion"])
    with col11:
        city = st.selectbox("City",
                             ["Ahmedabad", "Bangalore", "Chennai", "Delhi",
                              "Hyderabad", "Kolkata", "Mumbai", "Pune"])

    selected_model_tab1 = st.selectbox("Select Model for Prediction", list(models_dict.keys()))

    if st.button("Analyze Risk", use_container_width=True):

        # ✅ Encode categorical inputs using saved per-column encoders
        try:
            employment_enc   = encoders['Employment_Status'].transform([employment])[0]
            loan_purpose_enc = encoders['Loan_Purpose'].transform([loan_purpose])[0]
            city_enc         = encoders['City'].transform([city])[0]
        except ValueError as e:
            st.error(f"Encoding error — unseen category: {e}")
            st.stop()

        # ✅ Build input with ALL 10 features in exact training order
        input_data = {
            'Age':                  age,
            'Annual_Income':        income,
            'Employment_Status':    employment_enc,
            'Credit_History_Score': credit_score,
            'Number_of_Past_Loans': past_loans,
            'Debt_to_Income_Ratio': dti,
            'Loan_Amount':          loan_amount,
            'Loan_Term':            loan_term,
            'Interest_Rate':        interest_rate,
            'Loan_Purpose':         loan_purpose_enc,
            'City':                 city_enc
        }

        input_df    = pd.DataFrame([input_data])[feature_names]  # enforce column order
        scaled_data = scaler.transform(input_df)

        current_model = models_dict[selected_model_tab1]
        prob = current_model.predict_proba(scaled_data)[0][1]

        st.markdown("---")
        if prob > 0.5:
            st.error("### ❌ Result: High Risk")
        else:
            st.success("### ✅ Result: Low Risk")

        st.write(f"Default probability: **{prob:.2%}** — Model: **{selected_model_tab1}**")
        st.progress(float(prob), text=f"Risk Score: {prob:.2%}")


# --- TAB 2: PERFORMANCE ANALYTICS ---
with tab2:
    if X_test is not None and y_test is not None:

        selected_model_tab2 = st.selectbox(
            "Select Model to Analyze",
            list(models_dict.keys()),
            key="tab2_model_selector"
        )
        current_model = models_dict[selected_model_tab2]

        st.header(f"📈 Performance Analytics: {selected_model_tab2}")
        st.markdown("---")

        metrics = get_full_metrics(current_model, X_test, y_test)

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy",    f"{metrics['Accuracy']:.2%}")
        m2.metric("ROC AUC",     f"{metrics['ROC_AUC']:.3f}")
        m3.metric("Max KS Stat", f"{metrics['KS_Table']['KS'].max():.2f}")

        st.markdown("---")

        # KS Table + Confusion Matrix
        col_c, col_d = st.columns([1.5, 1])
        with col_c:
            st.subheader(f"📊 KS Statistic Table — {selected_model_tab2}")
            st.dataframe(
                metrics['KS_Table'].style.background_gradient(subset=['KS'], cmap='YlGn'),
                use_container_width=True
            )
        with col_d:
            st.subheader("🎯 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(metrics['Confusion_Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # Dynamic multi-model comparison
        st.subheader("🔄 Multi-Model Comparison")
        comp_rows = []
        for model_name, model_obj in models_dict.items():
            m = get_full_metrics(model_obj, X_test, y_test)
            comp_rows.append({
                "Model":    model_name,
                "Accuracy": f"{m['Accuracy']:.2%}",
                "ROC AUC":  f"{m['ROC_AUC']:.3f}",
                "Max KS":   f"{m['KS_Table']['KS'].max():.2f}"
            })

        comp_df = pd.DataFrame(comp_rows)

        def highlight_selected(row):
            return ['background-color: #d4edda; font-weight: bold'
                    if row['Model'] == selected_model_tab2 else '' for _ in row]

        st.dataframe(
            comp_df.style.apply(highlight_selected, axis=1),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("⚠️ X_test data not found. Please run training script to enable analytics.")