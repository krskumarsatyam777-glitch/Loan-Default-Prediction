import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/loan_model.pkl")
metadata = joblib.load("model/metadata.pkl")

feature_columns = metadata["feature_columns"]
reason_categories = metadata["reason_categories"]
job_categories = metadata["job_categories"]

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("🏦 Loan Default Risk Prediction")

st.write(
    """
    Enter the applicant's financial and employment details below.
    The trained Random Forest model will estimate whether the applicant
    is likely to default on the loan.
    """
)

# Display names for dropdowns

reason_mapping = {
    "Debt Consolidation": "DebtCon",
    "Home Improvement": "HomeImp"
}

job_mapping = {
    "Manager": "Mgr",
    "Office Worker": "Office",
    "Professional / Executive": "ProfExe",
    "Sales": "Sales",
    "Self Employed": "Self",
    "Other": "Other"
}

st.header("Loan Application Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        loan = st.number_input(
            "Amount of Loan",
            min_value=0.0,
            step=1000.0,
            help="Total loan amount requested."
        )

        value = st.number_input(
            "Property Value",
            min_value=0.0,
            step=1000.0,
            help="Current estimated value of the property."
        )

        selected_reason = st.selectbox(
            "Purpose of Loan",
            list(reason_mapping.keys()),
            help="Select the purpose for which the loan is being requested."
        )

        yoj = st.number_input(
            "Years at Current Job",
            min_value=0,
            max_value=60,
            step=1,
            help="Number of years employed at the current job."
        )

        clage = st.number_input(
            "Age of Oldest Credit Line (Months)",
            min_value=0,
            step=1,
            help="Age of the oldest active credit account in months."
        )

        clno = st.number_input(
            "Total Number of Credit Lines",
            min_value=0,
            step=1,
            help="Total number of active credit accounts."
        )

    with col2:

        mortdue = st.number_input(
            "Outstanding Mortgage",
            min_value=0.0,
            step=1000.0,
            help="Remaining mortgage balance."
        )

        selected_job = st.selectbox(
            "Occupation",
            list(job_mapping.keys()),
            help="Select the applicant's occupation."
        )

        derog = st.number_input(
            "Number of Derogatory Reports",
            min_value=0,
            step=1,
            help="Number of negative credit reports."
        )

        delinq = st.number_input(
            "Number of Delinquent Credit Lines",
            min_value=0,
            step=1,
            help="Number of delinquent credit accounts."
        )

        ninq = st.number_input(
            "Number of Recent Credit Inquiries",
            min_value=0,
            step=1,
            help="Number of recent credit inquiries."
        )

        debtinc = st.number_input(
            "Debt-to-Income Ratio (%)",
            min_value=0.0,
            step=0.1,
            help="Percentage of monthly income used to repay debt."
        )

    predict_button = st.form_submit_button(
        "🔍 Predict Loan Default",
        use_container_width=True
    )

# Convert display names back to the values used during training

reason = reason_mapping[selected_reason]
job = job_mapping[selected_job]

if predict_button:

    input_data = pd.DataFrame({
        "LOAN": [loan],
        "MORTDUE": [mortdue],
        "VALUE": [value],
        "REASON": [reason],
        "JOB": [job],
        "YOJ": [yoj],
        "DEROG": [derog],
        "DELINQ": [delinq],
        "CLAGE": [clage],
        "NINQ": [ninq],
        "CLNO": [clno],
        "DEBTINC": [debtinc]
    })

    # Set categorical data type using categories from training

    input_data["REASON"] = pd.Categorical(
        input_data["REASON"],
        categories=reason_categories
    )

    input_data["JOB"] = pd.Categorical(
        input_data["JOB"],
        categories=job_categories
    )

    # One-Hot Encoding

    input_data = pd.get_dummies(
        input_data,
        columns=["REASON", "JOB"],
        drop_first=True
    )

    # Match training columns

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict

    prediction = model.predict(input_data)[0]
    prediction_probability = model.predict_proba(input_data)[0]

    default_probability = prediction_probability[1]
    confidence = max(prediction_probability)

    # Risk Assessment

    # Predict

    prediction = model.predict(input_data)[0]
    prediction_probability = model.predict_proba(input_data)[0]

    default_probability = prediction_probability[1]

    # Risk Assessment

    st.header("Risk Assessment")

    if default_probability < 0.30:
        st.success("🟢 Low Risk")
        st.write(
            "The applicant appears to have a **low risk** of defaulting on the loan."
        )

    elif default_probability < 0.60:
        st.warning("🟡 Moderate Risk")
        st.write(
            "The applicant appears to have a **moderate risk** of defaulting on the loan."
        )

    else:
        st.error("🔴 High Risk")
        st.write(
            "The applicant appears to have a **high risk** of defaulting on the loan."
        )

    # Default Risk

    st.markdown("---")

    col1, col2 = st.columns([5, 1])

    with col1:
        st.subheader("Estimated Risk of Default")

    with col2:
        st.subheader(f"{default_probability * 100:.2f}%")

    # Make progress bar thicker

    st.markdown("""
    <style>
    div[data-testid="stProgress"] > div > div {
        height: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Center the progress bar

    left, center, right = st.columns([3, 4, 3])

    with center:
        st.progress(default_probability)

    st.caption("0% = Very Low Risk                           100% = Very High Risk")