import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "loan_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open(BASE_DIR / "scaler (3).pkl", "rb") as file:
        scaler = pickle.load(file)
    return model, scaler

# -------------------- Styling --------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.25rem;
        font-weight: 750;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        background: #f1f5f9;
        border: 1px solid #dbe3ec;
        margin-top: 1rem;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 9px;
        height: 3rem;
        font-weight: 650;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown('<div class="main-header">🏦 Loan Approval Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Enter the applicant details below to estimate the loan approval status.</div>',
    unsafe_allow_html=True
)

# -------------------- Model Loading --------------------
try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error("Unable to load the model or scaler. Make sure both pickle files are in the same folder as app.py.")
    st.exception(e)
    st.stop()

# Keep this exact order and spelling: it matches the fitted scaler.
FEATURES = [
    " no_of_dependents",
    " income_annum",
    " loan_amount",
    " loan_term",
    " cibil_score",
    " residential_assets_value",
    " luxury_assets_value"
]

# -------------------- Input Form --------------------
with st.form("loan_prediction_form"):
    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        dependents = st.number_input(
            "Number of Dependents",
            min_value=0, max_value=10, value=2, step=1,
            help="Number of people financially dependent on the applicant."
        )
        income = st.number_input(
            "Annual Income (₹)",
            min_value=0, value=5_000_000, step=100_000,
            help="Applicant's annual income in rupees."
        )
        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=0, value=15_000_000, step=100_000
        )
        loan_term = st.number_input(
            "Loan Term (Years)",
            min_value=1, max_value=40, value=10, step=1
        )

    with col2:
        cibil_score = st.number_input(
            "CIBIL Score",
            min_value=300, max_value=900, value=700, step=1,
            help="Credit score, generally ranging from 300 to 900."
        )
        residential_assets = st.number_input(
            "Residential Assets Value (₹)",
            min_value=0, value=7_000_000, step=100_000
        )
        luxury_assets = st.number_input(
            "Luxury Assets Value (₹)",
            min_value=0, value=15_000_000, step=100_000
        )

    submitted = st.form_submit_button("Predict Loan Status", type="primary")

# -------------------- Prediction --------------------
if submitted:
    input_data = pd.DataFrame([[
        dependents,
        income,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets,
        luxury_assets
    ]], columns=FEATURES)

    try:
        scaled_data = scaler.transform(input_data)
        prediction = int(model.predict(scaled_data)[0])

        st.divider()
        st.subheader("Prediction Result")

        # This assumes the training label encoding is 1 = Approved, 0 = Rejected.
        if prediction == 1:
            st.success("✅ Loan Status: APPROVED")
        else:
            st.error("❌ Loan Status: REJECTED")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_data)[0]
            class_index = list(model.classes_).index(prediction)
            confidence = float(probabilities[class_index]) * 100
            st.metric("Prediction Confidence", f"{confidence:.2f}%")
            st.progress(min(max(confidence / 100, 0.0), 1.0))

        st.caption(
            "This result is a machine-learning estimate, not an actual lending decision. "
            "The output depends on the model and its training data."
        )

    except Exception as e:
        st.error("Prediction failed. Please check the input values and model files.")
        st.exception(e)

with st.sidebar:
    st.title("About the Project")
    st.write(
        "This application uses a trained Logistic Regression model "
        "and StandardScaler to predict loan approval status."
    )
    st.markdown("**Input features**")
    st.write(
        "1. Number of Dependents\n"
        "2. Annual Income\n"
        "3. Loan Amount\n"
        "4. Loan Term\n"
        "5. CIBIL Score\n"
        "6. Residential Assets Value\n"
        "7. Luxury Assets Value"
    )
    st.divider()
    st.caption("For educational and demonstration purposes only.")
