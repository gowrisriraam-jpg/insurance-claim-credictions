import os
import pandas as pd
import streamlit as st
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Insurance Claim Prediction",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "insurance_claim_model.pkl"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    st.error(
        "❌ Model file not found! "
        "Please run main.py first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🏥 Insurance Claim Prediction")
st.write(
    "Enter customer details to predict whether "
    "an insurance claim is likely."
)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("📋 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🔮 Prediction",
        "📊 Model Information"
    ]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "🏠 Home":

    st.header("Welcome to Insurance Claim Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🤖 Algorithm",
            "Decision Tree"
        )

    with col2:
        st.metric(
            "🎯 Target",
            "Insurance Claim"
        )

    with col3:
        st.metric(
            "📈 Model",
            "Gini Criterion"
        )

    st.markdown("""
    ### About the Project

    This Machine Learning application predicts whether
    a customer is likely to make an insurance claim.

    **Machine Learning Workflow:**

    1. Load Insurance Dataset
    2. Data Cleaning
    3. Label Encoding
    4. Feature Selection
    5. Train Decision Tree
    6. Predict Insurance Claim
    """)

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif page == "🔮 Prediction":

    st.header("🔮 Insurance Claim Prediction")

    st.write(
        "Enter the customer's information below."
    )

    col1, col2 = st.columns(2)

    # -------------------------------
    # Customer Details
    # -------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            ["male", "female"]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0
        )

    with col2:

        smoker = st.selectbox(
            "Smoker",
            ["yes", "no"]
        )

        region = st.selectbox(
            "Region",
            [
                "southwest",
                "southeast",
                "northwest",
                "northeast"
            ]
        )

        charges = st.number_input(
            "Insurance Charges",
            min_value=0.0,
            max_value=100000.0,
            value=5000.0,
            step=100.0
        )

    st.divider()

    # --------------------------------------------------
    # Prediction Button
    # --------------------------------------------------

    if st.button(
        "🔍 Predict Insurance Claim",
        use_container_width=True
    ):

        # Same encoding used during training
        gender_value = 1 if gender == "male" else 0

        smoker_value = 1 if smoker == "yes" else 0

        region_mapping = {
            "northeast": 0,
            "northwest": 1,
            "southeast": 2,
            "southwest": 3
        }

        region_value = region_mapping[region]

        # Input DataFrame
        input_data = pd.DataFrame({
            "age": [age],
            "gender": [gender_value],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker_value],
            "region": [region_value],
            "charges": [charges]
        })

        # Get features expected by model
        if hasattr(model, "feature_names_in_"):

            input_data = input_data[
                list(model.feature_names_in_)
            ]

        # Prediction
        prediction = model.predict(input_data)[0]

        # Probability
        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                input_data
            )[0]

            confidence = max(probability) * 100

        else:
            confidence = 0

        st.divider()

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ Insurance Claim Predicted"
            )

            st.write(
                f"Prediction Confidence: "
                f"**{confidence:.2f}%**"
            )

        else:

            st.success(
                "✅ No Insurance Claim Predicted"
            )

            st.write(
                f"Prediction Confidence: "
                f"**{confidence:.2f}%**"
            )

# --------------------------------------------------
# MODEL INFORMATION PAGE
# --------------------------------------------------

elif page == "📊 Model Information":

    st.header("📊 Model Information")

    st.subheader("Machine Learning Algorithm")

    st.write(
        "Decision Tree Classifier"
    )

    st.subheader("Parameters")

    st.code("""
criterion = gini
max_depth = 5
random_state = 42
    """)

    st.subheader("Input Features")

    st.write("""
    • Age  
    • Gender  
    • BMI  
    • Children  
    • Smoker  
    • Region  
    • Insurance Charges
    """)

    st.subheader("Output")

    st.write("""
    **0 → No Insurance Claim**

    **1 → Insurance Claim**
    """)

st.divider()

st.caption(
    "Insurance Claim Prediction | "
    "Machine Learning Project"
)
