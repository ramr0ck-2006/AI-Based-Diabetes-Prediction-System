import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/tuned_gradient_boosting.pkl"

DECISION_THRESHOLD = 0.25

MODEL_NAME = "Tuned Gradient Boosting"

ACCURACY = 80.31
PRECISION = 40.24
RECALL = 59.25
F1_SCORE = 47.93
ROC_AUC = 0.8193


FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* Hero */

.hero {
    padding: 2rem 2.2rem;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #172554 55%,
        #1e3a8a 100%
    );
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.20);
}

.hero-title {
    font-size: 2.45rem;
    font-weight: 750;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #dbeafe;
}

/* Section headers */

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 1.4rem;
    margin-bottom: 0.2rem;
}

/* Metric cards */

.metric-card {
    padding: 1.2rem;
    border-radius: 15px;
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.20);
    text-align: center;
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 750;
}

.metric-label {
    color: #94a3b8;
    font-size: 0.85rem;
}

/* Prediction */

.prediction-card {
    padding: 1.6rem;
    border-radius: 18px;
    border: 1px solid #334155;
    background: #111827;
}

.probability {
    font-size: 3.2rem;
    font-weight: 800;
}

.prediction-label {
    color: #94a3b8;
    font-size: 0.95rem;
}

/* High prediction */

.risk-high {
    padding: 1.5rem;
    border-radius: 16px;
    background: #fee2e2;
    border: 1px solid #fecaca;
    color: #991b1b;
    margin-top: 1rem;
}

/* Low prediction */

.risk-low {
    padding: 1.5rem;
    border-radius: 16px;
    background: #dcfce7;
    border: 1px solid #bbf7d0;
    color: #166534;
    margin-top: 1rem;
}

/* Information */

.info-card {
    padding: 1.25rem;
    border-radius: 15px;
    background: #111827;
    border: 1px solid #334155;
    margin-top: 1rem;
}

/* Disclaimer */

.disclaimer {
    padding: 1.1rem;
    border-radius: 14px;
    background: #1e293b;
    border: 1px solid #475569;
    color: #cbd5e1;
    font-size: 0.88rem;
    margin-top: 1.5rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Prediction System")

    st.markdown("### Model")

    st.info(
        f"""
**{MODEL_NAME}**

Decision threshold:

**{DECISION_THRESHOLD:.2f}**
"""
    )

    st.divider()

    st.markdown("### Dataset")

    st.caption(
        "CDC BRFSS Diabetes Health Indicators"
    )

    st.caption(
        "253,680 original records"
    )

    st.caption(
        "21 predictive features"
    )

    st.divider()

    st.markdown("### System")

    st.success("Model loaded successfully")


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🩺 AI-Based Diabetes Prediction System
</div>

<div class="hero-subtitle">
Machine-learning based estimation of diabetes risk from
health, lifestyle and demographic indicators.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

st.caption(
    "Performance measured on the held-out test set."
)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Accuracy", f"{ACCURACY:.2f}%")

with m2:
    st.metric("Precision", f"{PRECISION:.2f}%")

with m3:
    st.metric("Recall", f"{RECALL:.2f}%")

with m4:
    st.metric("F1 Score", f"{F1_SCORE:.2f}%")

with m5:
    st.metric("ROC-AUC", f"{ROC_AUC:.4f}")


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter the available health and lifestyle indicators."
)


# ============================================================
# HEALTH INDICATORS
# ============================================================

with st.expander("❤️ Health Indicators", expanded=True):

    c1, c2, c3 = st.columns(3)

    with c1:

        high_bp = st.selectbox(
            "High Blood Pressure",
            ["No", "Yes"],
            help="Whether the person has been told they have high blood pressure."
        )

    with c2:

        high_chol = st.selectbox(
            "High Cholesterol",
            ["No", "Yes"]
        )

    with c3:

        chol_check = st.selectbox(
            "Cholesterol Check in Last 5 Years",
            ["No", "Yes"]
        )


    c1, c2, c3 = st.columns(3)

    with c1:

        bmi = st.number_input(
            "BMI",
            min_value=12.0,
            max_value=98.0,
            value=25.0,
            step=0.1
        )

    with c2:

        stroke = st.selectbox(
            "History of Stroke",
            ["No", "Yes"]
        )

    with c3:

        heart_disease = st.selectbox(
            "Heart Disease / Attack",
            ["No", "Yes"]
        )


# ============================================================
# LIFESTYLE
# ============================================================

with st.expander("🏃 Lifestyle & Healthcare", expanded=True):

    c1, c2, c3 = st.columns(3)

    with c1:

        smoker = st.selectbox(
            "Smoker",
            ["No", "Yes"]
        )

    with c2:

        physical_activity = st.selectbox(
            "Physical Activity",
            ["No", "Yes"]
        )

    with c3:

        alcohol = st.selectbox(
            "Heavy Alcohol Consumption",
            ["No", "Yes"]
        )


    c1, c2, c3 = st.columns(3)

    with c1:

        fruits = st.selectbox(
            "Consumes Fruit Regularly",
            ["No", "Yes"]
        )

    with c2:

        veggies = st.selectbox(
            "Consumes Vegetables Regularly",
            ["No", "Yes"]
        )

    with c3:

        healthcare = st.selectbox(
            "Has Healthcare Coverage",
            ["No", "Yes"]
        )


    c1, c2, c3 = st.columns(3)

    with c1:

        no_doc_cost = st.selectbox(
            "Unable to See Doctor Due to Cost",
            ["No", "Yes"]
        )

    with c2:

        diff_walk = st.selectbox(
            "Difficulty Walking / Climbing Stairs",
            ["No", "Yes"]
        )

    with c3:

        gen_health = st.slider(
            "General Health",
            1,
            5,
            3,
            help="1 = Excellent, 5 = Poor"
        )


# ============================================================
# WELL-BEING
# ============================================================

with st.expander("🧠 Physical & Mental Well-being", expanded=True):

    c1, c2, c3 = st.columns(3)

    with c1:

        ment_health = st.slider(
            "Poor Mental Health Days",
            0,
            30,
            0
        )

    with c2:

        phys_health = st.slider(
            "Poor Physical Health Days",
            0,
            30,
            0
        )

    with c3:

        sex = st.selectbox(
            "Sex",
            ["Female", "Male"]
        )


# ============================================================
# DEMOGRAPHICS
# ============================================================

with st.expander("🎓 Demographic & Socioeconomic Information", expanded=True):

    c1, c2, c3 = st.columns(3)

    with c1:

        age = st.slider(
            "Age Category",
            1,
            13,
            7,
            help="BRFSS age category. Higher values represent older age groups."
        )

    with c2:

        education = st.slider(
            "Education Level",
            1,
            6,
            4,
            help="BRFSS education category."
        )

    with c3:

        income = st.slider(
            "Income Level",
            1,
            8,
            5,
            help="BRFSS income category. Higher values represent higher income groups."
        )


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def yes_no(value):

    return 1 if value == "Yes" else 0


input_data = pd.DataFrame({

    "HighBP": [yes_no(high_bp)],

    "HighChol": [yes_no(high_chol)],

    "CholCheck": [yes_no(chol_check)],

    "BMI": [bmi],

    "Smoker": [yes_no(smoker)],

    "Stroke": [yes_no(stroke)],

    "HeartDiseaseorAttack": [
        yes_no(heart_disease)
    ],

    "PhysActivity": [
        yes_no(physical_activity)
    ],

    "Fruits": [
        yes_no(fruits)
    ],

    "Veggies": [
        yes_no(veggies)
    ],

    "HvyAlcoholConsump": [
        yes_no(alcohol)
    ],

    "AnyHealthcare": [
        yes_no(healthcare)
    ],

    "NoDocbcCost": [
        yes_no(no_doc_cost)
    ],

    "GenHlth": [
        gen_health
    ],

    "MentHlth": [
        ment_health
    ],

    "PhysHlth": [
        phys_health
    ],

    "DiffWalk": [
        yes_no(diff_walk)
    ],

    "Sex": [
        1 if sex == "Male" else 0
    ],

    "Age": [
        age
    ],

    "Education": [
        education
    ],

    "Income": [
        income
    ]

})

input_data = input_data[FEATURES]


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍  Predict Diabetes Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    probability = model.predict_proba(
        input_data
    )[0][1]

    probability_percent = probability * 100

    prediction = int(
        probability >= DECISION_THRESHOLD
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Prediction Result</div>',
        unsafe_allow_html=True
    )

    r1, r2 = st.columns(2)

    with r1:

        st.markdown(
            f"""
            <div class="prediction-card">

            <div class="prediction-label">
            Model-Estimated Probability
            </div>

            <div class="probability">
            {probability_percent:.1f}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with r2:

        if prediction == 1:

            result_text = "Higher Predicted Risk"

        else:

            result_text = "Lower Predicted Risk"


        st.markdown(
            f"""
            <div class="prediction-card">

            <div class="prediction-label">
            Model Classification
            </div>

            <div class="probability"
                 style="font-size:2rem; margin-top:0.5rem;">
            {result_text}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.progress(
        min(float(probability), 1.0),
        text=f"Model-estimated probability: {probability_percent:.1f}%"
    )


    # ========================================================
    # RESULT MESSAGE
    # ========================================================

    if probability >= DECISION_THRESHOLD:

        st.markdown(
            f"""
            <div class="risk-high">

            <h3>🔴 Higher Predicted Risk</h3>

            <p>
            The model's estimated probability is
            <b>{probability_percent:.1f}%</b>,
            which is above the selected classification threshold
            of <b>{DECISION_THRESHOLD * 100:.0f}%</b>.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="risk-low">

            <h3>🟢 Lower Predicted Risk</h3>

            <p>
            The model's estimated probability is
            <b>{probability_percent:.1f}%</b>,
            which is below the selected classification threshold
            of <b>{DECISION_THRESHOLD * 100:.0f}%</b>.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">📋 Patient Input Summary</div>',
        unsafe_allow_html=True
    )

    display_data = input_data.T.rename(
        columns={0: "Value"}
    )

    st.dataframe(
        display_data,
        use_container_width=True
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🧠 Model Explanation</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "SHAP shows how individual features influenced this "
        "model prediction. Positive values push the prediction "
        "higher; negative values push it lower."
    )

    try:

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            input_data
        )

        if isinstance(shap_values, list):

            values = np.asarray(
                shap_values[1]
            )[0]

        else:

            shap_array = np.asarray(
                shap_values
            )

            if shap_array.ndim == 3:

                values = shap_array[0, :, 1]

            elif shap_array.ndim == 2:

                values = shap_array[0]

            else:

                values = shap_array.flatten()


        explanation = pd.DataFrame({

            "Feature": FEATURES,

            "Impact": values

        })


        explanation["Absolute Impact"] = (
            explanation["Impact"].abs()
        )


        explanation = explanation.sort_values(
            "Absolute Impact",
            ascending=False
        ).head(10)


        fig, ax = plt.subplots(
            figsize=(10, 5.5)
        )


        ax.barh(
            explanation["Feature"][::-1],
            explanation["Impact"][::-1]
        )


        ax.axvline(
            0,
            linewidth=1
        )


        ax.set_xlabel(
            "SHAP Impact"
        )

        ax.set_title(
            "Top Features Influencing This Prediction"
        )


        plt.tight_layout()


        st.pyplot(
            fig,
            use_container_width=True
        )


        plt.close(fig)


    except Exception as e:

        st.warning(
            "The prediction was generated successfully, "
            "but the SHAP explanation could not be displayed."
        )

        st.caption(
            f"Technical details: {e}"
        )


# ============================================================
# ABOUT MODEL
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">ℹ️ About This System</div>',
    unsafe_allow_html=True
)

a1, a2 = st.columns(2)

with a1:

    st.markdown(
        """
        <div class="info-card">

        <b>Machine Learning Model</b>

        <p>
        The system uses a tuned Gradient Boosting classifier
        trained on CDC BRFSS Diabetes Health Indicators.
        </p>

        <p>
        Hyperparameters were optimized using
        cross-validation with ROC-AUC as the tuning metric.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

with a2:

    st.markdown(
        """
        <div class="info-card">

        <b>Explainable AI</b>

        <p>
        SHAP-based explanations are provided to show which
        patient features contributed most to each prediction.
        </p>

        <p>
        This improves transparency and makes the model's
        output easier to interpret.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

    <b>⚠️ Important Disclaimer</b>

    <br><br>

    This application is an educational machine-learning
    prediction system. It is not a medical diagnostic tool
    and should not be used as a substitute for professional
    medical advice, diagnosis, or treatment.

    <br><br>

    The displayed probability is a model-estimated output
    based on patterns learned from the underlying dataset.
    It does not guarantee an individual's actual medical
    condition.

    </div>
    """,
    unsafe_allow_html=True
)