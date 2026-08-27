from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ==============================
# Project Paths
# ==============================

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "dataset" / "student_performance.csv"
MODEL = ROOT / "models" / "student_model.pkl"
VIZ = ROOT / "visualizations"


# ==============================
# Streamlit Page Configuration
# ==============================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ==============================
# Load Dataset
# ==============================

@st.cache_data
def load_data():
    return pd.read_csv(DATA)


# ==============================
# Load Trained ML Model
# ==============================

@st.cache_resource
def load_model():
    return joblib.load(MODEL)


df = load_data()
model = load_model()


# ==============================
# Header
# ==============================

st.title("🎓 Student Performance Prediction")

st.caption(
    "Machine Learning classification project using "
    "Python, Pandas, NumPy, Matplotlib and Scikit-learn."
)


# ==============================
# Tabs
# ==============================

tab1, tab2 = st.tabs(
    ["🔮 Prediction", "📊 Data Analysis"]
)


# ==========================================================
# TAB 1 - PREDICTION
# ==========================================================

with tab1:

    st.subheader("Predict Student Performance")

    st.write(
        "Enter the student's details below to predict "
        "whether the student is likely to Pass or Fail."
    )

    col1, col2, col3 = st.columns(3)

    # Study Hours
    with col1:
        study_hours = st.number_input(
            "Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=0.5
        )

    # Attendance
    with col2:
        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0
        )

    # Previous Score
    with col3:
        previous_score = st.number_input(
            "Previous Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    st.write("")

    # Prediction Button
    if st.button(
        "🔮 Predict Performance",
        type="primary",
        use_container_width=True
    ):

        # Create input dataframe
        input_df = pd.DataFrame(
            [{
                "StudyHours": study_hours,
                "Attendance": attendance,
                "PreviousScore": previous_score
            }]
        )

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Prediction probability
        probabilities = model.predict_proba(input_df)[0]

        probability = probabilities[prediction]

        st.divider()

        # Display result
        if prediction == 1:

            st.success(
                "🎉 Predicted Result: PASS"
            )

        else:

            st.error(
                "⚠️ Predicted Result: FAIL"
            )

        # Confidence
        st.metric(
            "Prediction Confidence",
            f"{probability:.1%}"
        )


# ==========================================================
# TAB 2 - DATA ANALYSIS
# ==========================================================

with tab2:

    st.subheader("📊 Dataset Overview")

    # Dataset statistics
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Students",
            len(df)
        )

    with c2:
        st.metric(
            "Average Study Hours",
            f"{df['StudyHours'].mean():.1f}"
        )

    with c3:
        st.metric(
            "Average Attendance",
            f"{df['Attendance'].mean():.1f}%"
        )

    with c4:
        pass_rate = (
            df["Result"]
            .eq("Pass")
            .mean()
        )

        st.metric(
            "Pass Rate",
            f"{pass_rate:.1%}"
        )


    # Dataset preview
    st.divider()

    st.subheader("Student Dataset")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


    # Visualizations
    st.divider()

    st.subheader("📈 Visualizations")

    left, right = st.columns(2)

    # Left column
    with left:

        st.image(
            str(
                VIZ /
                "study_hours_vs_result.png"
            ),
            caption="Study Hours vs Result",
            use_container_width=True
        )

        st.image(
            str(
                VIZ /
                "score_distribution.png"
            ),
            caption="Previous Score Distribution",
            use_container_width=True
        )


    # Right column
    with right:

        st.image(
            str(
                VIZ /
                "attendance_vs_result.png"
            ),
            caption="Attendance vs Result",
            use_container_width=True
        )

        st.image(
            str(
                VIZ /
                "result_distribution_pie.png"
            ),
            caption="Student Result Distribution",
            use_container_width=True
        )


# ==============================
# Footer
# ==============================

st.divider()

st.caption(
    "Student Performance Prediction | "
    "Machine Learning Personal Project"
)