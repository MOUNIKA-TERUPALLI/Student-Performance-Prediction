from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PROJECT PATHS
# ============================================================

# train_model.py is inside:
# Student_Performance_Prediction_Streamlit/src/
#
# parent.parent gives:
# Student_Performance_Prediction_Streamlit/

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "dataset" / "student_performance.csv"
MODEL_DIR = ROOT / "models"
VIZ_DIR = ROOT / "visualizations"

MODEL_PATH = MODEL_DIR / "student_model.pkl"


# Create required folders
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK PATH
# ============================================================

print("=" * 60)
print("STUDENT PERFORMANCE PREDICTION")
print("=" * 60)

print("\nProject folder:")
print(ROOT)

print("\nDataset path:")
print(DATA_PATH)


if not DATA_PATH.exists():
    print("\nERROR: Dataset not found!")
    print("Expected file:")
    print(DATA_PATH)
    raise FileNotFoundError(
        f"\nDataset not found at:\n{DATA_PATH}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "StudyHours",
    "Attendance",
    "PreviousScore",
    "Result"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )


# ============================================================
# DATA INFORMATION
# ============================================================

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates().copy()


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

df = df.dropna().copy()


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "StudyHours",
    "Attendance",
    "PreviousScore"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.dropna().copy()


# ============================================================
# CLEAN RESULT COLUMN
# ============================================================

df["Result"] = (
    df["Result"]
    .astype(str)
    .str.strip()
    .str.title()
)


# Pass = 1
# Fail = 0

df["Result"] = df["Result"].map({
    "Fail": 0,
    "Pass": 1
})


# Remove invalid result values
df = df.dropna(
    subset=["Result"]
).copy()

df["Result"] = df["Result"].astype(int)


# ============================================================
# CHECK TARGET
# ============================================================

if df["Result"].nunique() < 2:

    raise ValueError(
        "Dataset must contain both Pass and Fail records."
    )


print("\nResult distribution:")

print(
    df["Result"]
    .map({
        0: "Fail",
        1: "Pass"
    })
    .value_counts()
)


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nStatistical Summary:")

print(
    df[numeric_columns].describe()
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[
    [
        "StudyHours",
        "Attendance",
        "PreviousScore"
    ]
]

y = df["Result"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# ============================================================
# CREATE MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n")
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy:.2%}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fail",
            "Pass"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved successfully!")

print(MODEL_PATH)


# ============================================================
# VISUALIZATION 1
# STUDY HOURS VS PREVIOUS SCORE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

fail_data = df[
    df["Result"] == 0
]

pass_data = df[
    df["Result"] == 1
]


plt.scatter(
    fail_data["StudyHours"],
    fail_data["PreviousScore"],
    label="Fail",
    alpha=0.7
)


plt.scatter(
    pass_data["StudyHours"],
    pass_data["PreviousScore"],
    label="Pass",
    alpha=0.7
)


plt.xlabel("Study Hours")
plt.ylabel("Previous Score")

plt.title(
    "Study Hours vs Previous Score by Result"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    VIZ_DIR / "study_hours_vs_result.png",
    dpi=160
)

plt.close()


# ============================================================
# VISUALIZATION 2
# ATTENDANCE VS PREVIOUS SCORE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.scatter(
    fail_data["Attendance"],
    fail_data["PreviousScore"],
    label="Fail",
    alpha=0.7
)


plt.scatter(
    pass_data["Attendance"],
    pass_data["PreviousScore"],
    label="Pass",
    alpha=0.7
)


plt.xlabel("Attendance (%)")
plt.ylabel("Previous Score")

plt.title(
    "Attendance vs Previous Score by Result"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    VIZ_DIR / "attendance_vs_result.png",
    dpi=160
)

plt.close()


# ============================================================
# VISUALIZATION 3
# PREVIOUS SCORE DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    df["PreviousScore"],
    bins=12
)

plt.xlabel("Previous Score")
plt.ylabel("Number of Students")

plt.title(
    "Previous Score Distribution"
)

plt.tight_layout()

plt.savefig(
    VIZ_DIR / "score_distribution.png",
    dpi=160
)

plt.close()


# ============================================================
# VISUALIZATION 4
# RESULT DISTRIBUTION PIE CHART
# ============================================================

result_counts = (
    df["Result"]
    .map({
        0: "Fail",
        1: "Pass"
    })
    .value_counts()
)


plt.figure(
    figsize=(6, 6)
)


plt.pie(
    result_counts.values,
    labels=result_counts.index.tolist(),
    autopct="%1.1f%%",
    startangle=90
)


plt.title(
    "Student Result Distribution"
)

plt.tight_layout()

plt.savefig(
    VIZ_DIR / "result_distribution_pie.png",
    dpi=160
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("PROJECT SETUP COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated model:")
print(
    "models/student_model.pkl"
)

print("\nGenerated visualizations:")

print(
    "visualizations/study_hours_vs_result.png"
)

print(
    "visualizations/attendance_vs_result.png"
)

print(
    "visualizations/score_distribution.png"
)

print(
    "visualizations/result_distribution_pie.png"
)

print("\nNext command:")
print(
    "python -m streamlit run app.py"
)

print("\nDone!")