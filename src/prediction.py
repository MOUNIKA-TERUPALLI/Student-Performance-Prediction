
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "student_model.pkl"

def get_number(prompt, low, high):
    while True:
        try:
            value = float(input(prompt))
            if low <= value <= high:
                return value
            print(f"Please enter a value between {low} and {high}.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    if not MODEL.exists():
        print("Model not found. Run: python src/train_model.py")
        return

    model = joblib.load(MODEL)

    print("=" * 50)
    print(" Student Performance Prediction")
    print("=" * 50)

    study = get_number("Enter study hours (0-24): ", 0, 24)
    attendance = get_number("Enter attendance percentage (0-100): ", 0, 100)
    previous = get_number("Enter previous score (0-100): ", 0, 100)

    X = pd.DataFrame([{
        "StudyHours": study,
        "Attendance": attendance,
        "PreviousScore": previous
    }])

    pred = model.predict(X)[0]
    label = "PASS" if pred == 1 else "FAIL"

    print("-" * 50)
    print(f"Predicted Result: {label}")
    print("-" * 50)

if __name__ == "__main__":
    main()
