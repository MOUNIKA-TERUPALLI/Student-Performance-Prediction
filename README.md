# Student Performance Prediction Using Machine Learning

## Overview
A beginner-friendly Machine Learning classification project that predicts whether a student is likely to **Pass or Fail** based on study hours, attendance, and previous score.

## Technologies
Python, Pandas, NumPy, Matplotlib, Scikit-learn, Joblib, Streamlit, Jupyter Notebook.

## Features
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Matplotlib visualizations
- Logistic Regression classification
- Accuracy, confusion matrix and classification report
- Saved ML model using Joblib
- Terminal prediction script
- Streamlit web interface
- Result distribution pie chart

## Project Structure
```text
Student_Performance_Prediction/
├── dataset/student_performance.csv
├── notebooks/student_performance_prediction.ipynb
├── src/train_model.py
├── src/prediction.py
├── models/student_model.pkl
├── visualizations/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Installation
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model
From the project root:
```bash
python src/train_model.py
```

This generates the trained model and Matplotlib visualizations.

## Run Terminal Prediction
```bash
python src/prediction.py
```

## Run Streamlit
```bash
streamlit run app.py
```

The browser opens a simple dashboard where you can enter Study Hours, Attendance and Previous Score and get a PASS/FAIL prediction.

## Jupyter Notebook
```bash
jupyter notebook
```
Open:
`notebooks/student_performance_prediction.ipynb`

Run all cells to see dataset inspection, preprocessing, EDA, visualizations, model training and evaluation.

## Machine Learning Model
Logistic Regression is used because the target is a binary classification problem:
- Fail = 0
- Pass = 1

## Important
The dataset is synthetic and intended for educational demonstration. The project does not claim production-level predictive accuracy.
