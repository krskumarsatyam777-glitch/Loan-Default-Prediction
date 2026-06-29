# Loan Default Prediction

A machine learning project for predicting loan default using multiple classification models and a Streamlit web application. The project includes data preprocessing, exploratory data analysis (EDA), model comparison, hyperparameter tuning, and deployment of the best-performing model.

---

## Live Demo

https://loan-default-prediction-fcbx.onrender.com

---

## Project Overview

Loan default prediction helps financial institutions identify applicants who are more likely to default before approving loans. This project builds and compares multiple machine learning models to classify loan applicants based on their likelihood of default.

The best-performing model is deployed as an interactive Streamlit web application for real-time predictions.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

---

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Missing Value Handling
5. One-Hot Encoding of Categorical Features
6. Train-Test Split
7. Model Training
8. Hyperparameter Tuning using GridSearchCV
9. Model Evaluation
10. Model Selection
11. Streamlit Deployment

---

## Models Evaluated

- Logistic Regression
- Decision Tree
- Tuned Decision Tree
- Random Forest
- Tuned Random Forest

---

## Model Performance

| Model | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall |
|-------|---------------:|--------------:|----------------:|---------------:|-------------:|------------:|
| **Tuned Random Forest** | **100.00%** | **91.22%** | **100.00%** | **82.68%** | **100.00%** | **70.87%** |
| Random Forest | 100.00% | 90.66% | 100.00% | **84.93%** | 100.00% | 64.71% |
| Decision Tree | 99.90% | 87.92% | 100.00% | 72.96% | 99.52% | 62.75% |
| Tuned Decision Tree | 85.83% | 85.18% | 60.98% | 60.22% | 80.41% | **75.91%** |
| Logistic Regression | 84.25% | 83.22% | 72.49% | 69.93% | 33.89% | 28.01% |

---

## Selected Model

The **Tuned Random Forest** achieved the best overall performance and was selected for deployment.

### Performance

- Test Accuracy: **91.22%**
- Test Precision: **82.68%**
- Test Recall: **70.87%**

---

## Deployment

During model development, the dataset was divided into training and testing sets to evaluate the performance of different machine learning models.

For deployment, only the **Tuned Random Forest** model was used. The selected model was **retrained on the complete dataset** before deployment so it could learn from all available data.

The trained model and the metadata required for preprocessing were saved using **Joblib**.

- `loan_model.pkl` – trained machine learning model
- `metadata.pkl` – metadata required for preprocessing user input before prediction

---

## Streamlit Application

The Streamlit application allows users to:

- Enter applicant details through an interactive interface.
- Predict whether a loan applicant is likely to default.
- Receive prediction results instantly.
- Perform predictions using the deployed Tuned Random Forest model.

---

## Project Structure

```
Loan-Default-Prediction/
│
├── app.py
├── train.py
├── README.md
├── requirements.txt
│
├── data/
│   └── loan_dataset.csv
│
├── model/
│   ├── loan_model.pkl
│   └── metadata.pkl
│
├── notebook/
│   └── Loan_Default_Prediction.ipynb
│
└── images/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/krskumarsatyam777-glitch/Loan-Default-Prediction.git
```

Move into the project directory

```bash
cd Loan-Default-Prediction
```

Install the required libraries

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Future Improvements

- Add probability score for loan default risk.
- Implement XGBoost and LightGBM for comparison.
- Add SHAP explainability for model predictions.
- Improve UI with additional visualizations.
- Support batch prediction using CSV upload.

---

