# Loan Default Prediction

A machine learning project for predicting loan default using multiple classification models and a Streamlit web application. The project includes data preprocessing, exploratory data analysis (EDA), model comparison, hyperparameter tuning, and deployment of the best-performing model.

---

## Live Demo

https://loan-default-prediction-fcbx.onrender.com

---

## Project Overview

Loan default prediction helps financial institutions identify applicants who are more likely to default before approving loans. This project uses the **HMEQ (Home Equity)** dataset, which contains historical loan performance records, to build and compare multiple machine learning models that classify applicants based on their likelihood of default.

The best-performing model is deployed as an interactive Streamlit web application for real-time predictions.

---

## Dataset

The dataset used is **HMEQ (Home Equity)**, containing **5960 rows** and **13 columns**, with the target variable `BAD` (1 = client defaulted on loan, 0 = loan repaid). Since the raw column names are abbreviated, here's what each one means:

| Column | Meaning |
|--------|---------|
| `BAD` | Target variable — 1 = defaulted, 0 = repaid |
| `LOAN` | Amount of loan requested |
| `MORTDUE` | Amount due on existing mortgage |
| `VALUE` | Current value of the property |
| `REASON` | Reason for the loan request — `DebtCon` (debt consolidation) or `HomeImp` (home improvement) |
| `JOB` | Applicant's occupational category |
| `YOJ` | Years at present job |
| `DEROG` | Number of major derogatory credit reports |
| `DELINQ` | Number of delinquent credit lines |
| `CLAGE` | Age of the oldest credit line, in months |
| `NINQ` | Number of recent credit inquiries |
| `CLNO` | Number of existing credit lines |
| `DEBTINC` | Debt-to-income ratio |

The dataset also shows a class imbalance — only about **20%** of applicants defaulted (`BAD = 1`) — which was factored into model training and evaluation (see below).

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

## Key EDA Insights

Detailed univariate and bivariate analysis for all features, with explanations after each step, is available in the [Jupyter notebook](notebook/Loan_Default_Prediction.ipynb). All EDA plots are also available in [`images/eda/`](images/eda/).

---

## Handling Class Imbalance

The dataset is imbalanced, with defaulters (`BAD = 1`) forming a minority of records. This was addressed by **[e.g. using `class_weight='balanced'` in model training / oversampling the minority class]**, since in a loan default use case, failing to catch an actual defaulter (false negative) is typically more costly than a false alarm — making **recall on the default class** an important metric alongside accuracy and precision.

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

### Feature Importance

The most influential features (by Random Forest feature importance) were:

1. `DEBTINC` — 
2. `DELINQ` — 
3. `CLAGE` — 
<img width="1210" height="1072" alt="image" src="https://github.com/user-attachments/assets/8a907636-02c4-4ec3-a21d-d72f441fd926" />


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

App screenshots (high risk and low risk predictions) are available in [`images/ui_demo/`](images/ui_demo/).

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
    ├── eda/
    └── ui_demo/
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

- **Add probability score for loan default risk.** Instead of a binary yes/no prediction, show the model's predicted probability of default (e.g. 72% risk) so users get a sense of confidence, not just a label.
- **Implement XGBoost and LightGBM for comparison.** Test gradient boosting models alongside the current tree-based models to see if they improve recall on the minority (default) class.
- **Add SHAP explainability for model predictions.** Show which features pushed a specific prediction toward "default" or "no default," so users and interviewers can see the model's reasoning, not just its output.
- **Improve UI with additional visualizations.** Add charts (e.g. feature importance, risk distribution) to the Streamlit app so users can explore the data and model behavior, not just get a single prediction.
- **Support batch prediction using CSV upload.** Let users upload a CSV of multiple applicants and get predictions for all of them at once, instead of entering details one at a time.

---
