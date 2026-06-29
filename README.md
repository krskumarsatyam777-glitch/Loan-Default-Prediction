# Loan Default Prediction

## Overview

This project predicts whether a loan applicant is likely to default based on applicant and loan-related information using supervised machine learning. Multiple classification algorithms were trained and evaluated, followed by hyperparameter tuning to identify the best-performing model. The final model was deployed as an interactive Streamlit web application.

---

## Objectives

- Perform data preprocessing and exploratory data analysis (EDA).
- Train and compare multiple machine learning models.
- Optimize models using GridSearchCV.
- Evaluate models using multiple performance metrics.
- Deploy the best-performing model using Streamlit.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Pickle

---

## Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Train-Test Split
6. Model Training
7. Hyperparameter Tuning
8. Model Evaluation
9. Model Comparison
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

## Model Selection

The **Tuned Random Forest** achieved the best overall performance on the test dataset and was selected as the final model for deployment.

**Final Model Performance**

- Test Accuracy: **91.22%**
- Test Precision: **82.68%**
- Test Recall: **70.87%**

---

## Streamlit Deployment

The Streamlit application uses **only the final Tuned Random Forest model**.

During model development in the notebook, the dataset was divided into training and testing sets to evaluate and compare the performance of different machine learning algorithms.

For deployment, the selected Tuned Random Forest model was **retrained on the complete dataset** before being saved as a `model.pkl` file. Training on the full dataset allows the deployed model to learn from all available observations, maximizing the information available for making predictions on new loan applications.

---

## Features

The Streamlit application allows users to:

- Enter applicant information through an interactive interface.
- Predict whether a loan is likely to default.
- Receive instant prediction results.
- Use the trained machine learning model without requiring knowledge of the underlying implementation.

---

## Project Structure

```
Loan_Default_Prediction/
│
├── app.py
├── model.pkl
├── loan_default_prediction.ipynb
├── loan_data.csv
├── requirements.txt
├── README.md
└── assets/
    ├── streamlit_app.png
    └── eda_visualizations.png
```

---

## Installation

Clone the repository

```bash
git clone <repository_url>
```

Move to the project directory

```bash
cd Loan_Default_Prediction
```

Install the required libraries

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## Future Improvements

- Incorporate Gradient Boosting models such as XGBoost or LightGBM.
- Add probability scores for default risk.
- Integrate SHAP for model interpretability.


---
