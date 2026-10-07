# Telecom Customer Churn Prediction

A practical end-to-end machine learning project for predicting telecom customer churn.

This project was built as an AI/ML internship preparation exercise using a public telecom customer churn dataset.

## Business Problem

Telecom companies want to identify customers who are at risk of leaving so that retention teams can take action before churn occurs.

The objective of this project is to build a machine learning system that predicts whether a customer is likely to churn based on account, service, contract, billing, and customer information.

## Dataset

The project uses a public telecom churn dataset containing:

- 7,043 customers
- 21 original columns
- Customer account information
- Telecom services
- Contract information
- Billing information
- Churn status

The target variable is:

`Churn`

Possible values:

- `Yes`
- `No`

## Project Workflow

The project currently includes:

1. Data inspection
2. Data cleaning
3. Exploratory data analysis
4. SQL analysis
5. Feature preprocessing
6. Logistic Regression baseline
7. Decision Tree comparison
8. Random Forest comparison
9. Cross-validation
10. Hyperparameter tuning
11. Threshold analysis
12. Feature importance
13. Permutation importance
14. Train / validation / final test split
15. Final model evaluation
16. Model serialization
17. Reusable prediction function
18. FastAPI prediction API
19. Input validation
20. Automated API tests

## Main EDA Findings

Overall churn rate:

- 26.54%

Churn by contract:

- Month-to-month: 42.71%
- One year: 11.27%
- Two year: 2.83%

Churn by tenure:

- 0–12 months: 47.44%
- 13–24 months: 28.71%
- 25–48 months: 20.39%
- 49–72 months: 9.51%

Customers who churned also had higher monthly charges on average.

## Model Comparison

Models evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Cross-validation showed that Logistic Regression and Random Forest had similar ROC-AUC, while Random Forest achieved substantially higher recall.

Because the assumed business objective prioritizes identifying churners, Random Forest was selected as the current champion model.

## Final Model

Model:

Random Forest Classifier

Selected parameters:

- n_estimators = 100
- max_depth = 8
- min_samples_split = 5
- class_weight = balanced
- random_state = 42

## Final Test Results

Final held-out test results:

- Accuracy: 0.7588
- Precision: 0.5319
- Recall: 0.7722
- F1-score: 0.6299
- ROC-AUC: 0.8390

Confusion matrix:

- True Negatives: 585
- False Positives: 191
- False Negatives: 64
- True Positives: 217

The model identified approximately 77% of actual churners in the final test set.

## Important Predictive Features

Permutation importance identified several important features:

- Contract
- InternetService
- tenure
- TotalCharges
- PaperlessBilling
- OnlineBackup
- StreamingTV
- PaymentMethod

These features are predictive signals and should not be interpreted as proof of causation.

## Project Structure

```text
telecom-churn-internship/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── experiment_log.md
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── api.py
│   ├── clean_data.py
│   ├── compare_models.py
│   ├── create_database.py
│   ├── eda_churn.py
│   ├── final_model_selection.py
│   ├── inspect_data.py
│   ├── predict_customer.py
│   ├── prepare_model_data.py
│   ├── sql_analysis.py
│   └── tune_random_forest.py
│
├── tests/
│   └── test_api.py
│
├── .gitignore
├── README.md
└── requirements.txt