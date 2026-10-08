# Telecom Customer Churn Prediction

An end-to-end machine learning project for predicting telecom customer churn using customer profile, service, contract, and billing information.

This project was built as a practical AI/ML internship preparation exercise and covers data validation, SQL analysis, preprocessing, model comparison, business-oriented threshold selection, uncertainty analysis, API deployment, testing, documentation, and Git/GitHub workflow.

## Business Problem

Telecom customers may leave a service or move to another provider. The goal is to identify customers at higher churn risk so a retention team can prioritize intervention.

This is a supervised binary classification problem:

- `1` = Churn
- `0` = No Churn

The model is intended to support human decision-making.

## Dataset

This project uses the IBM Telco Customer Churn sample dataset.

Primary source used for this project:

- Dataset name: Telco Customer Churn
- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Original IBM sample repository: IBM `telco-customer-churn-on-icp4d`
- Kaggle mirror: `blastchar/telco-customer-churn`

The dataset contains 7,043 customers and 21 original columns.

The raw CSV is intentionally not committed to this repository. Download the dataset from the original IBM source or the Kaggle mirror and place it at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### Dataset Licence / Usage Note

The commonly used Kaggle listing for this dataset states:

`Data files © Original Authors`

Because the dataset does not present a clearly stated permissive licence on that listing, this repository does not redistribute the raw CSV.

Users should obtain the dataset from the original source or a trusted mirror and follow the source's usage terms.

## Main Data Quality Finding

`TotalCharges` was loaded as a string because 11 rows contained blank strings.

- `df.isna()` reported 0 missing values
- 11 `TotalCharges` entries were blank strings
- all 11 affected customers had `tenure = 0`

For this project, those values were replaced with `0`.

This is a dataset-specific assumption and should be confirmed with a real data owner in production.

## EDA Highlights

Overall churn rate:

```text
26.54%
```

Contract churn:

| Contract | Churn Rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

Tenure churn:

| Tenure | Churn Rate |
|---|---:|
| 0–12 months | 47.44% |
| 13–24 months | 28.71% |
| 25–48 months | 20.39% |
| 49–72 months | 9.51% |

Average monthly charges:

```text
No churn: 61.27
Churn:    74.44
```

These are associations, not causal claims.

## SQL Analysis

SQLite was used to practice realistic data extraction and business analysis.

Analyses included:

- churn by contract
- churn by payment method
- churn by tenure group
- high-value churned month-to-month customers

Electronic-check churn rate:

```text
45.29%
```

## Preprocessing

Numeric features:

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
```

Categorical features are one-hot encoded.

The project uses:

- `StandardScaler`
- `OneHotEncoder(handle_unknown="ignore")`
- `ColumnTransformer`
- `Pipeline`

Using a single pipeline keeps training and inference consistent and reduces leakage risk.

## Models Compared

- Logistic Regression
- Decision Tree
- Random Forest

The project originally favored Random Forest because of its high recall.

A later review showed that this conclusion was too strong, so the model-selection process was improved.

## Revised Evaluation Design

The revised workflow uses:

```text
Training:   4930
Validation: 1056
Final test: 1057
```

The validation set is used for:

- threshold selection
- model comparison
- business-cost comparison

The final test split is evaluated only after validation-based selection.

Important limitation: earlier exploratory experiments used alternative splits from the same dataset. Therefore, the final test should be described as a held-out test split under the revised workflow, not as an externally untouched evaluation.

## Revised Random Forest Tuning

Random Forest was re-tuned using only the revised training split.

Best parameters:

```text
max_depth=8
min_samples_split=2
n_estimators=100
```

Best training CV F1:

```text
0.6331
```

## Business-Oriented Model Selection

Primary simulated cost assumption:

```text
False positive cost = 1 unit
False negative cost = 5 units
```

These are simulated values used for decision-analysis experiments, not measured telecom business costs.

The 1:5 ratio is an illustrative assumption chosen to represent a situation where failing to identify a real churner is considered more costly than unnecessarily contacting a customer who would not churn.

The project does not claim that 1:5 is the true economic ratio for a telecom operator. A real deployment would require the business to estimate costs such as campaign-contact expense, retention-offer expense, expected customer lifetime value, probability of successful retention, and the loss associated with an undetected churner.

### Validation Results

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | Flagged Rate | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.17 | 0.4491 | 0.9143 | 0.6024 | 0.8454 | 53.98% | 434 |
| Random Forest | 0.36 | 0.4490 | 0.8964 | 0.5983 | 0.8433 | 52.94% | 453 |

Under this 1:5 scenario, Logistic Regression was selected because it produced the lower validation cost.

This does **not** mean Logistic Regression is universally better.

## Selected Model

```text
Logistic Regression
```

Decision threshold:

```text
0.17
```

The saved model bundle contains:

- trained preprocessing/model pipeline
- decision threshold
- model name
- false-positive cost assumption
- false-negative cost assumption

## Final Held-Out Evaluation

Logistic Regression at threshold 0.17:

```text
Accuracy:  0.6868
Precision: 0.4547
Recall:    0.8932
F1:        0.6026
ROC-AUC:   0.8448
```

Flagged customers:

```text
552 / 1057 = 52.22%
```

Confusion matrix:

```text
TN 475
FP 301
FN 30
TP 251
```

Simulated business cost:

```text
451 units
```

## Logistic Regression vs Random Forest: Paired Bootstrap Check

The two candidate models were also compared using a paired bootstrap on the same final test customers.

For each bootstrap sample, the same resampled customer indices were used for both models. This directly estimates uncertainty in the business-cost difference between Logistic Regression and Random Forest.

Under the simulated cost assumption:

- false positive cost = 1 unit
- false negative cost = 5 units

Observed test costs were:

- Logistic Regression: 451 units
- Random Forest: 478 units
- observed difference (LR - RF): -27 units

Across 1,000 paired bootstrap samples:

- mean LR - RF cost difference: -26.17 units
- median difference: -26 units
- 95% bootstrap interval: [-66.03, 14.00]
- Logistic Regression had lower cost in 89.5% of bootstrap samples

Because the 95% interval includes zero, the evidence is not strong enough to conclude that Logistic Regression consistently has lower business cost than Random Forest.

Therefore, the two models should be treated as broadly competitive under this simulated scenario.

Logistic Regression remains the deployment model because:

- its observed cost is slightly lower,
- its predictive performance is competitive,
- it is simpler,
- and it is easier to interpret and explain.

The model choice should therefore not be described as a decisive performance win over Random Forest.

## Business-Rule Baseline Comparison

To check whether the machine-learning model adds value beyond a simple operational rule, the final held-out test split was also evaluated using several baselines under the same simulated cost assumption:

- false positive cost = 1 unit
- false negative cost = 5 units

| Strategy | Precision | Recall | F1 | Flagged Rate | Cost |
|---|---:|---:|---:|---:|---:|
| Predict nobody churns | 0.0000 | 0.0000 | 0.0000 | 0.00% | 1405 |
| Predict everybody churns | 0.2658 | 1.0000 | 0.4200 | 100.00% | 776 |
| Flag all month-to-month customers | 0.4248 | 0.8648 | 0.5698 | 54.12% | 519 |
| Logistic Regression | 0.4547 | 0.8932 | 0.6026 | 52.22% | 451 |

The month-to-month rule is a strong baseline.

Compared with that rule, Logistic Regression:

- reduced simulated cost from 519 to 451 units,
- reduced cost by approximately 13.1%,
- identified 8 additional churners,
- produced 28 fewer false positives,
- and flagged a slightly smaller share of customers.

This means the ML model adds measurable value, but the improvement over a simple business rule is moderate rather than dramatic.

The comparison also highlights why simple baselines should be included before claiming that a machine-learning model provides meaningful business improvement.

## Bootstrap Confidence Intervals

95% bootstrap confidence intervals for Logistic Regression:

```text
Precision: 0.4141–0.4991
Recall:    0.8561–0.9278
F1:        0.5617–0.6423
ROC-AUC:   0.8199–0.8712
```

Random Forest intervals overlap substantially, so the project does not claim clear statistical superiority.

## Cost Sensitivity

This analysis is performed on the validation set. For each cost scenario, the threshold is selected on that same validation set to minimize the simulated cost.

Therefore, these values should be interpreted as exploratory sensitivity results showing how model and threshold choices change under different assumptions. They are optimistic for performance estimation and should not be treated as independent final-test results.

The purpose of this section is not to identify a universally correct cost ratio, but to demonstrate that the preferred model and threshold depend on business assumptions.

False-positive cost was fixed at 1 unit.

| Missed Churn Cost | Selected Model | Threshold | Recall | F1 | Flagged Rate | Cost |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Random Forest | 0.59 | 0.7429 | 0.6390 | 35.13% | 307 |
| 3 | Random Forest | 0.52 | 0.8071 | 0.6384 | 40.53% | 364 |
| 5 | Logistic Regression | 0.17 | 0.9143 | 0.6024 | 53.98% | 434 |
| 10 | Random Forest | 0.24 | 0.9571 | 0.5624 | 63.73% | 525 |
| 20 | Logistic Regression | 0.06 | 0.9786 | 0.5274 | 71.88% | 605 |

Main lesson:

> There is no universally best model or threshold. The preferred operating point depends on business costs and operational capacity.

## Explainability

Earlier permutation importance highlighted:

- Contract
- InternetService
- tenure
- TotalCharges
- PaperlessBilling
- OnlineBackup
- StreamingTV
- PaymentMethod

These are predictive relationships, not causal claims.

## API

Run:

```powershell
uvicorn src.api:app --reload --port 8001
```

Swagger:

```text
http://127.0.0.1:8001/docs
```

Health:

```text
http://127.0.0.1:8001/health
```

## API Validation

The API validates:

- negative tenure
- negative charges
- valid `SeniorCitizen`
- valid contract values
- valid payment methods
- valid service categories
- phone-service consistency
- internet-service consistency
- tenure 0 / TotalCharges 0 dataset convention

Invalid categories such as:

```json
{"Contract": "banana"}
```

are rejected with HTTP 422.

## Automated Tests

Run:

```powershell
python -m pytest -q
```

Current result:

```text
6 passed
```

Tests cover:

- health endpoint
- valid prediction
- negative tenure
- invalid contract
- inconsistent phone service
- inconsistent internet service

## Reproducing from a Fresh Clone

### 1. Clone

```bash
git clone https://github.com/IT24100052/Telecom-churn-internship.git
cd Telecom-churn-internship
```

### 2. Create Python 3.11 environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Add dataset

Place:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 5. Inspect and clean

```powershell
python -m src.inspect_data
python -m src.clean_data
```

### 6. Train and select current model

```powershell
python -m src.business_model_comparison
```

This regenerates:

```text
models/telecom_churn_model.joblib
```

### 7. Run cost sensitivity

```powershell
python -m src.cost_sensitivity_analysis
```

### 8. Test prediction

```powershell
python -m src.predict_customer
```

Current example output is approximately:

```text
Model: Logistic Regression
Decision threshold: 0.17
Prediction: Churn
Churn probability: 75.55%
```

### 9. Run tests

```powershell
python -m pytest -q
```

### 10. Start API

```powershell
uvicorn src.api:app --reload --port 8001
```

## Main Limitations

- public dataset
- simulated cost assumptions
- no real campaign-capacity constraint
- no temporal prediction horizon
- no production monitoring
- no drift detection
- no scheduled retraining
- no formal fairness analysis
- no production authentication
- no cloud deployment
- no external untouched evaluation dataset

## Key Takeaway

The project demonstrates:

```text
Business problem
→ data validation
→ cleaning
→ EDA
→ SQL
→ preprocessing
→ model comparison
→ CV/tuning
→ validation threshold selection
→ cost analysis
→ uncertainty
→ held-out evaluation
→ model serialization
→ API
→ validation
→ testing
→ documentation
→ GitHub handover
```

The main lesson is that model choice should be driven by evidence, uncertainty, business costs, and operational constraints rather than a single metric.
