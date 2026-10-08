# Telecom Churn Prediction — Complete Project Notes

## Problem

Predict customer churn so a telecom retention team can identify customers at higher risk of leaving.

## Why the Project Was Built

To practice a realistic AI/ML workflow:

```text
Data → analysis → SQL → preprocessing → modeling → validation
→ business cost → uncertainty → deployment → tests → documentation
```

## Dataset

```text
7043 rows
21 original columns
```

Public telecom churn data only.

## Data Cleaning

`TotalCharges` contained 11 blank strings.

All 11 had:

```text
tenure = 0
```

They were replaced with 0 for this project.

## Main EDA Findings

```text
Overall churn: 26.54%
Month-to-month churn: 42.71%
Two-year churn: 2.83%
0–12 month churn: 47.44%
49–72 month churn: 9.51%
```

These are associations, not causal conclusions.

## SQL

SQLite was used to practice:

- SELECT
- WHERE
- GROUP BY
- ORDER BY
- CASE WHEN
- COUNT
- SUM
- HAVING

## ML Features

Removed:

```text
customerID
```

Target:

```text
Churn
```

Preprocessing:

```text
StandardScaler
OneHotEncoder
ColumnTransformer
Pipeline
```

## Models

Compared:

- Logistic Regression
- Decision Tree
- Random Forest

## Important Evaluation Correction

The original test set was reused during model development.

The workflow was revised to:

```text
Train: 4930
Validation: 1056
Test: 1057
```

## Revised RF Tuning

Training-only best parameters:

```text
max_depth=8
min_samples_split=2
n_estimators=100
```

## Why Cost Analysis Was Added

Earlier reasoning said missing a churner was more expensive than an unnecessary contact, but it did not quantify the trade-off.

Primary simulated scenario:

```text
FP cost = 1
FN cost = 5
```

## Validation Results

Logistic Regression:

```text
Threshold: 0.17
Recall: 0.9143
F1: 0.6024
ROC-AUC: 0.8454
Cost: 434
```

Random Forest:

```text
Threshold: 0.36
Recall: 0.8964
F1: 0.5983
ROC-AUC: 0.8433
Cost: 453
```

## Current Selected Model

```text
Logistic Regression
Threshold: 0.17
```

It is selected under the current 1:5 cost scenario.

## Final Held-Out Results

```text
Accuracy: 0.6868
Precision: 0.4547
Recall: 0.8932
F1: 0.6026
ROC-AUC: 0.8448
```

Confusion matrix:

```text
TN 475
FP 301
FN 30
TP 251
```

## Uncertainty

LR 95% bootstrap intervals:

```text
Precision: 0.4141–0.4991
Recall: 0.8561–0.9278
F1: 0.5617–0.6423
ROC-AUC: 0.8199–0.8712
```

RF intervals overlap substantially.

Therefore, the project does not claim LR is statistically clearly superior.

## Cost Sensitivity

```text
FN cost 2  → Random Forest @ 0.59
FN cost 3  → Random Forest @ 0.52
FN cost 5  → Logistic Regression @ 0.17
FN cost 10 → Random Forest @ 0.24
FN cost 20 → Logistic Regression @ 0.06
```

Main lesson:

> Model and threshold choice depend on business economics.

## Campaign Capacity

At high missed-churn cost, the model flags a large share of customers.

A real system should also include campaign-capacity limits.

## Saved Model Bundle

Contains:

```text
model
threshold
model_name
false_positive_cost
false_negative_cost
```

This ensures the deployed threshold matches the validation-selected threshold.

## API Improvement

The API now rejects invalid categories such as:

```text
Contract = banana
```

and checks service consistency.

## Tests

Current result:

```text
6 passed
```

## Reproducibility

A fresh clone can reproduce the project after adding the dataset locally:

```powershell
python -m src.inspect_data
python -m src.clean_data
python -m src.business_model_comparison
python -m src.cost_sensitivity_analysis
python -m src.predict_customer
python -m pytest -q
```

## Correct Final Explanation

> I initially favored Random Forest because of recall. After review, I strengthened the evaluation by re-tuning on the revised training split, selecting thresholds on validation data, adding explicit simulated business costs, bootstrap confidence intervals, and cost sensitivity. Under the primary 1:5 cost scenario, Logistic Regression at threshold 0.17 produced the lowest validation cost and became the deployed model. The models remain close, and different business assumptions can change the preferred model and threshold.
