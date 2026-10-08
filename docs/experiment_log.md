# Experiment Log — Telecom Customer Churn Prediction

## Objective

Build an end-to-end churn prediction workflow with reproducible preprocessing, model comparison, business-oriented threshold selection, uncertainty analysis, deployment, and testing.

## Dataset

```text
Rows: 7043
Columns: 21
Target: Churn
```

Target mapping:

```text
No = 0
Yes = 1
```

## Data Quality

`TotalCharges` was loaded as string data.

Findings:

```text
Blank TotalCharges values: 11
All affected tenure values: 0
```

Cleaning rule used:

```text
blank TotalCharges → 0
```

This is a dataset-specific assumption.

## EDA

```text
Overall churn: 26.54%
Month-to-month: 42.71%
One year: 11.27%
Two year: 2.83%
0–12 months tenure: 47.44%
49–72 months tenure: 9.51%
```

## Initial Logistic Regression

```text
Accuracy:  0.8055
Precision: 0.6572
Recall:    0.5588
F1:        0.6040
ROC-AUC:   0.8421
```

Initial threshold analysis:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.30 | 0.5193 | 0.7540 | 0.6150 |
| 0.40 | 0.5682 | 0.6684 | 0.6143 |
| 0.50 | 0.6572 | 0.5588 | 0.6040 |
| 0.60 | 0.7177 | 0.4011 | 0.5146 |

## Initial Decision Tree

```text
Accuracy:  0.7984
Precision: 0.6347
Recall:    0.5668
F1:        0.5989
ROC-AUC:   0.8297
```

## Initial Random Forest

```text
Accuracy:  0.7551
Precision: 0.5258
Recall:    0.7888
F1:        0.6310
ROC-AUC:   0.8411
```

## Initial 5-Fold CV

Logistic Regression:

```text
F1: 0.5923
ROC-AUC: 0.8462
```

Decision Tree:

```text
F1: 0.5649
ROC-AUC: 0.8289
```

Random Forest:

```text
F1: 0.6357
ROC-AUC: 0.8460
```

## Evaluation Issue Found

The earlier test split was reused for threshold and model decisions.

This meant it should not be described as fully untouched.

The workflow was revised to use separate training, validation, and held-out test splits.

## Revised Split

```text
Train: 4930
Validation: 1056
Test: 1057
```

## Revised Training-Only RF Tuning

Grid:

```text
n_estimators: 100, 200
max_depth: 5, 8, 12
min_samples_split: 2, 5
```

Best:

```text
max_depth=8
min_samples_split=2
n_estimators=100
Best training CV F1=0.6331
```

## Primary Cost Scenario

```text
FP cost = 1
FN cost = 5
```

These are simulated cost units.

## Validation Comparison

Logistic Regression:

```text
Threshold: 0.17
Precision: 0.4491
Recall: 0.9143
F1: 0.6024
ROC-AUC: 0.8454
Flagged: 53.98%
FP: 314
FN: 24
Cost: 434
```

Random Forest:

```text
Threshold: 0.36
Precision: 0.4490
Recall: 0.8964
F1: 0.5983
ROC-AUC: 0.8433
Flagged: 52.94%
FP: 308
FN: 29
Cost: 453
```

## Selected Model

Under the primary 1:5 cost scenario:

```text
Logistic Regression
Threshold = 0.17
```

Reason:

- lower validation cost
- slightly higher recall
- slightly higher F1
- slightly higher ROC-AUC
- simpler model

This does not imply universal superiority.

## Final Held-Out Results — Logistic Regression

```text
Accuracy:  0.6868
Precision: 0.4547
Recall:    0.8932
F1:        0.6026
ROC-AUC:   0.8448
Flagged:   52.22%
```

Confusion matrix:

```text
TN 475
FP 301
FN 30
TP 251
```

Cost:

```text
451
```

## Final Held-Out Results — Random Forest

```text
Accuracy:  0.6916
Precision: 0.4576
Recall:    0.8648
F1:        0.5985
ROC-AUC:   0.8416
Flagged:   50.24%
```

Confusion matrix:

```text
TN 488
FP 288
FN 38
TP 243
```

Cost:

```text
478
```

## Bootstrap 95% Confidence Intervals

Logistic Regression:

```text
Precision: 0.4141–0.4991
Recall:    0.8561–0.9278
F1:        0.5617–0.6423
ROC-AUC:   0.8199–0.8712
```

Random Forest:

```text
Precision: 0.4159–0.5019
Recall:    0.8253–0.9055
F1:        0.5570–0.6392
ROC-AUC:   0.8169–0.8674
```

The intervals overlap substantially.

## Cost Sensitivity

| FN Cost | Winner | Threshold | Recall | Flagged | Cost |
|---:|---|---:|---:|---:|---:|
| 2 | Random Forest | 0.59 | 0.7429 | 35.13% | 307 |
| 3 | Random Forest | 0.52 | 0.8071 | 40.53% | 364 |
| 5 | Logistic Regression | 0.17 | 0.9143 | 53.98% | 434 |
| 10 | Random Forest | 0.24 | 0.9571 | 63.73% | 525 |
| 20 | Logistic Regression | 0.06 | 0.9786 | 71.88% | 605 |

Conclusion:

> Preferred model and threshold depend on business cost assumptions and operational capacity.

## Deployment Update

The saved model artifact now includes:

```text
model
threshold
model_name
false_positive_cost
false_negative_cost
```

This prevents analysis/deployment threshold mismatch.

## API Validation Update

The API now rejects:

- invalid categorical values
- negative tenure
- invalid charge values
- inconsistent phone-service states
- inconsistent internet-service states

Current automated test result:

```text
6 passed
```

## Current Final Conclusion

The project no longer claims that Random Forest is universally better.

Under the primary simulated 1:5 cost scenario, Logistic Regression was selected using the validation set because it produced the lower validation cost:

```text
Logistic Regression: 434 units
Random Forest:       453 units
```

On the held-out test split, Logistic Regression also had the lower observed simulated cost:

```text
Logistic Regression: 451 units
Random Forest:       478 units
```

However, the paired-bootstrap comparison showed that the 95% confidence interval for the Logistic Regression minus Random Forest cost difference included zero.

Therefore, the evidence is not strong enough to conclude that Logistic Regression consistently has lower business cost than Random Forest.

The final interpretation is:

> Logistic Regression and Random Forest are broadly competitive under the simulated 1:5 cost scenario. Logistic Regression remains the deployment model because it achieved competitive predictive performance and observed cost while also being simpler and easier to interpret. Cost-sensitivity analysis further showed that the preferred model and decision threshold can change when business assumptions change.

The project therefore treats model selection as a business and statistical trade-off rather than declaring one algorithm universally superior.
