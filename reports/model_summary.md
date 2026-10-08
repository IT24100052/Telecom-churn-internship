# Telecom Churn Model Summary

## Selected Model

```text
Logistic Regression
```

Decision threshold:

```text
0.17
```

Primary simulated cost assumption:

```text
FP cost = 1
FN cost = 5
```

These are not real telecom business costs.

## Validation Comparison

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | Flagged | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.17 | 0.4491 | 0.9143 | 0.6024 | 0.8454 | 53.98% | 434 |
| Random Forest | 0.36 | 0.4490 | 0.8964 | 0.5983 | 0.8433 | 52.94% | 453 |

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

Flagged rate:

```text
52.22%
```

## 95% Bootstrap Confidence Intervals

```text
Precision: 0.4141–0.4991
Recall: 0.8561–0.9278
F1: 0.5617–0.6423
ROC-AUC: 0.8199–0.8712
```

## Cost Sensitivity

| FN Cost | Best Model | Threshold |
|---:|---|---:|
| 2 | Random Forest | 0.59 |
| 3 | Random Forest | 0.52 |
| 5 | Logistic Regression | 0.17 |
| 10 | Random Forest | 0.24 |
| 20 | Logistic Regression | 0.06 |

## Interpretation

Logistic Regression is the selected model only under the current primary 1:5 cost scenario.

The preferred model and threshold can change when business costs change.

The final test should be described as a held-out test under the revised workflow, not an externally untouched evaluation.

## Model Artifact

```text
models/telecom_churn_model.joblib
```

The bundle contains:

- model pipeline
- selected threshold
- model name
- simulated cost assumptions

Regenerate with:

```powershell
python -m src.business_model_comparison
```
