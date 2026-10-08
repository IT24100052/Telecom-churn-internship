# Models

This folder stores generated trained model artifacts.

Current artifact:

```text
telecom_churn_model.joblib
```

The `.joblib` file is intentionally excluded from Git.

Regenerate it with:

```powershell
python -m src.business_model_comparison
```

The saved bundle contains:

```text
model
threshold
model_name
false_positive_cost
false_negative_cost
```

Under the current primary simulated cost assumption:

```text
FP cost = 1
FN cost = 5
```

the selected model is:

```text
Logistic Regression
Threshold = 0.17
```

The selected model can change if the business-cost assumptions change.
