# Telecom Churn Project — Supervisor Questions & Answers

## 1. What problem are you solving?

Predicting telecom customer churn so a retention team can prioritize customers at risk of leaving.

## 2. What type of problem is it?

Supervised binary classification.

## 3. What dataset did you use?

A public telecom churn dataset with 7,043 rows and 21 original columns.

## 4. What data-quality issue did you find?

`TotalCharges` contained 11 blank strings and was loaded as a string column.

## 5. Why didn't `isna()` find them?

Because blank strings are not the same as NaN.

## 6. Why set those values to 0?

All affected customers had tenure 0. It was a dataset-specific assumption for new customers.

## 7. What did EDA show?

Month-to-month and short-tenure customers had much higher churn rates.

## 8. Does that prove causation?

No. It shows association only.

## 9. Why use SQL?

To simulate how ML work often starts from database queries rather than CSV files.

## 10. Why remove customerID?

It is an identifier rather than a meaningful behavioral feature.

## 11. Why use a pipeline?

To keep preprocessing consistent and reduce leakage risk.

## 12. What models did you compare?

Logistic Regression, Decision Tree, and Random Forest.

## 13. Why wasn't accuracy enough?

Because churn is the minority class and accuracy can hide poor churn detection.

## 14. What is recall here?

The percentage of actual churners correctly identified.

## 15. Why did you initially prefer Random Forest?

Earlier experiments showed stronger recall.

## 16. Why did you revise that conclusion?

The models were being compared at different operating points, and the earlier test data had been reused.

## 17. What is the revised split?

```text
Train 4930
Validation 1056
Test 1057
```

## 18. Why re-tune RF?

To ensure hyperparameter search used only the revised training split.

## 19. What are the revised RF parameters?

```text
max_depth=8
min_samples_split=2
n_estimators=100
```

## 20. Why add business cost?

Because “missing churn is worse” was too vague without a quantitative assumption.

## 21. What primary cost assumption did you use?

```text
FP = 1
FN = 5
```

## 22. Are those real telecom costs?

No. They are simulated units.

## 23. What LR threshold was selected?

0.17.

## 24. What RF threshold was selected?

0.36.

## 25. Which model won under 1:5?

Logistic Regression.

## 26. Why?

Validation cost was 434 for LR and 453 for RF.

## 27. What were LR validation metrics?

```text
Precision 0.4491
Recall 0.9143
F1 0.6024
ROC-AUC 0.8454
```

## 28. What were RF validation metrics?

```text
Precision 0.4490
Recall 0.8964
F1 0.5983
ROC-AUC 0.8433
```

## 29. Is LR definitely better?

No. The differences are small and bootstrap confidence intervals overlap.

## 30. What are LR final held-out metrics?

```text
Accuracy 0.6868
Precision 0.4547
Recall 0.8932
F1 0.6026
ROC-AUC 0.8448
```

## 31. What is the LR confusion matrix?

```text
TN 475
FP 301
FN 30
TP 251
```

## 32. What does 89.32% recall mean?

About 89% of actual churners were identified in the held-out test split.

## 33. Why is precision lower?

The low threshold intentionally catches more churners and therefore creates more false positives.

## 34. Why not always use 0.50?

The best operating threshold depends on business trade-offs.

## 35. What did bootstrap intervals show?

There is meaningful uncertainty and heavy overlap between LR and RF.

## 36. What is the LR recall CI?

Approximately 0.8561–0.9278.

## 37. What is the LR F1 CI?

Approximately 0.5617–0.6423.

## 38. What did cost sensitivity show?

The preferred model changes with the cost of missing a churner.

## 39. Which model wins at FN cost 2?

Random Forest at threshold 0.59.

## 40. Which wins at FN cost 5?

Logistic Regression at threshold 0.17.

## 41. Which wins at FN cost 10?

Random Forest at threshold 0.24.

## 42. What is the main lesson from sensitivity analysis?

There is no universally best model or threshold.

## 43. What operational issue appears at high FN cost?

The model flags a very large share of customers.

## 44. What should be added in a real retention campaign?

Campaign-capacity constraints.

## 45. Why save the threshold with the model?

To make deployment use the same decision rule selected during validation.

## 46. What if you used `.predict()` after selecting 0.17?

It could use the classifier's default decision rule and break deployment consistency.

## 47. What does the model bundle store?

```text
model
threshold
model_name
false_positive_cost
false_negative_cost
```

## 48. Which model is currently deployed?

Logistic Regression at threshold 0.17.

## 49. What is the example churn probability now?

About 75.55%.

## 50. Why did it change from 87.48%?

The old prediction came from Random Forest; the current deployed model is Logistic Regression.

## 51. Why strengthen API validation?

Because unknown strings could previously be silently ignored by one-hot encoding.

## 52. What happens with `Contract="banana"` now?

HTTP 422.

## 53. Why 422 instead of 500?

422 means invalid client input; 500 means an unexpected server error.

## 54. What service-consistency checks exist?

PhoneService/MultipleLines and InternetService/internet-dependent service consistency.

## 55. Why not force TotalCharges = tenure × MonthlyCharges?

Current monthly charges may not equal historical monthly charges.

## 56. How many tests pass?

6.

## 57. What do they test?

Health, valid prediction, negative tenure, invalid contract, inconsistent phone service, inconsistent internet service.

## 58. Why simplify requirements.txt?

To remove unrelated packages and improve portability.

## 59. Why change its encoding?

The older file was UTF-16; UTF-8 is more portable for pip and CI.

## 60. Can a fresh clone run immediately?

It needs the dataset added locally because the CSV is intentionally ignored.

## 61. How do you regenerate the model?

```powershell
python -m src.business_model_comparison
```

## 62. Why is the `.joblib` ignored?

It is generated and reproducible.

## 63. Why isn't the final test called externally untouched?

Earlier experiments used alternative splits from the same dataset, so some information exposure occurred during the broader project history.

## 64. What is the strongest current conclusion?

Under the simulated 1:5 cost scenario, LR at threshold 0.17 produced the lowest validation cost.

## 65. What is the biggest caveat?

The business costs are simulated.

## 66. What business data would you request next?

Retention contact cost, churn value, customer lifetime value, campaign capacity, prediction horizon, and retention success rate.

## 67. Why not maximize F1 only?

F1 weights precision and recall symmetrically, while business costs may not.

## 68. Why not maximize recall only?

You could flag almost everyone and get high recall, which may be operationally useless.

## 69. Why not select by ROC-AUC only?

ROC-AUC is threshold-independent, but deployment requires a specific operating point.

## 70. Would a cost change always require retraining?

No. Sometimes the same probability model can be kept and only the threshold changed.

## 71. What if campaign capacity is 20%?

I would select the top-risk customers or choose a threshold constrained to approximately 20% flagged.

## 72. What if production performance drops?

Check schema changes, drift, class balance, target definition, and pipeline consistency before retraining.

## 73. What is the biggest evaluation limitation?

No independent external dataset.

## 74. What is the biggest deployment limitation?

No production authentication, monitoring, model registry, or cloud infrastructure.

## 75. What is the biggest strength?

The project covers the full workflow beyond model training.

## 76. What did you learn from the RF-vs-LR correction?

A model should not be called “best” based on one metric or threshold.

## 77. What did you learn from the test-set correction?

Repeated test use can make evaluation less trustworthy.

## 78. What did you learn from API validation?

Model robustness is not a replacement for input validation.

## 79. What did you learn from cost analysis?

Model selection is part of a business decision.

## 80. Is this production-ready?

No. It is a strong portfolio/internship project, but production would require real business costs, monitoring, security, governance, and external validation.

## 81. One-minute explanation

I built an end-to-end telecom churn prediction system using a public dataset of 7,043 customers. I inspected and cleaned the data, performed EDA and SQL analysis, built preprocessing pipelines, and compared Logistic Regression, Decision Tree, and Random Forest. After review, I improved the evaluation by separating training, validation, and held-out test data, re-tuning Random Forest only on training data, selecting thresholds using validation business cost, and adding bootstrap confidence intervals and cost-sensitivity analysis. Under a simulated 1:5 false-positive-to-false-negative cost ratio, Logistic Regression at threshold 0.17 had the lowest validation cost and became the deployed model. I saved the model with its threshold, exposed it through FastAPI, added strict validation and six automated tests, and documented the workflow.

## 82. Best answer if asked about your biggest correction

I initially over-stated the Random Forest advantage and the independence of the final test. After review, I corrected both by using validation-based model selection, explicit cost assumptions, uncertainty analysis, and more careful wording.

## 83. What would you improve next?

- probability calibration
- SHAP explanations
- campaign-capacity analysis
- drift monitoring
- model registry
- CI/CD
- deployment
- fairness checks

## 84. Main lesson

Good ML requires reliable data, fair evaluation, uncertainty, business trade-offs, deployment consistency, testing, and clear communication—not only a high metric.
