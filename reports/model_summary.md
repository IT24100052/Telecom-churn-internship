# Telecom Churn Model Summary

## Selected Model

Random Forest Classifier

## Final Test Results

- Accuracy: 0.7588
- Precision: 0.5319
- Recall: 0.7722
- F1-score: 0.6299
- ROC-AUC: 0.8390

## Confusion Matrix

- True Negatives: 585
- False Positives: 191
- False Negatives: 64
- True Positives: 217

## Main Business Interpretation

The final model identified approximately 77% of actual churners in the held-out test set.

The project prioritized recall because the simulated business assumption was that missing a real churner was more costly than contacting a customer who would have stayed.

## Main EDA Findings

- Overall churn rate: 26.54%
- Month-to-month churn: 42.71%
- One-year contract churn: 11.27%
- Two-year contract churn: 2.83%
- 0–12 month tenure churn: 47.44%
- 49–72 month tenure churn: 9.51%

## Important Predictive Features

Permutation importance highlighted:

- Contract
- InternetService
- tenure
- TotalCharges
- PaperlessBilling
- OnlineBackup
- StreamingTV
- PaymentMethod

These are predictive relationships and should not be interpreted as proof of causation.