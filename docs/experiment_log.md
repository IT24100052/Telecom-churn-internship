# Experiment Log

## Project
Telecom Customer Churn Prediction

## Objective
Predict customer churn while prioritizing recall because missing a real churner is assumed to be more costly than contacting some customers who would have stayed.

## Data Preparation
- Dataset size: 7,043 customers
- Target: Churn
- Removed from model features:
  - customerID
  - Churn
- `TotalCharges` contained 11 blank values.
- All 11 affected customers had tenure = 0.
- Blank `TotalCharges` values were treated as 0 and converted to numeric.

## Initial Train/Test Experiment
Split:
- 80% training
- 20% test
- stratified by churn
- random_state = 42

### Logistic Regression
Accuracy: 0.8055
Precision: 0.6572
Recall: 0.5588
F1: 0.6040
ROC-AUC: 0.8421

### Threshold Analysis
Threshold 0.30:
- Precision: 0.5193
- Recall: 0.7540
- F1: 0.6150

Threshold 0.40:
- Precision: 0.5682
- Recall: 0.6684
- F1: 0.6143

Threshold 0.50:
- Precision: 0.6572
- Recall: 0.5588
- F1: 0.6040

### Decision Tree
Accuracy: 0.7984
Precision: 0.6347
Recall: 0.5668
F1: 0.5989
ROC-AUC: 0.8297

### Random Forest
Accuracy: 0.7551
Precision: 0.5258
Recall: 0.7888
F1: 0.6310
ROC-AUC: 0.8411

## Cross-Validation
5-fold stratified cross-validation on training data.

### Logistic Regression
Accuracy: 0.8021
Precision: 0.6529
Recall: 0.5431
F1: 0.5923
ROC-AUC: 0.8462

### Decision Tree
Accuracy: 0.7914
Precision: 0.6345
Recall: 0.5130
F1: 0.5649
ROC-AUC: 0.8289

### Random Forest
Accuracy: 0.7593
Precision: 0.5315
Recall: 0.7913
F1: 0.6357
ROC-AUC: 0.8460

## Random Forest Tuning
Search space:
- n_estimators: 100, 200
- max_depth: 5, 8, 12
- min_samples_split: 2, 5

Best parameters:
- n_estimators = 100
- max_depth = 8
- min_samples_split = 5

Best CV F1:
0.6364

## Corrected Final Evaluation Design
The earlier test set had been used repeatedly during model comparison, so it was no longer treated as a true final test set.

A new split was created:

- Training: 4,930
- Validation: 1,056
- Final test: 1,057

Churn rate remained approximately 26.5% across all splits.

## Validation Results

### Logistic Regression at threshold 0.30
Precision: 0.5188
Recall: 0.7893
F1: 0.6261
ROC-AUC: 0.8454

### Random Forest
Accuracy: 0.7509
Precision: 0.5192
Recall: 0.8214
F1: 0.6362
ROC-AUC: 0.8422

## Selected Model
Random Forest

Reason:
The Random Forest achieved higher recall and F1 while maintaining similar precision to Logistic Regression at the business-oriented threshold.

## Final Held-Out Test Results
Accuracy: 0.7588
Precision: 0.5319
Recall: 0.7722
F1: 0.6299
ROC-AUC: 0.8390

Confusion matrix:
- TN: 585
- FP: 191
- FN: 64
- TP: 217

## Explainability
Top permutation-importance features:
- Contract
- InternetService
- tenure
- TotalCharges
- PaperlessBilling
- OnlineBackup
- StreamingTV
- PaymentMethod

## Notes
- Feature importance indicates predictive usefulness, not causation.
- Final test results should not be used for further tuning.
- Future work should include monitoring, drift detection, calibration, cost-sensitive evaluation, and retraining strategy.
