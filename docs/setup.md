# Setup and Reproducibility

## Python

Recommended:

```text
Python 3.11
```

## Clone

```bash
git clone https://github.com/IT24100052/Telecom-churn-internship.git
cd Telecom-churn-internship
```

## Virtual Environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## Dataset

Place the raw CSV at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset is intentionally excluded from Git.

## Inspect

```powershell
python -m src.inspect_data
```

Expected key findings:

```text
Rows: 7043
Columns: 21
Blank TotalCharges values: 11
Duplicate rows: 0
```

## Clean

```powershell
python -m src.clean_data
```

Creates:

```text
data/processed/telco_churn_clean.csv
```

## Optional SQL

```powershell
python -m src.create_database
python -m src.sql_analysis
```

## Train and Select Model

```powershell
python -m src.business_model_comparison
```

Under the current 1:5 cost assumption, this should select:

```text
Logistic Regression
Threshold: 0.17
```

It also creates:

```text
models/telecom_churn_model.joblib
reports/validation_business_comparison.csv
reports/final_model_comparison.csv
reports/logistic_threshold_analysis.csv
reports/random_forest_threshold_analysis.csv
```

## Cost Sensitivity

```powershell
python -m src.cost_sensitivity_analysis
```

Creates:

```text
reports/cost_sensitivity_analysis.csv
```

## Prediction

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

## Tests

```powershell
python -m pytest -q
```

Expected:

```text
6 passed
```

## API

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

## Git-Ignored Generated Files

```text
.venv/
data/raw/*.csv
data/processed/*.csv
data/processed/*.db
models/*.joblib
.env
```

A fresh clone therefore needs the dataset added locally before retraining and running prediction tests.
