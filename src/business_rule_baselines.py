from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path(
    "data/processed/telco_churn_clean.csv"
)

MODEL_PATH = Path(
    "models/telecom_churn_model.joblib"
)

REPORT_PATH = Path(
    "reports/business_rule_baselines.csv"
)

RANDOM_STATE = 42

FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 5


def evaluate_rule(
    name,
    y_true,
    predictions,
    probabilities=None,
):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    result = {
        "strategy": name,
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "flagged_count": int(
            predictions.sum()
        ),
        "flagged_rate": float(
            predictions.mean()
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "business_cost": int(
            fp * FALSE_POSITIVE_COST
            + fn * FALSE_NEGATIVE_COST
        ),
    }

    if probabilities is not None:
        result["roc_auc"] = roc_auc_score(
            y_true,
            probabilities,
        )
    else:
        result["roc_auc"] = None

    return result


def main():
    df = pd.read_csv(
        DATA_PATH
    )

    X = df.drop(
        columns=[
            "customerID",
            "Churn",
        ]
    )

    y = df["Churn"].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    # Recreate the same revised split
    # used by business_model_comparison.py
    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    _, X_test, _, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=y_temp,
        )
    )

    rows = []

    # -------------------------------------------------
    # Baseline 1: predict nobody churns
    # -------------------------------------------------

    predict_none = pd.Series(
        0,
        index=X_test.index,
        dtype=int,
    )

    rows.append(
        evaluate_rule(
            "Predict nobody churns",
            y_test,
            predict_none,
        )
    )

    # -------------------------------------------------
    # Baseline 2: predict everybody churns
    # -------------------------------------------------

    predict_all = pd.Series(
        1,
        index=X_test.index,
        dtype=int,
    )

    rows.append(
        evaluate_rule(
            "Predict everybody churns",
            y_test,
            predict_all,
        )
    )

    # -------------------------------------------------
    # Baseline 3:
    # flag every month-to-month customer
    # -------------------------------------------------

    month_to_month = (
        X_test["Contract"]
        == "Month-to-month"
    ).astype(int)

    rows.append(
        evaluate_rule(
            "Flag all month-to-month customers",
            y_test,
            month_to_month,
        )
    )

    # -------------------------------------------------
    # ML model
    # -------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. "
            "Run "
            "'python -m src.business_model_comparison' "
            "first."
        )

    bundle = joblib.load(
        MODEL_PATH
    )

    model = bundle["model"]
    threshold = float(
        bundle["threshold"]
    )
    model_name = bundle[
        "model_name"
    ]

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    model_predictions = (
        probabilities >= threshold
    ).astype(int)

    rows.append(
        evaluate_rule(
            f"ML model: {model_name}",
            y_test,
            model_predictions,
            probabilities,
        )
    )

    results = pd.DataFrame(
        rows
    )

    print(
        "=== BUSINESS-RULE BASELINES "
        "VS ML MODEL ==="
    )

    print(
        results[
            [
                "strategy",
                "precision",
                "recall",
                "f1",
                "flagged_rate",
                "fp",
                "fn",
                "business_cost",
                "roc_auc",
            ]
        ].to_string(
            index=False
        )
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        REPORT_PATH,
        index=False,
    )

    print(
        "\nSaved:",
        REPORT_PATH,
    )

    print(
        "\nCost assumption:"
    )

    print(
        f"False positive = "
        f"{FALSE_POSITIVE_COST} unit"
    )

    print(
        f"False negative = "
        f"{FALSE_NEGATIVE_COST} units"
    )


if __name__ == "__main__":
    main()