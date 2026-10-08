from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


DATA_PATH = Path(
    "data/processed/telco_churn_clean.csv"
)

REPORT_PATH = Path(
    "reports/cost_sensitivity_analysis.csv"
)

RANDOM_STATE = 42


def build_preprocessor(
    numeric_features,
    categorical_features,
):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )


def calculate_result(
    y_true,
    probabilities,
    threshold,
    false_positive_cost,
    false_negative_cost,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    total_cost = (
        fp * false_positive_cost
        + fn * false_negative_cost
    )

    return {
        "threshold": float(threshold),
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
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "flagged_rate": float(
            predictions.mean()
        ),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
        "business_cost": int(total_cost),
    }


def find_best_threshold(
    y_true,
    probabilities,
    fp_cost,
    fn_cost,
):
    candidates = []

    for threshold in np.arange(
        0.05,
        0.96,
        0.01,
    ):
        result = calculate_result(
            y_true,
            probabilities,
            threshold,
            fp_cost,
            fn_cost,
        )

        candidates.append(result)

    results = pd.DataFrame(
        candidates
    )

    best = results.sort_values(
        by=[
            "business_cost",
            "f1",
        ],
        ascending=[
            True,
            False,
        ],
    ).iloc[0]

    return best


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

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    (
        X_validation,
        _,
        y_validation,
        _,
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    numeric_features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    categorical_features = [
        column
        for column in X.columns
        if column
        not in numeric_features
    ]

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    logistic_model.fit(
        X_train,
        y_train,
    )

    logistic_probabilities = (
        logistic_model.predict_proba(
            X_validation
        )[:, 1]
    )

    rf_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    parameter_grid = {
        "classifier__n_estimators": [
            100,
            200,
        ],
        "classifier__max_depth": [
            5,
            8,
            12,
        ],
        "classifier__min_samples_split": [
            2,
            5,
        ],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    search.fit(
        X_train,
        y_train,
    )

    random_forest_model = (
        search.best_estimator_
    )

    random_forest_probabilities = (
        random_forest_model.predict_proba(
            X_validation
        )[:, 1]
    )

    # FP cost stays at 1.
    # We vary the cost of missing a real churner.
    cost_scenarios = [
        2,
        3,
        5,
        10,
        20,
    ]

    rows = []

    for fn_cost in cost_scenarios:
        for model_name, probabilities in [
            (
                "Logistic Regression",
                logistic_probabilities,
            ),
            (
                "Random Forest",
                random_forest_probabilities,
            ),
        ]:
            best = find_best_threshold(
                y_validation,
                probabilities,
                fp_cost=1,
                fn_cost=fn_cost,
            )

            rows.append(
                {
                    "model": model_name,
                    "fp_cost": 1,
                    "fn_cost": fn_cost,
                    "threshold": (
                        best["threshold"]
                    ),
                    "precision": (
                        best["precision"]
                    ),
                    "recall": (
                        best["recall"]
                    ),
                    "f1": best["f1"],
                    "roc_auc": (
                        best["roc_auc"]
                    ),
                    "flagged_rate": (
                        best["flagged_rate"]
                    ),
                    "fp": int(
                        best["fp"]
                    ),
                    "fn": int(
                        best["fn"]
                    ),
                    "business_cost": int(
                        best["business_cost"]
                    ),
                }
            )

    results = pd.DataFrame(
        rows
    )

    print(
        "=== COST SENSITIVITY ANALYSIS ==="
    )

    print(
        results.to_string(
            index=False
        )
    )

    print(
        "\n=== BEST MODEL BY COST SCENARIO ==="
    )

    winners = (
        results.sort_values(
            [
                "fn_cost",
                "business_cost",
                "f1",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .groupby(
            "fn_cost",
            as_index=False,
        )
        .first()
    )

    print(
        winners[
            [
                "fn_cost",
                "model",
                "threshold",
                "precision",
                "recall",
                "f1",
                "flagged_rate",
                "business_cost",
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
        "\nIMPORTANT:"
    )

    print(
        "These are simulated cost scenarios, "
        "not real telecom business costs."
    )


if __name__ == "__main__":
    main()