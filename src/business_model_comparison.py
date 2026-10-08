from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/processed/telco_churn_clean.csv")
MODEL_PATH = Path("models/telecom_churn_model.joblib")
REPORTS_DIR = Path("reports")

RANDOM_STATE = 42

# -------------------------------------------------
# Simulated business cost assumptions
# -------------------------------------------------
# These are NOT real telecom costs.
#
# One unnecessary retention contact = 1 cost unit.
# Missing one real churner = 5 cost units.
# -------------------------------------------------

FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 5

BOOTSTRAP_SAMPLES = 1000


def build_preprocessor(
    numeric_features,
    categorical_features,
):
    """
    Create the preprocessing pipeline.

    Numeric features:
        StandardScaler

    Categorical features:
        OneHotEncoder
    """

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
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
        ]
    )


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    """
    Convert probabilities into predictions using
    the supplied threshold and calculate metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    business_cost = (
        fp * FALSE_POSITIVE_COST
        + fn * FALSE_NEGATIVE_COST
    )

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "flagged_count": int(
            predictions.sum()
        ),
        "flagged_rate": float(
            predictions.mean()
        ),
        "business_cost": int(
            business_cost
        ),
    }


def find_best_threshold(
    y_true,
    probabilities,
):
    """
    Select a threshold using the validation set.

    Primary objective:
        Lowest simulated business cost.

    Tie breaker:
        Higher F1 score.
    """

    rows = []

    for threshold in np.arange(
        0.10,
        0.91,
        0.01,
    ):
        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )

        rows.append(metrics)

    results = pd.DataFrame(rows)

    best_row = results.sort_values(
        by=[
            "business_cost",
            "f1",
        ],
        ascending=[
            True,
            False,
        ],
    ).iloc[0]

    return (
        float(best_row["threshold"]),
        results,
    )


def bootstrap_confidence_intervals(
    y_true,
    probabilities,
    threshold,
    n_bootstrap=1000,
    random_state=42,
):
    """
    Estimate 95% bootstrap confidence intervals
    for key evaluation metrics.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(
        probabilities
    )

    rng = np.random.default_rng(
        random_state
    )

    bootstrap_results = []

    for _ in range(n_bootstrap):
        indices = rng.integers(
            0,
            len(y_true),
            len(y_true),
        )

        sampled_y = y_true[indices]

        sampled_probabilities = (
            probabilities[indices]
        )

        # ROC-AUC requires both classes.
        if len(
            np.unique(sampled_y)
        ) < 2:
            continue

        metrics = calculate_metrics(
            sampled_y,
            sampled_probabilities,
            threshold,
        )

        bootstrap_results.append(
            {
                "precision": metrics[
                    "precision"
                ],
                "recall": metrics[
                    "recall"
                ],
                "f1": metrics[
                    "f1"
                ],
                "roc_auc": metrics[
                    "roc_auc"
                ],
            }
        )

    bootstrap_df = pd.DataFrame(
        bootstrap_results
    )

    intervals = {}

    for metric in [
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:
        intervals[metric] = (
            bootstrap_df[
                metric
            ].quantile(0.025),
            bootstrap_df[
                metric
            ].quantile(0.975),
        )

    return intervals


def print_result(
    name,
    metrics,
    intervals,
):
    """
    Print final model evaluation results.
    """

    print(
        f"\n=== {name} ==="
    )

    print(
        f"Threshold: "
        f"{metrics['threshold']:.2f}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        "Flagged customers: "
        f"{metrics['flagged_count']} "
        f"({metrics['flagged_rate'] * 100:.2f}%)"
    )

    print(
        "Confusion matrix: "
        f"TN={metrics['tn']}, "
        f"FP={metrics['fp']}, "
        f"FN={metrics['fn']}, "
        f"TP={metrics['tp']}"
    )

    print(
        "Business cost: "
        f"{metrics['business_cost']} units"
    )

    print(
        "\n95% bootstrap "
        "confidence intervals:"
    )

    for metric, interval in (
        intervals.items()
    ):
        print(
            f"{metric}: "
            f"{interval[0]:.4f} "
            f"- {interval[1]:.4f}"
        )


def main():

    # -------------------------------------------------
    # Load cleaned dataset
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Revised 70 / 15 / 15 split
    # -------------------------------------------------

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
        X_test,
        y_validation,
        y_test,
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print(
        "=== REVISED DATA SPLIT ==="
    )

    print(
        "Training:",
        X_train.shape,
    )

    print(
        "Validation:",
        X_validation.shape,
    )

    print(
        "Final test:",
        X_test.shape,
    )

    # -------------------------------------------------
    # Feature definitions
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------

    logistic_pipeline = Pipeline(
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

    logistic_pipeline.fit(
        X_train,
        y_train,
    )

    logistic_validation_probabilities = (
        logistic_pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    (
        logistic_threshold,
        logistic_threshold_table,
    ) = find_best_threshold(
        y_validation,
        logistic_validation_probabilities,
    )

    # -------------------------------------------------
    # Random Forest
    #
    # Hyperparameter tuning uses ONLY
    # the revised training split.
    # -------------------------------------------------

    random_forest_pipeline = Pipeline(
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
        estimator=(
            random_forest_pipeline
        ),
        param_grid=parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    search.fit(
        X_train,
        y_train,
    )

    random_forest_pipeline = (
        search.best_estimator_
    )

    print(
        "\n=== RANDOM FOREST "
        "TRAINING-ONLY TUNING ==="
    )

    print(
        "Best parameters:"
    )

    print(
        search.best_params_
    )

    print(
        "Best training CV F1:",
        round(
            search.best_score_,
            4,
        ),
    )

    random_forest_validation_probabilities = (
        random_forest_pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    (
        random_forest_threshold,
        random_forest_threshold_table,
    ) = find_best_threshold(
        y_validation,
        random_forest_validation_probabilities,
    )

    # -------------------------------------------------
    # Validation comparison
    # -------------------------------------------------

    logistic_validation_metrics = (
        calculate_metrics(
            y_validation,
            logistic_validation_probabilities,
            logistic_threshold,
        )
    )

    random_forest_validation_metrics = (
        calculate_metrics(
            y_validation,
            random_forest_validation_probabilities,
            random_forest_threshold,
        )
    )

    validation_comparison = (
        pd.DataFrame(
            [
                {
                    "model":
                    "Logistic Regression",
                    **logistic_validation_metrics,
                },
                {
                    "model":
                    "Random Forest",
                    **random_forest_validation_metrics,
                },
            ]
        )
    )

    print(
        "\n=== VALIDATION "
        "COST-BASED COMPARISON ==="
    )

    print(
        validation_comparison[
            [
                "model",
                "threshold",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "flagged_rate",
                "fp",
                "fn",
                "business_cost",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # Champion selection
    #
    # IMPORTANT:
    # Selection is based ONLY on validation data.
    # The final test set is not used here.
    # -------------------------------------------------

    if (
        logistic_validation_metrics[
            "business_cost"
        ]
        <=
        random_forest_validation_metrics[
            "business_cost"
        ]
    ):
        champion_name = (
            "Logistic Regression"
        )

        champion_pipeline = (
            logistic_pipeline
        )

        champion_threshold = (
            logistic_threshold
        )

    else:
        champion_name = (
            "Random Forest"
        )

        champion_pipeline = (
            random_forest_pipeline
        )

        champion_threshold = (
            random_forest_threshold
        )

    print(
        "\n=== CHAMPION SELECTED "
        "FROM VALIDATION ==="
    )

    print(
        "Model:",
        champion_name,
    )

    print(
        "Threshold:",
        round(
            champion_threshold,
            2,
        ),
    )

    # -------------------------------------------------
    # Save champion model bundle
    # -------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bundle = {
        "model": (
            champion_pipeline
        ),
        "threshold": float(
            champion_threshold
        ),
        "model_name": (
            champion_name
        ),
        "false_positive_cost": (
            FALSE_POSITIVE_COST
        ),
        "false_negative_cost": (
            FALSE_NEGATIVE_COST
        ),
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    print(
        "Saved:",
        MODEL_PATH,
    )

    # -------------------------------------------------
    # Final evaluation
    #
    # Thresholds were selected on validation.
    #
    # Do NOT change thresholds after looking
    # at these test results.
    # -------------------------------------------------

    logistic_test_probabilities = (
        logistic_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    random_forest_test_probabilities = (
        random_forest_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    logistic_test_metrics = (
        calculate_metrics(
            y_test,
            logistic_test_probabilities,
            logistic_threshold,
        )
    )

    random_forest_test_metrics = (
        calculate_metrics(
            y_test,
            random_forest_test_probabilities,
            random_forest_threshold,
        )
    )

    # -------------------------------------------------
    # Bootstrap confidence intervals
    # -------------------------------------------------

    logistic_intervals = (
        bootstrap_confidence_intervals(
            y_test.to_numpy(),
            logistic_test_probabilities,
            logistic_threshold,
            BOOTSTRAP_SAMPLES,
            RANDOM_STATE,
        )
    )

    random_forest_intervals = (
        bootstrap_confidence_intervals(
            y_test.to_numpy(),
            random_forest_test_probabilities,
            random_forest_threshold,
            BOOTSTRAP_SAMPLES,
            RANDOM_STATE,
        )
    )

    print_result(
        "LOGISTIC REGRESSION "
        "FINAL EVALUATION",
        logistic_test_metrics,
        logistic_intervals,
    )

    print_result(
        "RANDOM FOREST "
        "FINAL EVALUATION",
        random_forest_test_metrics,
        random_forest_intervals,
    )

    # -------------------------------------------------
    # Save evidence files
    # -------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_comparison.to_csv(
        REPORTS_DIR
        / "validation_business_comparison.csv",
        index=False,
    )

    final_comparison = pd.DataFrame(
        [
            {
                "model":
                "Logistic Regression",
                **logistic_test_metrics,
            },
            {
                "model":
                "Random Forest",
                **random_forest_test_metrics,
            },
        ]
    )

    final_comparison.to_csv(
        REPORTS_DIR
        / "final_model_comparison.csv",
        index=False,
    )

    logistic_threshold_table.to_csv(
        REPORTS_DIR
        / "logistic_threshold_analysis.csv",
        index=False,
    )

    random_forest_threshold_table.to_csv(
        REPORTS_DIR
        / "random_forest_threshold_analysis.csv",
        index=False,
    )

    print(
        "\n=== EVIDENCE FILES SAVED ==="
    )

    print(
        "reports/"
        "validation_business_comparison.csv"
    )

    print(
        "reports/"
        "final_model_comparison.csv"
    )

    print(
        "reports/"
        "logistic_threshold_analysis.csv"
    )

    print(
        "reports/"
        "random_forest_threshold_analysis.csv"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The test results are reported "
        "under the revised split."
    )

    print(
        "However, earlier exploratory "
        "experiments used alternative "
        "splits from the same dataset."
    )

    print(
        "Therefore these results should "
        "not be described as an externally "
        "untouched evaluation."
    )


if __name__ == "__main__":
    main()