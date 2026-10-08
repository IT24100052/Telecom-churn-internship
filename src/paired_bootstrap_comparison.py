from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.business_model_comparison import (
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    RANDOM_STATE,
    build_preprocessor,
    find_best_threshold,
)


DATA_PATH = Path(
    "data/processed/telco_churn_clean.csv"
)

REPORT_PATH = Path(
    "reports/paired_bootstrap_model_comparison.csv"
)

BOOTSTRAP_SAMPLES_PATH = Path(
    "reports/paired_bootstrap_cost_samples.csv"
)

BOOTSTRAP_SAMPLES = 1000


def calculate_cost(
    y_true,
    predictions,
):
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)

    false_positives = np.sum(
        (y_true == 0)
        & (predictions == 1)
    )

    false_negatives = np.sum(
        (y_true == 1)
        & (predictions == 0)
    )

    return (
        false_positives
        * FALSE_POSITIVE_COST
        +
        false_negatives
        * FALSE_NEGATIVE_COST
    )


def main():
    # ---------------------------------------------
    # Load data
    # ---------------------------------------------

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

    # ---------------------------------------------
    # Recreate the same revised 70/15/15 split
    # ---------------------------------------------

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
        "=== DATA SPLIT ==="
    )

    print(
        "Training:",
        len(X_train),
    )

    print(
        "Validation:",
        len(X_validation),
    )

    print(
        "Test:",
        len(X_test),
    )

    # ---------------------------------------------
    # Feature definitions
    # ---------------------------------------------

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

    # ---------------------------------------------
    # Logistic Regression
    # ---------------------------------------------

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
        _,
    ) = find_best_threshold(
        y_validation,
        logistic_validation_probabilities,
    )

    # ---------------------------------------------
    # Random Forest
    # Same training-only tuning as main experiment
    # ---------------------------------------------

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
        estimator=random_forest_pipeline,
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

    random_forest_validation_probabilities = (
        random_forest_pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    (
        random_forest_threshold,
        _,
    ) = find_best_threshold(
        y_validation,
        random_forest_validation_probabilities,
    )

    print(
        "\n=== VALIDATION-SELECTED THRESHOLDS ==="
    )

    print(
        "Logistic Regression:",
        round(
            logistic_threshold,
            2,
        ),
    )

    print(
        "Random Forest:",
        round(
            random_forest_threshold,
            2,
        ),
    )

    # ---------------------------------------------
    # Test predictions
    # ---------------------------------------------

    logistic_probabilities = (
        logistic_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    random_forest_probabilities = (
        random_forest_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    logistic_predictions = (
        logistic_probabilities
        >= logistic_threshold
    ).astype(int)

    random_forest_predictions = (
        random_forest_probabilities
        >= random_forest_threshold
    ).astype(int)

    y_test_array = y_test.to_numpy()

    logistic_test_cost = calculate_cost(
        y_test_array,
        logistic_predictions,
    )

    random_forest_test_cost = calculate_cost(
        y_test_array,
        random_forest_predictions,
    )

    observed_difference = (
        logistic_test_cost
        - random_forest_test_cost
    )

    print(
        "\n=== OBSERVED TEST COST ==="
    )

    print(
        "Logistic Regression:",
        logistic_test_cost,
    )

    print(
        "Random Forest:",
        random_forest_test_cost,
    )

    print(
        "LR - RF cost difference:",
        observed_difference,
    )

    # ---------------------------------------------
    # Paired bootstrap
    #
    # IMPORTANT:
    # Both models use the SAME sampled customer
    # indices in each bootstrap iteration.
    # ---------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    bootstrap_rows = []

    for bootstrap_id in range(
        BOOTSTRAP_SAMPLES
    ):
        indices = rng.integers(
            low=0,
            high=len(y_test_array),
            size=len(y_test_array),
        )

        sampled_y = (
            y_test_array[indices]
        )

        sampled_logistic = (
            logistic_predictions[indices]
        )

        sampled_random_forest = (
            random_forest_predictions[indices]
        )

        logistic_cost = calculate_cost(
            sampled_y,
            sampled_logistic,
        )

        random_forest_cost = (
            calculate_cost(
                sampled_y,
                sampled_random_forest,
            )
        )

        cost_difference = (
            logistic_cost
            - random_forest_cost
        )

        bootstrap_rows.append(
            {
                "bootstrap_sample":
                    bootstrap_id + 1,
                "logistic_cost":
                    int(logistic_cost),
                "random_forest_cost":
                    int(
                        random_forest_cost
                    ),
                "cost_difference_lr_minus_rf":
                    int(cost_difference),
            }
        )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    lower_bound = (
        bootstrap_df[
            "cost_difference_lr_minus_rf"
        ].quantile(0.025)
    )

    upper_bound = (
        bootstrap_df[
            "cost_difference_lr_minus_rf"
        ].quantile(0.975)
    )

    mean_difference = (
        bootstrap_df[
            "cost_difference_lr_minus_rf"
        ].mean()
    )

    median_difference = (
        bootstrap_df[
            "cost_difference_lr_minus_rf"
        ].median()
    )

    probability_lr_lower = (
        (
            bootstrap_df[
                "cost_difference_lr_minus_rf"
            ]
            < 0
        ).mean()
    )

    print(
        "\n=== PAIRED BOOTSTRAP COST COMPARISON ==="
    )

    print(
        "Bootstrap samples:",
        BOOTSTRAP_SAMPLES,
    )

    print(
        "Mean LR - RF difference:",
        round(
            mean_difference,
            2,
        ),
    )

    print(
        "Median LR - RF difference:",
        round(
            median_difference,
            2,
        ),
    )

    print(
        "95% CI:",
        f"[{lower_bound:.2f}, "
        f"{upper_bound:.2f}]",
    )

    print(
        "Bootstrap proportion where "
        "LR cost < RF cost:",
        f"{probability_lr_lower:.3f}",
    )

    # ---------------------------------------------
    # Interpretation
    # ---------------------------------------------

    if (
        lower_bound <= 0
        <= upper_bound
    ):
        conclusion = (
            "The 95% paired-bootstrap interval "
            "includes zero. The test data do not "
            "provide strong evidence that either "
            "model has consistently lower cost "
            "under this simulated cost scenario. "
            "Logistic Regression remains a "
            "reasonable deployment choice because "
            "it is simpler and more interpretable."
        )
    elif upper_bound < 0:
        conclusion = (
            "The paired-bootstrap interval is "
            "entirely below zero, supporting lower "
            "cost for Logistic Regression under "
            "this simulated scenario."
        )
    else:
        conclusion = (
            "The paired-bootstrap interval is "
            "entirely above zero, supporting lower "
            "cost for Random Forest under this "
            "simulated scenario."
        )

    print(
        "\nConclusion:"
    )

    print(
        conclusion
    )

    # ---------------------------------------------
    # Save evidence
    # ---------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "comparison":
                    "Logistic Regression vs Random Forest",
                "cost_difference_definition":
                    "LR cost - RF cost",
                "logistic_test_cost":
                    int(logistic_test_cost),
                "random_forest_test_cost":
                    int(random_forest_test_cost),
                "observed_cost_difference":
                    int(observed_difference),
                "bootstrap_samples":
                    BOOTSTRAP_SAMPLES,
                "mean_bootstrap_difference":
                    float(mean_difference),
                "median_bootstrap_difference":
                    float(median_difference),
                "ci_95_lower":
                    float(lower_bound),
                "ci_95_upper":
                    float(upper_bound),
                "proportion_lr_lower_cost":
                    float(
                        probability_lr_lower
                    ),
                "false_positive_cost":
                    FALSE_POSITIVE_COST,
                "false_negative_cost":
                    FALSE_NEGATIVE_COST,
                "conclusion":
                    conclusion,
            }
        ]
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        REPORT_PATH,
        index=False,
    )

    bootstrap_df.to_csv(
        BOOTSTRAP_SAMPLES_PATH,
        index=False,
    )

    print(
        "\nSaved:",
        REPORT_PATH,
    )

    print(
        "Saved:",
        BOOTSTRAP_SAMPLES_PATH,
    )


if __name__ == "__main__":
    main()