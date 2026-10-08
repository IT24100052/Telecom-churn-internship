from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


DATA_PATH = Path(
    "data/processed/telco_churn_clean.csv"
)

MODEL_PATH = Path(
    "models/telecom_churn_model.joblib"
)

PERMUTATION_REPORT_PATH = Path(
    "reports/logistic_permutation_importance.csv"
)

COEFFICIENT_REPORT_PATH = Path(
    "reports/logistic_coefficients.csv"
)

RANDOM_STATE = 42


def main():
    # -------------------------------------------------
    # Load data
    # -------------------------------------------------

    df = pd.read_csv(DATA_PATH)

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
    # Recreate the same 70 / 15 / 15 split
    #
    # Explainability is calculated on the validation
    # split, not used to alter the final test results.
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

    print(
        "=== EXPLAINABILITY DATA ==="
    )

    print(
        "Validation customers:",
        len(X_validation),
    )

    # -------------------------------------------------
    # Load deployed model
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
    model_name = bundle["model_name"]

    print(
        "Model:",
        model_name,
    )

    if model_name != "Logistic Regression":
        raise ValueError(
            "This explainability script is intended "
            "for the deployed Logistic Regression "
            "model."
        )

    # -------------------------------------------------
    # 1. Raw-feature permutation importance
    #
    # This measures the drop in validation ROC-AUC
    # when each original input feature is shuffled.
    # -------------------------------------------------

    permutation = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="roc_auc",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    permutation_df = pd.DataFrame(
        {
            "feature": X_validation.columns,
            "mean_auc_drop":
                permutation.importances_mean,
            "std_auc_drop":
                permutation.importances_std,
        }
    )

    permutation_df = (
        permutation_df
        .sort_values(
            "mean_auc_drop",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------
    # 2. Logistic Regression coefficients
    #
    # Coefficients describe the fitted model after
    # preprocessing.
    #
    # Positive coefficient:
    # associated with higher predicted churn odds.
    #
    # Negative coefficient:
    # associated with lower predicted churn odds.
    #
    # These are model associations, not causal effects.
    # -------------------------------------------------

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    coefficients = (
        classifier.coef_[0]
    )

    coefficient_df = pd.DataFrame(
        {
            "transformed_feature":
                feature_names,
            "coefficient":
                coefficients,
            "odds_multiplier":
                np.exp(coefficients),
            "absolute_coefficient":
                np.abs(coefficients),
        }
    )

    coefficient_df[
        "direction"
    ] = np.where(
        coefficient_df["coefficient"] > 0,
        "Higher predicted churn",
        "Lower predicted churn",
    )

    coefficient_df = (
        coefficient_df
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------
    # Display evidence
    # -------------------------------------------------

    print(
        "\n=== RAW-FEATURE PERMUTATION IMPORTANCE ==="
    )

    print(
        permutation_df.head(10).to_string(
            index=False
        )
    )

    print(
        "\n=== STRONGEST LOGISTIC COEFFICIENTS ==="
    )

    print(
        coefficient_df[
            [
                "transformed_feature",
                "coefficient",
                "odds_multiplier",
                "direction",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    # -------------------------------------------------
    # Save reports
    # -------------------------------------------------

    PERMUTATION_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    permutation_df.to_csv(
        PERMUTATION_REPORT_PATH,
        index=False,
    )

    coefficient_df.to_csv(
        COEFFICIENT_REPORT_PATH,
        index=False,
    )

    print(
        "\nSaved:",
        PERMUTATION_REPORT_PATH,
    )

    print(
        "Saved:",
        COEFFICIENT_REPORT_PATH,
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Permutation importance measures predictive "
        "dependence, not causality."
    )

    print(
        "Coefficient signs describe associations "
        "inside the fitted Logistic Regression model "
        "and should not be interpreted as causal "
        "effects."
    )


if __name__ == "__main__":
    main()