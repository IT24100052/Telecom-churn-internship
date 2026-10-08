from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "telecom_churn_model.joblib"
)


REQUIRED_FIELDS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


@lru_cache(maxsize=1)
def load_model_bundle():
    """
    Load the selected model, operating threshold,
    and model metadata.

    The result is cached so the model artifact
    is loaded only once per Python process.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file was not found at "
            f"{MODEL_PATH}. "
            "Run "
            "'python -m src.business_model_comparison' "
            "first to train and save the model."
        )

    bundle = joblib.load(
        MODEL_PATH
    )

    required_bundle_keys = [
        "model",
        "threshold",
        "model_name",
    ]

    missing_keys = [
        key
        for key in required_bundle_keys
        if key not in bundle
    ]

    if missing_keys:
        raise ValueError(
            "Saved model artifact is using "
            "an older format. Missing keys: "
            f"{missing_keys}. "
            "Run "
            "'python -m src.business_model_comparison' "
            "to regenerate it."
        )

    return bundle


def validate_customer_data(
    customer_data,
):
    """
    Perform basic model-level validation.

    More detailed categorical validation
    is also performed by FastAPI/Pydantic.
    """

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in customer_data
    ]

    if missing_fields:
        raise ValueError(
            "Missing required fields: "
            f"{missing_fields}"
        )

    if customer_data["tenure"] < 0:
        raise ValueError(
            "tenure cannot be negative"
        )

    if customer_data[
        "SeniorCitizen"
    ] not in [0, 1]:
        raise ValueError(
            "SeniorCitizen must be 0 or 1"
        )

    if (
        customer_data[
            "MonthlyCharges"
        ]
        < 0
    ):
        raise ValueError(
            "MonthlyCharges cannot be negative"
        )

    if (
        customer_data[
            "TotalCharges"
        ]
        < 0
    ):
        raise ValueError(
            "TotalCharges cannot be negative"
        )

    if (
        customer_data["tenure"] == 0
        and customer_data[
            "TotalCharges"
        ] != 0
    ):
        raise ValueError(
            "TotalCharges must be 0 "
            "when tenure is 0 for this "
            "project's dataset convention."
        )


def predict_churn(
    customer_data,
):
    """
    Predict customer churn using the
    validation-selected model and threshold.
    """

    validate_customer_data(
        customer_data
    )

    bundle = load_model_bundle()

    model = bundle["model"]

    decision_threshold = float(
        bundle["threshold"]
    )

    model_name = bundle[
        "model_name"
    ]

    customer_df = pd.DataFrame(
        [customer_data]
    )

    churn_probability = (
        model.predict_proba(
            customer_df
        )[0][1]
    )

    prediction = int(
        churn_probability
        >= decision_threshold
    )

    prediction_label = (
        "Churn"
        if prediction == 1
        else "No Churn"
    )

    return {
        "prediction":
        prediction_label,

        "churn_probability":
        round(
            float(
                churn_probability
            ),
            4,
        ),

        "decision_threshold":
        round(
            decision_threshold,
            4,
        ),

        "model":
        model_name,
    }


if __name__ == "__main__":

    example_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 6,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService":
        "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract":
        "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod":
        "Electronic check",
        "MonthlyCharges": 89.90,
        "TotalCharges": 540.00,
    }

    result = predict_churn(
        example_customer
    )

    print(
        "=== CUSTOMER CHURN "
        "PREDICTION ==="
    )

    print(
        "Model:",
        result["model"],
    )

    print(
        "Decision threshold:",
        result[
            "decision_threshold"
        ],
    )

    print(
        "Prediction:",
        result["prediction"],
    )

    print(
        "Churn probability:",
        result[
            "churn_probability"
        ] * 100,
        "%",
    )