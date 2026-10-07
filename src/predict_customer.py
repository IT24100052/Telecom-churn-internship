import joblib
import pandas as pd


MODEL_PATH = "models/telecom_churn_model.joblib"


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


model = joblib.load(MODEL_PATH)


def predict_churn(customer_data):
    """
    Predict churn for one customer.
    """

    # -----------------------------
    # Check required fields
    # -----------------------------

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in customer_data
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {missing_fields}"
        )


    # -----------------------------
    # Validate numeric inputs
    # -----------------------------

    if customer_data["tenure"] < 0:
        raise ValueError(
            "tenure cannot be negative"
        )

    if customer_data["SeniorCitizen"] not in [0, 1]:
        raise ValueError(
            "SeniorCitizen must be 0 or 1"
        )

    if customer_data["MonthlyCharges"] < 0:
        raise ValueError(
            "MonthlyCharges cannot be negative"
        )

    if customer_data["TotalCharges"] < 0:
        raise ValueError(
            "TotalCharges cannot be negative"
        )


    # -----------------------------
    # Convert input to DataFrame
    # -----------------------------

    customer_df = pd.DataFrame(
        [customer_data]
    )


    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(
        customer_df
    )[0]

    probability = model.predict_proba(
        customer_df
    )[0][1]


    return {
        "prediction": (
            "Churn"
            if prediction == 1
            else "No Churn"
        ),
        "churn_probability": round(
            float(probability),
            4
        ),
    }


# --------------------------------------------------
# Example validation test
# --------------------------------------------------

if __name__ == "__main__":

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",

        # Intentionally invalid for testing
        "tenure": 6,

        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.90,
        "TotalCharges": 540.00,
    }

    result = predict_churn(
        customer
    )

    print(
        "=== CUSTOMER CHURN PREDICTION ==="
    )

    print(
        "Prediction:",
        result["prediction"]
    )

    print(
        "Churn probability:",
        round(
            result["churn_probability"] * 100,
            2
        ),
        "%"
    )