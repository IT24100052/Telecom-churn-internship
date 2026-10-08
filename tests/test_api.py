from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)

MODEL_PATH = Path("models/telecom_churn_model.joblib")


def require_model():
    if not MODEL_PATH.exists():
        pytest.skip(
            "Model artifact not found. "
            "Run 'python -m src.business_model_comparison' first."
        )


VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
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


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_customer():
    require_model()

    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "churn_probability" in data

    assert data["prediction"] in [
        "Churn",
        "No Churn",
    ]

    assert (
        0
        <= data["churn_probability"]
        <= 1
    )


def test_predict_rejects_negative_tenure():
    require_model()

    customer = deepcopy(
        VALID_CUSTOMER
    )

    customer["tenure"] = -3

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 422


def test_predict_rejects_invalid_contract():
    require_model()

    customer = deepcopy(
        VALID_CUSTOMER
    )

    customer["Contract"] = "banana"

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 422


def test_predict_rejects_inconsistent_phone_service():
    require_model()

    customer = deepcopy(
        VALID_CUSTOMER
    )

    customer["PhoneService"] = "No"
    customer["MultipleLines"] = "Yes"

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 422


def test_predict_rejects_inconsistent_internet_service():
    require_model()

    customer = deepcopy(
        VALID_CUSTOMER
    )

    customer["InternetService"] = "No"
    customer["OnlineSecurity"] = "Yes"

    customer["OnlineBackup"] = (
        "No internet service"
    )

    customer["DeviceProtection"] = (
        "No internet service"
    )

    customer["TechSupport"] = (
        "No internet service"
    )

    customer["StreamingTV"] = (
        "No internet service"
    )

    customer["StreamingMovies"] = (
        "No internet service"
    )

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 422