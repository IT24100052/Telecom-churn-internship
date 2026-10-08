from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

from src.predict_customer import predict_churn


app = FastAPI(
    title="Telecom Churn Prediction API",
    version="1.1.0",
)


YesNo = Literal["Yes", "No"]

InternetDependentService = Literal[
    "Yes",
    "No",
    "No internet service",
]


class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"]

    SeniorCitizen: Literal[0, 1]

    Partner: YesNo
    Dependents: YesNo

    tenure: int = Field(ge=0)

    PhoneService: YesNo

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service",
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]

    OnlineSecurity: InternetDependentService
    OnlineBackup: InternetDependentService
    DeviceProtection: InternetDependentService
    TechSupport: InternetDependentService
    StreamingTV: InternetDependentService
    StreamingMovies: InternetDependentService

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]

    PaperlessBilling: YesNo

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_service_consistency(self):
        # Phone service consistency
        if self.PhoneService == "No":
            if self.MultipleLines != "No phone service":
                raise ValueError(
                    "MultipleLines must be 'No phone service' "
                    "when PhoneService is 'No'."
                )

        if self.PhoneService == "Yes":
            if self.MultipleLines == "No phone service":
                raise ValueError(
                    "MultipleLines cannot be 'No phone service' "
                    "when PhoneService is 'Yes'."
                )

        internet_fields = [
            self.OnlineSecurity,
            self.OnlineBackup,
            self.DeviceProtection,
            self.TechSupport,
            self.StreamingTV,
            self.StreamingMovies,
        ]

        # Internet service consistency
        if self.InternetService == "No":
            if any(
                value != "No internet service"
                for value in internet_fields
            ):
                raise ValueError(
                    "Internet-related services must be "
                    "'No internet service' when InternetService is 'No'."
                )

        if self.InternetService != "No":
            if any(
                value == "No internet service"
                for value in internet_fields
            ):
                raise ValueError(
                    "'No internet service' is invalid when "
                    "InternetService is DSL or Fiber optic."
                )

        # Dataset-specific new-customer consistency check
        if self.tenure == 0 and self.TotalCharges != 0:
            raise ValueError(
                "TotalCharges must be 0 when tenure is 0 "
                "for this project's dataset convention."
            )

        return self


@app.get("/")
def root():
    return {
        "message": "Telecom Churn Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    customer_data = customer.model_dump()
    return predict_churn(customer_data)