from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict_customer import predict_churn


app = FastAPI(
    title="Telecom Churn Prediction API",
    version="1.0.0",
)


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


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

    result = predict_churn(customer_data)

    return result