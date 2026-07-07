from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

app = FastAPI()

# Load model
try:
    model = joblib.load("loan_model.joblib")
except Exception:
    model = None


# Request Schema
class LoanRequest(BaseModel):
    Age: int = Field(
        ..., ge=18, le=60,
        description="Age of the applicant must be between 18 and 60"
    )
    Salary: float = Field(
        ..., gt=10000,
        description="Salary of the applicant must be greater than 10000"
    )


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Loan Prediction API is Running"
    }


# Health endpoint
@app.get("/health")
def health():
    if model is None:
        return {
            "status": "Unhealthy",
            "message": "Model is not loaded"
        }

    return {
        "status": "Healthy",
        "message": "Loaded"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: LoanRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    if data.Salary > 1000000:
        raise HTTPException(
            status_code=400,
            detail="Salary is too high"
        )

    try:
        input_data = [[data.Age, data.Salary]]
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result = "Loan Approved"
        else:
            result = "Loan Rejected"

        return {
            "prediction": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
                   