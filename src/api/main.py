from fastapi import FastAPI
import joblib

from pydantic_models import (
    PredictionRequest,
    PredictionResponse
)

app = FastAPI(
    title="Credit Risk API"
)

# Load model
model = joblib.load("best_model.pkl")


@app.get("/")
def root():

    return {
        "message": "Credit Risk API Running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: PredictionRequest):

    features = [[
        data.TotalTransactionAmount,
        data.AverageTransactionAmount,
        data.TransactionCount,
        data.StdTransactionAmount
    ]]

    probability = model.predict_proba(
        features
    )[0][1]

    prediction = int(
        probability >= 0.5
    )

    return PredictionResponse(
        risk_probability=float(probability),
        prediction=prediction
    )