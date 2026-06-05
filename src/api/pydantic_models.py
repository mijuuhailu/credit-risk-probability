from pydantic import BaseModel


class PredictionRequest(BaseModel):

    TotalTransactionAmount: float
    AverageTransactionAmount: float
    TransactionCount: float
    StdTransactionAmount: float


class PredictionResponse(BaseModel):

    risk_probability: float
    prediction: int