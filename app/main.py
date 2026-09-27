"""
UPI-SHIELD FastAPI application.
"""

from fastapi import FastAPI

from app.predictor import predict_transaction
from app.schemas import (
    PredictionResponse,
    TransactionRequest,
)


app = FastAPI(
    title="UPI-SHIELD API",
    description=(
        "Adaptive, Cost-Aware and Explainable "
        "Fraud Risk Scoring Framework"
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    """API health endpoint."""

    return {
        "project": "UPI-SHIELD",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    """Health check endpoint."""

    return {
        "status": "healthy",
        "model": "XGBoost",
        "probability_source": "raw_xgboost",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    transaction: TransactionRequest,
):
    """Predict fraud risk for a transaction."""

    result = predict_transaction(
        transaction.model_dump()
    )

    return result