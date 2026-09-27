"""
UPI-SHIELD end-to-end fraud risk prediction.

Pipeline:
Transaction data
    -> Feature engineering
    -> Saved preprocessor
    -> XGBoost model
    -> Fraud probability
    -> 0-100 risk score
    -> Risk band
"""

from pathlib import Path

import joblib
import pandas as pd

from src.features.pipeline import engineer_features
from src.risk_scoring.risk_score import (
    generate_risk_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "preprocessor.joblib"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_fraud_model.joblib"
)


def load_model_artifacts():
    """Load the saved preprocessing and model artifacts."""

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    return preprocessor, model


def predict_fraud_risk(
    transaction: dict,
) -> dict:
    """
    Generate fraud probability and risk score
    for a single transaction.

    Parameters
    ----------
    transaction : dict
        Transaction feature values.

    Returns
    -------
    dict
        Fraud probability, risk score,
        and risk band.
    """

    dataframe = pd.DataFrame(
        [transaction]
    )

    # Feature engineering does not require
    # the fraud target for new transactions.
    dataframe = engineer_features(
        dataframe
    )

    # Use the same selected features that were
    # used during model training.
    from src.features.pipeline import (
        SELECTED_FEATURES,
    )

    missing_features = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    x_data = dataframe[
        SELECTED_FEATURES
    ]

    preprocessor, model = (
        load_model_artifacts()
    )

    transformed_data = (
        preprocessor.transform(
            x_data
        )
    )

    probability = (
        model.predict_proba(
            transformed_data
        )[:, 1][0]
    )

    result = generate_risk_result(
        float(probability)
    )

    return result


def main():
    """Run sample risk predictions."""

    print("=" * 60)
    print("UPI-SHIELD END-TO-END RISK PREDICTION")
    print("=" * 60)

    sample_transactions = [
        {
            "TransactionDT": 86400,
            "TransactionAmt": 25.00,
            "ProductCD": "W",
            "card1": 10000,
            "card2": 111.0,
            "card3": 150.0,
            "card4": "visa",
            "card5": 226.0,
            "card6": "debit",
            "addr1": 100.0,
            "addr2": 87.0,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com",
            "DeviceType": "mobile",
            "DeviceInfo": "Android",
        },
        {
            "TransactionDT": 172800,
            "TransactionAmt": 950.00,
            "ProductCD": "C",
            "card1": 15000,
            "card2": 200.0,
            "card3": 185.0,
            "card4": "mastercard",
            "card5": 102.0,
            "card6": "credit",
            "addr1": 500.0,
            "addr2": 87.0,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "yahoo.com",
            "DeviceType": "desktop",
            "DeviceInfo": "Windows",
        },
    ]

    for index, transaction in enumerate(
        sample_transactions,
        start=1,
    ):
        print()
        print(
            f"Transaction {index}"
        )
        print("-" * 40)

        result = predict_fraud_risk(
            transaction
        )

        print(
            f"Fraud probability: "
            f"{result['fraud_probability']:.6f}"
        )

        print(
            f"Risk score: "
            f"{result['risk_score']}/100"
        )

        print(
            f"Risk band: "
            f"{result['risk_band']}"
        )

    print()
    print(
        "PHASE 8 STEP 2 COMPLETE"
    )


if __name__ == "__main__":
    main()