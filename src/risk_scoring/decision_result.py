"""
UPI-SHIELD integrated risk and decision engine.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.features.pipeline import engineer_features
from src.risk_scoring.risk_score import (
    generate_risk_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSOR_PATH = (
    PROJECT_ROOT / "models" / "preprocessor.joblib"
)

MODEL_PATH = (
    PROJECT_ROOT / "models" / "xgboost_fraud_model.joblib"
)

CONFIG_PATH = (
    PROJECT_ROOT / "models" / "risk_scoring_config.json"
)


def load_configuration():
    """Load deployment risk-scoring configuration."""

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_artifacts():
    """Load saved model artifacts."""

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    return preprocessor, model


def predict_transaction(
    transaction: dict,
) -> dict:
    """
    Generate probability, risk score,
    risk band, and cost-aware decision.
    """

    dataframe = pd.DataFrame(
        [transaction]
    )

    dataframe = engineer_features(
        dataframe
    )

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

    preprocessor, model = load_artifacts()

    transformed_data = (
        preprocessor.transform(
            x_data
        )
    )

    probability = float(
        model.predict_proba(
            transformed_data
        )[:, 1][0]
    )

    risk = generate_risk_result(
        probability
    )

    config = load_configuration()

    threshold = float(
        config["decision_threshold"]
    )

    if probability >= threshold:
        decision = "FRAUD"
    else:
        decision = "LEGITIMATE"

    result = {
        **risk,
        "decision_threshold": threshold,
        "decision": decision,
        "probability_source": config[
            "probability_source"
        ],
    }

    return result


def main():
    """Run integrated demonstration."""

    print("=" * 60)
    print("UPI-SHIELD INTEGRATED RISK DECISION ENGINE")
    print("=" * 60)

    transaction = {
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
    }

    result = predict_transaction(
        transaction
    )

    print()
    print("PREDICTION RESULT")
    print("-" * 40)

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

    print(
        f"Decision threshold: "
        f"{result['decision_threshold']:.2f}"
    )

    print(
        f"Decision: "
        f"{result['decision']}"
    )

    print(
        f"Probability source: "
        f"{result['probability_source']}"
    )

    print()
    print(
        "PHASE 8 STEP 4 COMPLETE"
    )


if __name__ == "__main__":
    main()