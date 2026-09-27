"""
UPI-SHIELD prediction engine.

Connects:
    Transaction input
        ↓
    Feature engineering
        ↓
    Saved preprocessor
        ↓
    Saved XGBoost model
        ↓
    Fraud probability
        ↓
    Risk score
        ↓
    Cost-aware decision
        ↓
    SHAP explanation
"""

import json
from pathlib import Path

import pandas as pd
import joblib

from src.features.behavioral import (
    create_behavioral_features,
)
from src.features.pipeline import (
    SELECTED_FEATURES,
)
from src.risk_scoring.risk_score import (
    probability_to_risk_score,
    risk_band,
)
from src.explainability.explain_transaction import (
    generate_explanation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

PREPROCESSOR_PATH = (
    MODELS_DIR / "preprocessor.joblib"
)

MODEL_PATH = (
    MODELS_DIR / "xgboost_fraud_model.joblib"
)

POLICY_PATH = (
    MODELS_DIR / "decision_policy.json"
)

CONFIG_PATH = (
    MODELS_DIR / "risk_scoring_config.json"
)


# Load saved artifacts once when the API starts.
PREPROCESSOR = joblib.load(
    PREPROCESSOR_PATH
)

MODEL = joblib.load(
    MODEL_PATH
)

with open(
    POLICY_PATH,
    "r",
    encoding="utf-8",
) as file:
    POLICY = json.load(file)

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:
    CONFIG = json.load(file)


def prepare_transaction(
    transaction: dict,
) -> pd.DataFrame:
    """
    Prepare a new transaction for prediction.
    """

    dataframe = pd.DataFrame(
        [transaction]
    )

    dataframe = create_behavioral_features(
        dataframe
    )

    for feature in SELECTED_FEATURES:
        if feature not in dataframe.columns:
            dataframe[feature] = None

    dataframe = dataframe[
        SELECTED_FEATURES
    ]

    return dataframe


def predict_transaction(
    transaction: dict,
) -> dict:
    """
    Generate a complete UPI-SHIELD prediction.
    """

    dataframe = prepare_transaction(
        transaction
    )

    transformed = PREPROCESSOR.transform(
        dataframe
    )

    fraud_probability = float(
        MODEL.predict_proba(transformed)[0][1]
    )

    score = probability_to_risk_score(
        fraud_probability
    )

    band = risk_band(score)

    threshold = float(
        POLICY["fraud_threshold"]
    )

    if fraud_probability >= threshold:
        decision = "FRAUD"
    else:
        decision = "LEGITIMATE"

    explanations = generate_explanation(
        transaction,
        top_n=5,
    )

    return {
        "fraud_probability": round(
            fraud_probability,
            6,
        ),
        "risk_score": score,
        "risk_band": band,
        "decision_threshold": threshold,
        "decision": decision,
        "probability_source": CONFIG[
            "probability_source"
        ],
        "explanations": explanations,
    }


def main():
    """Test the prediction engine."""

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

    print("=" * 60)
    print("UPI-SHIELD API PREDICTION ENGINE")
    print("=" * 60)

    result = predict_transaction(
        transaction
    )

    print()
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
        f"{result['decision_threshold']}"
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
    print("Top explanations:")

    for explanation in result[
        "explanations"
    ]:
        print(
            f"{explanation['symbol']} "
            f"{explanation['feature']} "
            f"({explanation['shap_value']:+.6f})"
        )

    print()
    print("PHASE 11 STEP 2 COMPLETE")


if __name__ == "__main__":
    main()
