"""
UPI-SHIELD SHAP-based explainability.

Provides feature-level explanations for
individual XGBoost fraud predictions.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.features.pipeline import engineer_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSOR_PATH = (
    PROJECT_ROOT / "models" / "preprocessor.joblib"
)

MODEL_PATH = (
    PROJECT_ROOT / "models" / "xgboost_fraud_model.joblib"
)


def load_artifacts():
    """Load the saved preprocessor and XGBoost model."""

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    return preprocessor, model


def explain_transaction(
    transaction: dict,
    top_n: int = 10,
) -> list:
    """
    Generate SHAP feature contributions
    for one transaction.

    Parameters
    ----------
    transaction : dict
        Transaction feature values.

    top_n : int
        Number of most influential features.

    Returns
    -------
    list
        Feature contribution records.
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

    x_data = dataframe[
        SELECTED_FEATURES
    ]

    preprocessor, model = load_artifacts()

    transformed_data = (
        preprocessor.transform(
            x_data
        )
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        transformed_data
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(
        shap_values
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    values = shap_values[0]

    feature_contributions = []

    for feature_name, contribution in zip(
        feature_names,
        values,
    ):
        feature_contributions.append(
            {
                "feature": feature_name,
                "shap_value": float(
                    contribution
                ),
                "direction": (
                    "increases_risk"
                    if contribution > 0
                    else "decreases_risk"
                ),
            }
        )

    feature_contributions.sort(
        key=lambda item: abs(
            item["shap_value"]
        ),
        reverse=True,
    )

    return feature_contributions[
        :top_n
    ]


def main():
    """Run SHAP explanation example."""

    print("=" * 60)
    print("UPI-SHIELD SHAP EXPLAINABILITY")
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

    print()
    print("Generating explanation...")

    explanations = explain_transaction(
        transaction,
        top_n=10,
    )

    print()
    print("TOP FEATURE CONTRIBUTIONS")
    print("-" * 60)

    for index, item in enumerate(
        explanations,
        start=1,
    ):
        print(
            f"{index:2d}. "
            f"{item['feature']:<45} "
            f"{item['shap_value']:+.6f} "
            f"({item['direction']})"
        )

    print()
    print(
        "PHASE 9 STEP 1 COMPLETE"
    )


if __name__ == "__main__":
    main()