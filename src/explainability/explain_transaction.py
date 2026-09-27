"""
UPI-SHIELD integrated transaction explainability.

Combines:
    Transaction
        ↓
    SHAP explanation
        ↓
    Human-readable factors
"""

from pathlib import Path

from src.explainability.shap_explainer import (
    explain_transaction,
)

from src.explainability.format_explanation import (
    format_explanation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_explanation(
    transaction: dict,
    top_n: int = 10,
) -> list:
    """
    Generate human-readable explanations
    for a transaction.
    """

    shap_results = explain_transaction(
        transaction,
        top_n=top_n,
    )

    explanations = format_explanation(
        shap_results
    )

    return explanations


def main():
    """Run integrated explainability."""

    print("=" * 60)
    print("UPI-SHIELD INTEGRATED EXPLAINABILITY")
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

    explanations = generate_explanation(
        transaction,
        top_n=10,
    )

    print()
    print("TOP RISK FACTORS")
    print("-" * 60)

    for index, explanation in enumerate(
        explanations,
        start=1,
    ):
        print(
            f"{index:2d}. "
            f"{explanation['message']}"
            f" "
            f"[SHAP: "
            f"{explanation['shap_value']:+.6f}]"
        )

    print()
    print(
        "PHASE 9 STEP 3 COMPLETE"
    )


if __name__ == "__main__":
    main()