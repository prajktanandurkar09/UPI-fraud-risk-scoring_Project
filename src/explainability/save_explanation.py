"""
UPI-SHIELD explanation report generator.

Generates a structured JSON explanation report
for a sample transaction.
"""

import json
from pathlib import Path

from src.explainability.explain_transaction import (
    generate_explanation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

OUTPUT_FILE = RESULTS_DIR / "explanation_report.json"


def main():
    """Generate and save explanation report."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    explanations = generate_explanation(
        transaction,
        top_n=10,
    )

    report = {
        "project": "UPI-SHIELD",
        "explanation_method": "SHAP TreeExplainer",
        "interpretation": (
            "SHAP values represent model contributions "
            "toward or away from the fraud-risk prediction. "
            "They are not causal explanations."
        ),
        "transaction": transaction,
        "top_risk_factors": explanations,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print("=" * 60)
    print("UPI-SHIELD EXPLANATION REPORT")
    print("=" * 60)

    print()
    print("Explanation method:")
    print("SHAP TreeExplainer")

    print()
    print("Top factors saved:")
    print(len(explanations))

    print()
    print("Report saved to:")
    print(OUTPUT_FILE)

    print()
    print("PHASE 9 STEP 4 COMPLETE")


if __name__ == "__main__":
    main()
