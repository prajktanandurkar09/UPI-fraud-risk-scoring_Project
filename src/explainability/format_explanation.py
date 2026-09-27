"""
UPI-SHIELD human-readable explanation formatter.
"""


def clean_feature_name(
    feature_name: str,
) -> str:
    """Convert technical feature names into readable names."""

    name = feature_name

    prefixes = [
        "numeric__",
        "categorical__",
    ]

    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]

    replacements = {
        "TransactionAmt": "Transaction amount",
        "TransactionDT": "Transaction time",
        "transaction_amount_log": (
            "Log-transformed transaction amount"
        ),
        "transaction_amount_decimal": (
            "Transaction amount decimal component"
        ),
        "transaction_day": "Transaction day",
        "transaction_hour": "Transaction hour",
        "transaction_weekday": "Transaction weekday",
        "transaction_minute": "Transaction minute",
        "email_domain_match": (
            "Email domain relationship"
        ),
        "card_information_missing_count": (
            "Missing card information"
        ),
        "address_missing_count": (
            "Missing address information"
        ),
        "ProductCD": "Product category",
        "card1": "Card identifier 1",
        "card2": "Card identifier 2",
        "card3": "Card identifier 3",
        "card4": "Card type",
        "card5": "Card identifier 5",
        "card6": "Card category",
        "addr1": "Address region",
        "addr2": "Address region 2",
        "P_emaildomain": "Purchaser email domain",
        "R_emaildomain": "Recipient email domain",
        "DeviceType": "Device type",
        "DeviceInfo": "Device information",
    }

    if name in replacements:
        return replacements[name]

    return name.replace("_", " ").title()


def format_explanation(
    feature_contributions: list,
) -> list:
    """
    Convert SHAP feature contributions into
    human-readable explanation records.
    """

    explanations = []

    for item in feature_contributions:
        shap_value = item["shap_value"]

        if shap_value > 0:
            direction = "increases_risk"
            symbol = "↑"
            message = "increased the model's risk prediction"
        else:
            direction = "decreases_risk"
            symbol = "↓"
            message = "decreased the model's risk prediction"

        feature = clean_feature_name(
            item["feature"]
        )

        explanations.append(
            {
                "feature": feature,
                "technical_feature": item["feature"],
                "shap_value": shap_value,
                "direction": direction,
                "symbol": symbol,
                "message": (
                    f"{symbol} {feature} "
                    f"{message}"
                ),
            }
        )

    return explanations


def main():
    """Demonstrate explanation formatting."""

    sample = [
        {
            "feature": "numeric__TransactionAmt",
            "shap_value": -0.635175,
            "direction": "decreases_risk",
        },
        {
            "feature": "numeric__email_domain_match",
            "shap_value": 0.430792,
            "direction": "increases_risk",
        },
        {
            "feature": "categorical__DeviceType_desktop",
            "shap_value": 0.143518,
            "direction": "increases_risk",
        },
    ]

    explanations = format_explanation(
        sample
    )

    print("=" * 60)
    print("UPI-SHIELD HUMAN-READABLE EXPLANATIONS")
    print("=" * 60)

    print()

    for explanation in explanations:
        print(
            explanation["message"]
        )

    print()
    print(
        "PHASE 9 STEP 2 COMPLETE"
    )


if __name__ == "__main__":
    main()