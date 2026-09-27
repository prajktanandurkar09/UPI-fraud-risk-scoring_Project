import math

import pandas as pd


def create_behavioral_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create transaction-level behavioral features.

    These features use information available in the
    transaction and do not use the fraud target.
    """

    result = df.copy()

    # Transaction amount features
    if "TransactionAmt" in result.columns:
        result["transaction_amount_log"] = (
            result["TransactionAmt"]
            .clip(lower=0)
            .add(1)
            .apply(math.log)
        )

        result["transaction_amount_decimal"] = (
            result["TransactionAmt"] % 1
        )

    # Email-domain consistency
    if {
        "P_emaildomain",
        "R_emaildomain",
    }.issubset(result.columns):

        result["email_domain_match"] = (
            result["P_emaildomain"]
            == result["R_emaildomain"]
        ).astype("int8")

    # Card information completeness
    card_columns = [
        column
        for column in result.columns
        if column.lower().startswith("card")
    ]

    if card_columns:
        result["card_information_missing_count"] = (
            result[card_columns]
            .isna()
            .sum(axis=1)
            .astype("int16")
        )

    # Address information completeness
    address_columns = [
        column
        for column in [
            "addr1",
            "addr2",
        ]
        if column in result.columns
    ]

    if address_columns:
        result["address_missing_count"] = (
            result[address_columns]
            .isna()
            .sum(axis=1)
            .astype("int8")
        )

    return result