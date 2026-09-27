"""
UPI-SHIELD 0-100 Fraud Risk Scoring Engine.

Converts fraud probability into an interpretable
0-100 risk score.
"""


def probability_to_risk_score(
    probability: float,
) -> int:
    """
    Convert fraud probability to a 0-100 risk score.

    Parameters
    ----------
    probability : float
        Fraud probability in the range [0, 1].

    Returns
    -------
    int
        Risk score in the range [0, 100].
    """

    if probability < 0.0:
        probability = 0.0

    if probability > 1.0:
        probability = 1.0

    score = round(probability * 100)

    return int(score)


def risk_band(
    risk_score: int,
) -> str:
    """
    Convert a risk score into an interpretable risk band.

    Score ranges
    ------------
    0-24   : Low
    25-49  : Moderate
    50-74  : High
    75-100 : Critical
    """

    if risk_score < 25:
        return "LOW"

    if risk_score < 50:
        return "MODERATE"

    if risk_score < 75:
        return "HIGH"

    return "CRITICAL"


def generate_risk_result(
    probability: float,
) -> dict:
    """
    Generate a complete risk-scoring result.
    """

    score = probability_to_risk_score(
        probability
    )

    band = risk_band(score)

    return {
        "fraud_probability": round(
            probability,
            6,
        ),
        "risk_score": score,
        "risk_band": band,
    }


def main():
    """
    Demonstrate the risk-scoring engine.
    """

    print("=" * 60)
    print("UPI-SHIELD 0-100 RISK SCORING ENGINE")
    print("=" * 60)

    test_probabilities = [
        0.00,
        0.02,
        0.09,
        0.15,
        0.25,
        0.50,
        0.75,
        0.90,
        1.00,
    ]

    print()
    print(
        "Probability -> Risk Score -> Risk Band"
    )
    print("-" * 60)

    for probability in test_probabilities:
        result = generate_risk_result(
            probability
        )

        print(
            f"{result['fraud_probability']:.2f}"
            f" -> "
            f"{result['risk_score']:3d}"
            f" -> "
            f"{result['risk_band']}"
        )

    print()
    print("PHASE 8 STEP 1 COMPLETE")


if __name__ == "__main__":
    main()