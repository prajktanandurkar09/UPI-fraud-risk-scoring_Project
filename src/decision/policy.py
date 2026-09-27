"""
Deployment-ready decision policy for UPI-SHIELD.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

THRESHOLD_RESULTS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "cost_sensitive_threshold_results.json"
)

POLICY_PATH = (
    PROJECT_ROOT
    / "models"
    / "decision_policy.json"
)


def load_optimal_threshold():
    """Load the threshold selected using validation data."""

    with open(
        THRESHOLD_RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    threshold = results[
        "threshold_selection"
    ][
        "optimal_threshold"
    ]

    return float(threshold)


def create_policy():
    """Create the deployment decision policy."""

    threshold = load_optimal_threshold()

    policy = {
        "policy_name": "UPI-SHIELD Cost-Aware Decision Policy",
        "version": "1.0",
        "threshold_source": "validation",
        "fraud_threshold": threshold,
        "default_threshold": 0.50,
        "decision_rule": {
            "if_probability_below_threshold": (
                "LEGITIMATE"
            ),
            "if_probability_at_or_above_threshold": (
                "FRAUD"
            ),
        },
    }

    POLICY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        POLICY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            policy,
            file,
            indent=4,
        )

    return policy


def decide(probability, threshold):
    """
    Convert fraud probability into a binary decision.
    """

    if probability >= threshold:
        return "FRAUD"

    return "LEGITIMATE"


def main():
    print("=" * 60)
    print("UPI-SHIELD DECISION POLICY")
    print("=" * 60)

    policy = create_policy()

    print()
    print(
        "Validation-selected threshold:"
    )
    print(
        f"{policy['fraud_threshold']:.2f}"
    )

    print()
    print("Decision examples:")

    examples = [
        0.02,
        0.05,
        0.09,
        0.15,
        0.50,
        0.90,
    ]

    for probability in examples:
        decision = decide(
            probability,
            policy["fraud_threshold"],
        )

        print(
            f"Probability {probability:.2f}"
            f" -> {decision}"
        )

    print()
    print("Policy saved to:")
    print(POLICY_PATH)

    print()
    print("PHASE 6 STEP 3 COMPLETE")


if __name__ == "__main__":
    main()