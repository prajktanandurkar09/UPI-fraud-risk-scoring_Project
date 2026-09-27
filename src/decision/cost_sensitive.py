"""
Cost-aware fraud decision system for UPI-SHIELD.

This module evaluates fraud probabilities under different decision
thresholds and selects an operating threshold using validation data.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_fraud_model.joblib"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

THRESHOLD_START = 0.01
THRESHOLD_END = 0.99
THRESHOLD_STEP = 0.01

# Initial cost assumptions.
# These are framework parameters, not measured real-world UPI costs.
FALSE_POSITIVE_COST = 1.0
FALSE_NEGATIVE_COST = 10.0


def calculate_cost(y_true, probabilities, threshold):
    """
    Calculate total decision cost at a given threshold.
    """

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    total_cost = (
        fp * FALSE_POSITIVE_COST
        + fn * FALSE_NEGATIVE_COST
    )

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_cost": float(fp * FALSE_POSITIVE_COST),
        "false_negative_cost": float(fn * FALSE_NEGATIVE_COST),
        "total_cost": float(total_cost),
    }


def find_optimal_threshold(y_true, probabilities):
    """
    Find the threshold that minimizes total validation cost.
    """

    thresholds = np.arange(
        THRESHOLD_START,
        THRESHOLD_END + THRESHOLD_STEP,
        THRESHOLD_STEP,
    )

    results = []

    for threshold in thresholds:
        result = calculate_cost(
            y_true,
            probabilities,
            threshold,
        )
        results.append(result)

    optimal_result = min(
        results,
        key=lambda item: item["total_cost"],
    )

    return optimal_result, results


def save_results(optimal_result, all_results):
    """
    Save cost-aware threshold analysis.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "false_positive_cost": FALSE_POSITIVE_COST,
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "threshold_search": {
            "start": THRESHOLD_START,
            "end": THRESHOLD_END,
            "step": THRESHOLD_STEP,
        },
        "optimal_validation_result": optimal_result,
        "all_threshold_results": all_results,
    }

    output_path = (
        RESULTS_DIR / "cost_sensitive_threshold_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
        )

    return output_path


def main():
    print("Loading XGBoost model...")

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model does not support probability prediction."
        )

    print("Model loaded successfully.")

    print()
    print("IMPORTANT:")
    print(
        "This module defines framework-level cost assumptions."
    )
    print(
        "They are not claimed to represent actual UPI operational costs."
    )
    print()

    print(
        f"False Positive Cost: {FALSE_POSITIVE_COST}"
    )
    print(
        f"False Negative Cost: {FALSE_NEGATIVE_COST}"
    )

    print()
    print(
        "PHASE 6 COMPONENT CREATED"
    )
    print(
        "Threshold optimization will use validation probabilities."
    )

    print()
    print(
        "Next step: connect this module to the existing validation"
    )
    print(
        "feature pipeline and evaluate the thresholds."
    )


if __name__ == "__main__":
    main()
