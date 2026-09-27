"""
Validation-based cost-sensitive threshold optimization.

UPI-SHIELD
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.features.pipeline import engineer_features, prepare_xy


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "validation.csv"
)

TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test.csv"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT / "models" / "preprocessor.joblib"
)

MODEL_PATH = (
    PROJECT_ROOT / "models" / "xgboost_fraud_model.joblib"
)

RESULTS_DIR = (
    PROJECT_ROOT / "experiments" / "results"
)

# Framework-level initial cost assumptions.
FALSE_POSITIVE_COST = 1.0
FALSE_NEGATIVE_COST = 10.0

THRESHOLDS = np.arange(0.01, 1.00, 0.01)


def calculate_metrics(y_true, probabilities, threshold):
    """
    Calculate confusion matrix and cost at a threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    false_positive_cost = (
        fp * FALSE_POSITIVE_COST
    )

    false_negative_cost = (
        fn * FALSE_NEGATIVE_COST
    )

    total_cost = (
        false_positive_cost
        + false_negative_cost
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_cost": float(
            false_positive_cost
        ),
        "false_negative_cost": float(
            false_negative_cost
        ),
        "total_cost": float(total_cost),
    }


def prepare_probabilities(
    csv_path,
    preprocessor,
    model,
    dataset_name,
):
    """
    Load a dataset, engineer features and generate
    model probabilities.
    """

    print()
    print(f"Loading {dataset_name} dataset...")

    dataframe = pd.read_csv(csv_path)

    print(
        f"{dataset_name} rows: {len(dataframe):,}"
    )

    print(
        f"Preparing {dataset_name} features..."
    )

    dataframe = engineer_features(dataframe)

    x, y = prepare_xy(dataframe)

    print(
        f"Transforming {dataset_name} features..."
    )

    x_transformed = preprocessor.transform(x)

    print(
        f"Generating {dataset_name} probabilities..."
    )

    probabilities = model.predict_proba(
        x_transformed
    )[:, 1]

    return y.to_numpy(), probabilities


def main():
    print("=" * 60)
    print("UPI-SHIELD COST-AWARE THRESHOLD OPTIMIZATION")
    print("=" * 60)

    print()
    print("Loading preprocessor...")
    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    print("Loading XGBoost model...")
    model = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    y_validation, validation_probabilities = (
        prepare_probabilities(
            VALIDATION_PATH,
            preprocessor,
            model,
            "Validation",
        )
    )

    print()
    print("Searching validation thresholds...")

    validation_results = []

    for threshold in THRESHOLDS:
        result = calculate_metrics(
            y_validation,
            validation_probabilities,
            threshold,
        )

        validation_results.append(result)

    optimal_validation = min(
        validation_results,
        key=lambda result: result["total_cost"],
    )

    # --------------------------------------------------
    # TEST
    # --------------------------------------------------

    y_test, test_probabilities = (
        prepare_probabilities(
            TEST_PATH,
            preprocessor,
            model,
            "Test",
        )
    )

    test_result = calculate_metrics(
        y_test,
        test_probabilities,
        optimal_validation["threshold"],
    )

    # --------------------------------------------------
    # BASELINE POLICY
    # --------------------------------------------------

    baseline_validation = calculate_metrics(
        y_validation,
        validation_probabilities,
        0.50,
    )

    baseline_test = calculate_metrics(
        y_test,
        test_probabilities,
        0.50,
    )

    validation_cost_savings = (
        baseline_validation["total_cost"]
        - optimal_validation["total_cost"]
    )

    test_cost_savings = (
        baseline_test["total_cost"]
        - test_result["total_cost"]
    )

    validation_savings_percentage = (
        validation_cost_savings
        / baseline_validation["total_cost"]
        * 100
        if baseline_validation["total_cost"] > 0
        else 0.0
    )

    test_savings_percentage = (
        test_cost_savings
        / baseline_test["total_cost"]
        * 100
        if baseline_test["total_cost"] > 0
        else 0.0
    )

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("COST-AWARE VALIDATION RESULT")
    print("=" * 60)

    print(
        f"Optimal threshold: "
        f"{optimal_validation['threshold']:.2f}"
    )

    print(
        f"Validation total cost: "
        f"{optimal_validation['total_cost']:.2f}"
    )

    print(
        f"Validation precision: "
        f"{optimal_validation['precision']:.6f}"
    )

    print(
        f"Validation recall: "
        f"{optimal_validation['recall']:.6f}"
    )

    print(
        f"Validation F1: "
        f"{optimal_validation['f1']:.6f}"
    )

    print()
    print(
        f"Validation cost savings vs 0.50: "
        f"{validation_cost_savings:.2f}"
    )

    print(
        f"Validation savings percentage: "
        f"{validation_savings_percentage:.2f}%"
    )

    print()
    print("=" * 60)
    print("TEST RESULT USING VALIDATION-SELECTED THRESHOLD")
    print("=" * 60)

    print(
        f"Test threshold: "
        f"{test_result['threshold']:.2f}"
    )

    print(
        f"Test total cost: "
        f"{test_result['total_cost']:.2f}"
    )

    print(
        f"Test precision: "
        f"{test_result['precision']:.6f}"
    )

    print(
        f"Test recall: "
        f"{test_result['recall']:.6f}"
    )

    print(
        f"Test F1: "
        f"{test_result['f1']:.6f}"
    )

    print()
    print(
        f"Test cost savings vs 0.50: "
        f"{test_cost_savings:.2f}"
    )

    print(
        f"Test savings percentage: "
        f"{test_savings_percentage:.2f}%"
    )

    print()
    print("Validation confusion matrix:")

    print(
        f"TN={optimal_validation['tn']}, "
        f"FP={optimal_validation['fp']}, "
        f"FN={optimal_validation['fn']}, "
        f"TP={optimal_validation['tp']}"
    )

    print()
    print("Test confusion matrix:")

    print(
        f"TN={test_result['tn']}, "
        f"FP={test_result['fp']}, "
        f"FN={test_result['fn']}, "
        f"TP={test_result['tp']}"
    )

    output = {
        "cost_assumptions": {
            "false_positive_cost": (
                FALSE_POSITIVE_COST
            ),
            "false_negative_cost": (
                FALSE_NEGATIVE_COST
            ),
        },
        "threshold_selection": {
            "selection_dataset": "validation",
            "baseline_threshold": 0.50,
            "optimal_threshold": (
                optimal_validation["threshold"]
            ),
        },
        "validation": {
            "optimal_result": optimal_validation,
            "baseline_result": baseline_validation,
            "cost_savings": (
                validation_cost_savings
            ),
            "savings_percentage": (
                validation_savings_percentage
            ),
        },
        "test": {
            "result_at_validation_threshold": (
                test_result
            ),
            "baseline_result": baseline_test,
            "cost_savings": test_cost_savings,
            "savings_percentage": (
                test_savings_percentage
            ),
        },
        "all_validation_thresholds": (
            validation_results
        ),
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "cost_sensitive_threshold_results.json"
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

    print()
    print("Results saved to:")
    print(output_path)

    print()
    print("PHASE 6 STEP 2 COMPLETE")


if __name__ == "__main__":
    main()