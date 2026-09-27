"""
Cost sensitivity analysis for UPI-SHIELD.

Evaluates how the validation-optimal threshold changes
under different false-negative cost assumptions.
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
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.csv"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "preprocessor.joblib"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_fraud_model.joblib"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
)

THRESHOLDS = np.arange(
    0.01,
    1.00,
    0.01,
)

FALSE_POSITIVE_COST = 1.0

FALSE_NEGATIVE_COSTS = [
    5.0,
    10.0,
    20.0,
    50.0,
]


def calculate_cost(
    y_true,
    probabilities,
    threshold,
    false_positive_cost,
    false_negative_cost,
):
    """Calculate total decision cost."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    total_cost = (
        fp * false_positive_cost
        + fn * false_negative_cost
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

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "total_cost": float(total_cost),
    }


def main():
    print("=" * 60)
    print("UPI-SHIELD COST SENSITIVITY ANALYSIS")
    print("=" * 60)

    print()
    print("Loading validation dataset...")

    dataframe = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation rows: {len(dataframe):,}"
    )

    print("Engineering validation features...")

    dataframe = engineer_features(
        dataframe
    )

    x_validation, y_validation = prepare_xy(
        dataframe
    )

    print("Loading preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    print("Loading XGBoost model...")

    model = joblib.load(
        MODEL_PATH
    )

    print("Transforming validation data...")

    x_validation_transformed = (
        preprocessor.transform(
            x_validation
        )
    )

    print("Generating validation probabilities...")

    probabilities = model.predict_proba(
        x_validation_transformed
    )[:, 1]

    sensitivity_results = []

    print()
    print(
        "Evaluating cost scenarios..."
    )

    for false_negative_cost in (
        FALSE_NEGATIVE_COSTS
    ):

        scenario_results = []

        for threshold in THRESHOLDS:

            result = calculate_cost(
                y_validation,
                probabilities,
                threshold,
                FALSE_POSITIVE_COST,
                false_negative_cost,
            )

            scenario_results.append(
                result
            )

        optimal_result = min(
            scenario_results,
            key=lambda result:
            result["total_cost"],
        )

        sensitivity_results.append(
            {
                "false_positive_cost": (
                    FALSE_POSITIVE_COST
                ),
                "false_negative_cost": (
                    false_negative_cost
                ),
                "cost_ratio": (
                    f"1:{int(false_negative_cost)}"
                ),
                "optimal_threshold": (
                    optimal_result[
                        "threshold"
                    ]
                ),
                "total_cost": (
                    optimal_result[
                        "total_cost"
                    ]
                ),
                "precision": (
                    optimal_result[
                        "precision"
                    ]
                ),
                "recall": (
                    optimal_result[
                        "recall"
                    ]
                ),
                "tn": optimal_result["tn"],
                "fp": optimal_result["fp"],
                "fn": optimal_result["fn"],
                "tp": optimal_result["tp"],
            }
        )

    print()
    print("=" * 60)
    print("COST SENSITIVITY RESULTS")
    print("=" * 60)

    for result in sensitivity_results:

        print()
        print(
            f"Cost ratio: "
            f"{result['cost_ratio']}"
        )

        print(
            f"Optimal threshold: "
            f"{result['optimal_threshold']:.2f}"
        )

        print(
            f"Total cost: "
            f"{result['total_cost']:.2f}"
        )

        print(
            f"Precision: "
            f"{result['precision']:.6f}"
        )

        print(
            f"Recall: "
            f"{result['recall']:.6f}"
        )

    output = {
        "analysis": (
            "Validation threshold sensitivity "
            "under different false-negative costs."
        ),
        "false_positive_cost": (
            FALSE_POSITIVE_COST
        ),
        "false_negative_costs": (
            FALSE_NEGATIVE_COSTS
        ),
        "results": sensitivity_results,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "cost_sensitivity_results.json"
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
    print("PHASE 6 STEP 4 COMPLETE")


if __name__ == "__main__":
    main()