"""
Independent evaluation of probability calibration.

The calibrator was fitted on validation data.
The test dataset is used only for final evaluation.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.features.pipeline import engineer_features, prepare_xy


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
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

CALIBRATOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "probability_calibrator.joblib"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "calibration_test_results.json"
)


def calculate_ece(
    y_true,
    probabilities,
    n_bins=10,
):
    """Calculate Expected Calibration Error."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        if not np.any(mask):
            continue

        observed_rate = np.mean(
            y_true[mask]
        )

        mean_probability = np.mean(
            probabilities[mask]
        )

        bin_weight = np.mean(mask)

        ece += (
            abs(
                observed_rate
                - mean_probability
            )
            * bin_weight
        )

    return float(ece)


def main():
    print("=" * 60)
    print("UPI-SHIELD INDEPENDENT CALIBRATION EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------

    print()
    print("Loading test dataset...")

    dataframe = pd.read_csv(
        TEST_PATH
    )

    print(
        f"Test rows: {len(dataframe):,}"
    )

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------

    print(
        "Engineering test features..."
    )

    dataframe = engineer_features(
        dataframe
    )

    x_test, y_test = prepare_xy(
        dataframe
    )

    # --------------------------------------------------
    # LOAD SAVED ARTIFACTS
    # --------------------------------------------------

    print(
        "Loading preprocessor..."
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    print(
        "Loading XGBoost model..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Loading saved calibrator..."
    )

    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    # --------------------------------------------------
    # TRANSFORM TEST DATA
    # --------------------------------------------------

    print(
        "Transforming test features..."
    )

    x_test_transformed = (
        preprocessor.transform(
            x_test
        )
    )

    # --------------------------------------------------
    # RAW PROBABILITIES
    # --------------------------------------------------

    print(
        "Generating raw XGBoost probabilities..."
    )

    raw_probabilities = (
        model.predict_proba(
            x_test_transformed
        )[:, 1]
    )

    # --------------------------------------------------
    # CALIBRATED PROBABILITIES
    # --------------------------------------------------

    print(
        "Applying saved calibrator..."
    )

    calibrated_probabilities = (
        calibrator.predict(
            raw_probabilities
        )
    )

    y_test = y_test.to_numpy()

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    raw_brier = brier_score_loss(
        y_test,
        raw_probabilities,
    )

    calibrated_brier = brier_score_loss(
        y_test,
        calibrated_probabilities,
    )

    raw_ece = calculate_ece(
        y_test,
        raw_probabilities,
    )

    calibrated_ece = calculate_ece(
        y_test,
        calibrated_probabilities,
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("TEST CALIBRATION RESULTS")
    print("=" * 60)

    print()
    print("RAW XGBOOST")
    print(
        f"Brier Score: {raw_brier:.6f}"
    )
    print(
        f"ECE:         {raw_ece:.6f}"
    )

    print()
    print("CALIBRATED XGBOOST")
    print(
        f"Brier Score: {calibrated_brier:.6f}"
    )
    print(
        f"ECE:         {calibrated_ece:.6f}"
    )

    print()
    print("Brier improvement:")
    print(
        f"{raw_brier - calibrated_brier:.6f}"
    )

    print()
    print("ECE improvement:")
    print(
        f"{raw_ece - calibrated_ece:.6f}"
    )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    results = {
        "evaluation_dataset": "test",
        "calibrator_fit_dataset": "validation",
        "test_used_for_calibrator_fitting": False,
        "calibration_method": (
            "IsotonicRegression"
        ),
        "raw": {
            "brier_score": raw_brier,
            "ece": raw_ece,
        },
        "calibrated": {
            "brier_score": calibrated_brier,
            "ece": calibrated_ece,
        },
        "improvement": {
            "brier_score": (
                raw_brier
                - calibrated_brier
            ),
            "ece": (
                raw_ece
                - calibrated_ece
            ),
        },
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print()
    print(
        "Results saved to:"
    )
    print(
        RESULTS_PATH
    )

    print()
    print(
        "PHASE 7 STEP 2 COMPLETE"
    )


if __name__ == "__main__":
    main()