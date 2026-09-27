"""
Probability calibration for UPI-SHIELD.

The calibrator is fitted using the validation dataset only.
The test dataset is reserved for final evaluation.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

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

CALIBRATOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "probability_calibrator.joblib"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "calibration_results.json"
)


def calculate_ece(
    y_true,
    probabilities,
    n_bins=10,
):
    """
    Calculate Expected Calibration Error (ECE).

    The probability range [0, 1] is divided into equal-width bins.
    """

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

        bin_accuracy = np.mean(
            y_true[mask]
        )

        bin_confidence = np.mean(
            probabilities[mask]
        )

        bin_fraction = np.mean(mask)

        ece += (
            abs(
                bin_accuracy
                - bin_confidence
            )
            * bin_fraction
        )

    return float(ece)


def main():
    print("=" * 60)
    print("UPI-SHIELD PROBABILITY CALIBRATION")
    print("=" * 60)

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    print()
    print("Loading validation dataset...")

    dataframe = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation rows: {len(dataframe):,}"
    )

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------

    print(
        "Engineering validation features..."
    )

    dataframe = engineer_features(
        dataframe
    )

    x_validation, y_validation = prepare_xy(
        dataframe
    )

    # --------------------------------------------------
    # LOAD PREPROCESSOR + MODEL
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

    # --------------------------------------------------
    # TRANSFORM DATA
    # --------------------------------------------------

    print(
        "Transforming validation features..."
    )

    x_validation_transformed = (
        preprocessor.transform(
            x_validation
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
            x_validation_transformed
        )[:, 1]
    )

    y_validation = y_validation.to_numpy()

    # --------------------------------------------------
    # RAW CALIBRATION METRICS
    # --------------------------------------------------

    raw_brier = brier_score_loss(
        y_validation,
        raw_probabilities,
    )

    raw_ece = calculate_ece(
        y_validation,
        raw_probabilities,
    )

    print()
    print(
        "RAW XGBOOST CALIBRATION"
    )
    print("=" * 60)

    print(
        f"Brier Score: {raw_brier:.6f}"
    )

    print(
        f"ECE:         {raw_ece:.6f}"
    )

    # --------------------------------------------------
    # FIT ISOTONIC CALIBRATOR
    # --------------------------------------------------

    print()
    print(
        "Fitting isotonic calibration model..."
    )

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        raw_probabilities,
        y_validation,
    )

    # --------------------------------------------------
    # CALIBRATED PROBABILITIES
    # --------------------------------------------------

    print(
        "Generating calibrated probabilities..."
    )

    calibrated_probabilities = (
        calibrator.predict(
            raw_probabilities
        )
    )

    # --------------------------------------------------
    # CALIBRATION METRICS
    # --------------------------------------------------

    calibrated_brier = brier_score_loss(
        y_validation,
        calibrated_probabilities,
    )

    calibrated_ece = calculate_ece(
        y_validation,
        calibrated_probabilities,
    )

    print()
    print(
        "CALIBRATED VALIDATION RESULTS"
    )
    print("=" * 60)

    print(
        f"Brier Score: {calibrated_brier:.6f}"
    )

    print(
        f"ECE:         {calibrated_ece:.6f}"
    )

    # --------------------------------------------------
    # SAVE CALIBRATOR
    # --------------------------------------------------

    CALIBRATOR_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        calibrator,
        CALIBRATOR_PATH,
    )

    print()
    print(
        "Calibrator saved to:"
    )

    print(
        CALIBRATOR_PATH
    )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    results = {
        "calibration_method": (
            "IsotonicRegression"
        ),
        "fit_dataset": "validation",
        "test_dataset_used": False,
        "raw_model": "XGBoost",
        "raw_brier_score": raw_brier,
        "raw_ece": raw_ece,
        "calibrated_brier_score": (
            calibrated_brier
        ),
        "calibrated_ece": calibrated_ece,
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
        "Calibration results saved to:"
    )

    print(
        RESULTS_PATH
    )

    print()
    print(
        "PHASE 7 STEP 1 COMPLETE"
    )


if __name__ == "__main__":
    main()
