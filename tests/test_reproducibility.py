"""
UPI-SHIELD reproducibility verification tests.

Verifies that the artifacts and experiment results
required to reproduce the project are available.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = (
    PROJECT_ROOT / "experiments" / "results"
)


def test_required_models_exist():
    """Verify all required model artifacts exist."""

    required_models = [
        "preprocessor.joblib",
        "xgboost_fraud_model.joblib",
        "baseline_logistic_regression.joblib",
        "probability_calibrator.joblib",
    ]

    for filename in required_models:
        assert (
            MODELS_DIR / filename
        ).exists(), (
            f"Missing model artifact: {filename}"
        )


def test_required_configuration_exists():
    """Verify deployment configuration files exist."""

    required_configs = [
        "decision_policy.json",
        "risk_scoring_config.json",
    ]

    for filename in required_configs:
        assert (
            MODELS_DIR / filename
        ).exists(), (
            f"Missing configuration: {filename}"
        )


def test_experiment_results_exist():
    """Verify important experiment result files exist."""

    required_results = [
        "baseline_results.json",
        "xgboost_results.json",
        "cost_sensitive_threshold_results.json",
        "cost_sensitivity_results.json",
        "calibration_results.json",
        "calibration_test_results.json",
        "explanation_report.json",
    ]

    for filename in required_results:
        assert (
            RESULTS_DIR / filename
        ).exists(), (
            f"Missing experiment result: {filename}"
        )


def test_result_files_are_valid_json():
    """Verify experiment result files contain valid JSON."""

    json_files = [
        "baseline_results.json",
        "xgboost_results.json",
        "cost_sensitive_threshold_results.json",
        "cost_sensitivity_results.json",
        "calibration_results.json",
        "calibration_test_results.json",
        "explanation_report.json",
    ]

    for filename in json_files:

        filepath = RESULTS_DIR / filename

        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        assert isinstance(data, dict)


def test_risk_configuration_is_reproducible():
    """Verify the saved risk configuration."""

    config_file = (
        MODELS_DIR / "risk_scoring_config.json"
    )

    with open(
        config_file,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    assert config["probability_source"] == (
        "raw_xgboost"
    )

    assert config["risk_score_formula"] == (
        "probability * 100"
    )

    assert config["decision_threshold"] == 0.09

    assert config["false_positive_cost"] == 1.0

    assert config["false_negative_cost"] == 10.0