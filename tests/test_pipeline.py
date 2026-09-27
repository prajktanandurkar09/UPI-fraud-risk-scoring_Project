"""
UPI-SHIELD automated pipeline tests.

These tests verify:
1. Required model artifacts exist.
2. Risk scoring stays within 0-100.
3. Risk bands are assigned correctly.
4. Decision threshold is valid.
5. Required configuration files are valid.
"""

import json
from pathlib import Path

from src.risk_scoring.risk_score import (
    probability_to_risk_score,
    risk_band,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"


def test_required_model_artifacts_exist():
    """Verify required trained artifacts exist."""

    required_files = [
        "preprocessor.joblib",
        "xgboost_fraud_model.joblib",
        "decision_policy.json",
        "risk_scoring_config.json",
    ]

    for filename in required_files:
        filepath = MODELS_DIR / filename

        assert filepath.exists(), (
            f"Missing required artifact: {filename}"
        )


def test_risk_score_range():
    """Verify risk score stays between 0 and 100."""

    probabilities = [
        0.0,
        0.02,
        0.09,
        0.25,
        0.50,
        0.75,
        0.90,
        1.0,
    ]

    for probability in probabilities:
        score = probability_to_risk_score(
            probability
        )

        assert 0 <= score <= 100


def test_risk_bands():
    """Verify defined risk-band boundaries."""

    assert risk_band(0) == "LOW"
    assert risk_band(24) == "LOW"

    assert risk_band(25) == "MODERATE"
    assert risk_band(49) == "MODERATE"

    assert risk_band(50) == "HIGH"
    assert risk_band(74) == "HIGH"

    assert risk_band(75) == "CRITICAL"
    assert risk_band(100) == "CRITICAL"


def test_decision_threshold():
    """Verify deployment decision threshold is valid."""

    policy_file = (
        MODELS_DIR / "decision_policy.json"
    )

    with open(
        policy_file,
        "r",
        encoding="utf-8",
    ) as file:
        policy = json.load(file)

    threshold = policy["fraud_threshold"]

    assert 0 < threshold < 1


def test_risk_scoring_configuration():
    """Verify risk scoring configuration."""

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

    assert (
        config["risk_bands"]["LOW"]["min_score"]
        == 0
    )

    assert (
        config["risk_bands"]["CRITICAL"]["max_score"]
        == 100
    )
