from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


TARGET_COLUMN = "isFraud"


def load_split(filename: str) -> pd.DataFrame:
    """Load a processed dataset split."""

    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    return pd.read_csv(path)


def prepare_features(
    df: pd.DataFrame,
):
    """Create the same features used during training."""

    from src.features.pipeline import (
        engineer_features,
        prepare_xy,
    )

    featured_df = engineer_features(df)

    X, y = prepare_xy(featured_df)

    return X, y


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict:
    """Calculate fraud detection metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
    }


def main() -> None:
    """Evaluate XGBoost on validation and test data."""

    print("\nLoading datasets...")

    validation_df = load_split(
        "validation.csv"
    )

    test_df = load_split(
        "test.csv"
    )

    print(
        f"Validation rows: {len(validation_df):,}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    print("\nPreparing validation features...")

    X_validation, y_validation = (
        prepare_features(validation_df)
    )

    print("Preparing test features...")

    X_test, y_test = prepare_features(
        test_df
    )

    preprocessor_path = (
        MODEL_DIR / "preprocessor.joblib"
    )

    model_path = (
        MODEL_DIR / "xgboost_fraud_model.joblib"
    )

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            "preprocessor.joblib not found."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            "xgboost_fraud_model.joblib not found."
        )

    print("\nLoading preprocessor...")

    preprocessor = joblib.load(
        preprocessor_path
    )

    print("Loading XGBoost model...")

    model = joblib.load(
        model_path
    )

    print(
        "\nTransforming validation data..."
    )

    X_validation_transformed = (
        preprocessor.transform(
            X_validation
        )
    )

    print(
        "Transforming test data..."
    )

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        "\nGenerating validation probabilities..."
    )

    validation_probabilities = (
        model.predict_proba(
            X_validation_transformed
        )[:, 1]
    )

    print(
        "Generating test probabilities..."
    )

    test_probabilities = (
        model.predict_proba(
            X_test_transformed
        )[:, 1]
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
    )

    print("\nXGBOOST VALIDATION RESULTS")
    print("=" * 55)

    for name, value in validation_metrics.items():
        print(
            f"{name}: {value:.6f}"
        )

    print("\nXGBOOST TEST RESULTS")
    print("=" * 55)

    for name, value in test_metrics.items():
        print(
            f"{name}: {value:.6f}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "model": "XGBoost",
        "threshold": 0.50,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    import json

    results_path = (
        RESULTS_DIR
        / "xgboost_results.json"
    )

    results_path.write_text(
        json.dumps(
            results,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        f"\nResults saved to:\n"
        f"{results_path}"
    )

    print(
        "\nPHASE 5 EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()