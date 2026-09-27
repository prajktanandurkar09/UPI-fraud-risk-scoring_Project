from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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


def load_data(
    filename: str,
) -> pd.DataFrame:
    """Load a processed dataset."""

    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    return pd.read_csv(path)


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Recreate Phase 3 feature engineering and
    preprocessing for one split.
    """

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
    """Calculate baseline evaluation metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
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

    return metrics


def main() -> None:
    """Train and evaluate baseline Logistic Regression."""

    print("\nLoading datasets...")

    train_df = load_data("train.csv")
    validation_df = load_data("validation.csv")
    test_df = load_data("test.csv")

    print(
        f"Train:      {len(train_df):,} rows"
    )

    print(
        f"Validation: {len(validation_df):,} rows"
    )

    print(
        f"Test:       {len(test_df):,} rows"
    )

    print("\nPreparing features...")

    X_train, y_train = prepare_features(
        train_df
    )

    X_validation, y_validation = (
        prepare_features(validation_df)
    )

    X_test, y_test = prepare_features(
        test_df
    )

    preprocessor_path = (
        MODEL_DIR / "preprocessor.joblib"
    )

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            "preprocessor.joblib not found. "
            "Complete Phase 3 first."
        )

    print(
        "\nLoading Phase 3 preprocessor..."
    )

    preprocessor = joblib.load(
        preprocessor_path
    )

    print(
        "\nTransforming training features..."
    )

    X_train_transformed = (
        preprocessor.transform(X_train)
    )

    print(
        "Transforming validation features..."
    )

    X_validation_transformed = (
        preprocessor.transform(
            X_validation
        )
    )

    print(
        "Transforming test features..."
    )

    X_test_transformed = (
        preprocessor.transform(X_test)
    )

    print(
        "\nTraining Logistic Regression..."
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    model.fit(
        X_train_transformed,
        y_train,
    )

    print(
        "\nGenerating validation predictions..."
    )

    validation_probabilities = (
        model.predict_proba(
            X_validation_transformed
        )[:, 1]
    )

    print(
        "Generating test predictions..."
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

    print("\nVALIDATION RESULTS")
    print("=" * 50)

    for name, value in validation_metrics.items():
        print(
            f"{name}: {value:.6f}"
        )

    print("\nTEST RESULTS")
    print("=" * 50)

    for name, value in test_metrics.items():
        print(
            f"{name}: {value:.6f}"
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR / "baseline_logistic_regression.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    results = {
        "model": "Logistic Regression",
        "threshold": 0.50,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    results_path = (
        RESULTS_DIR
        / "baseline_results.json"
    )

    import json

    results_path.write_text(
        json.dumps(
            results,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        f"\nBaseline model saved to:\n"
        f"{model_path}"
    )

    print(
        f"\nResults saved to:\n"
        f"{results_path}"
    )

    print(
        "\nPHASE 4 BASELINE COMPLETE"
    )


if __name__ == "__main__":
    main()