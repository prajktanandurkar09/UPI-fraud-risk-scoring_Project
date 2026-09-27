from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

from src.features.pipeline import (
    engineer_features,
    prepare_xy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


def load_data(filename: str) -> pd.DataFrame:
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
    """Create Phase 3 features."""

    featured_df = engineer_features(df)

    X, y = prepare_xy(featured_df)

    return X, y


def main() -> None:
    """Train the advanced XGBoost fraud model."""

    print("\nLoading datasets...")

    train_df = load_data("train.csv")
    validation_df = load_data("validation.csv")

    print(
        f"Train rows:      {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(validation_df):,}"
    )

    print("\nEngineering features...")

    X_train, y_train = prepare_features(
        train_df
    )

    X_validation, y_validation = (
        prepare_features(validation_df)
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
        "\nLoading preprocessing pipeline..."
    )

    preprocessor = joblib.load(
        preprocessor_path
    )

    print(
        "\nTransforming training data..."
    )

    X_train_transformed = (
        preprocessor.transform(X_train)
    )

    print(
        "Transforming validation data..."
    )

    X_validation_transformed = (
        preprocessor.transform(
            X_validation
        )
    )

    print("\nTraining XGBoost...")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train_transformed,
        y_train,
        eval_set=[
            (
                X_validation_transformed,
                y_validation,
            )
        ],
        verbose=50,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR / "xgboost_fraud_model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nXGBoost model saved to:\n"
        f"{model_path}"
    )

    print(
        "\nPHASE 5 TRAINING COMPLETE"
    )


if __name__ == "__main__":
    main()