from pathlib import Path

import joblib
import pandas as pd

from src.features.behavioral import create_behavioral_features
from src.features.encoders import build_preprocessor
from src.features.temporal import create_temporal_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


TARGET_COLUMN = "isFraud"


# Controlled feature set.
# TransactionID is intentionally excluded.
SELECTED_FEATURES = [
    "TransactionDT",
    "TransactionAmt",

    "transaction_day",
    "transaction_hour",
    "transaction_weekday",
    "transaction_minute",
    "hour_sin",
    "hour_cos",

    "transaction_amount_log",
    "transaction_amount_decimal",

    "email_domain_match",
    "card_information_missing_count",
    "address_missing_count",

    "ProductCD",

    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",

    "addr1",
    "addr2",

    "P_emaildomain",
    "R_emaildomain",

    "DeviceType",
    "DeviceInfo",
]


def load_split(filename: str) -> pd.DataFrame:
    """Load a processed dataset split."""

    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found:\n{path}"
        )

    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal and behavioral features."""

    result = create_temporal_features(df)

    result = create_behavioral_features(result)

    return result


def select_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Select only approved predictive features."""

    available_features = [
        column
        for column in SELECTED_FEATURES
        if column in df.columns
    ]

    return df[available_features].copy()


def prepare_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictive features and target."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    y = df[TARGET_COLUMN].copy()

    X = select_features(df)

    return X, y


def main() -> None:
    """Run the complete feature engineering pipeline."""

    print("\nLoading datasets...")

    train_df = load_split("train.csv")
    validation_df = load_split("validation.csv")
    test_df = load_split("test.csv")

    print(f"Train:      {len(train_df):,} rows")
    print(f"Validation: {len(validation_df):,} rows")
    print(f"Test:       {len(test_df):,} rows")

    print("\nEngineering training features...")
    train_featured = engineer_features(train_df)

    print("Engineering validation features...")
    validation_featured = engineer_features(validation_df)

    print("Engineering test features...")
    test_featured = engineer_features(test_df)

    X_train, y_train = prepare_xy(train_featured)
    X_validation, y_validation = prepare_xy(
        validation_featured
    )
    X_test, y_test = prepare_xy(test_featured)

    print("\nSelected feature count:")
    print(f"X_train columns: {X_train.shape[1]}")

    print("\nSelected features:")
    for column in X_train.columns:
        print(f"  [OK] {column}")

    # Explicit leakage check.
    if "TransactionID" in X_train.columns:
        raise RuntimeError(
            "LEAKAGE ERROR: TransactionID found in features."
        )

    if TARGET_COLUMN in X_train.columns:
        raise RuntimeError(
            "LEAKAGE ERROR: isFraud found in features."
        )

    print("\nLeakage checks:")
    print("  [PASS] TransactionID excluded")
    print("  [PASS] isFraud excluded")

    print(
        "\nBuilding preprocessing pipeline..."
    )

    preprocessor = build_preprocessor(X_train)

    print(
        "\nFitting preprocessing ONLY on training data..."
    )

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    print("Transforming validation data...")

    X_validation_transformed = (
        preprocessor.transform(
            X_validation
        )
    )

    print("Transforming test data...")

    X_test_transformed = preprocessor.transform(
        X_test
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    preprocessor_path = (
        MODEL_DIR / "preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    print(
        f"\nPreprocessor saved to:\n"
        f"{preprocessor_path}"
    )

    print("\nTRANSFORMED DATA")
    print("=" * 50)

    print(
        "Training matrix:",
        X_train_transformed.shape,
    )

    print(
        "Validation matrix:",
        X_validation_transformed.shape,
    )

    print(
        "Test matrix:",
        X_test_transformed.shape,
    )

    print("\nTarget distribution:")

    print(
        f"Train fraud:      {int(y_train.sum()):,}"
    )

    print(
        f"Validation fraud: {int(y_validation.sum()):,}"
    )

    print(
        f"Test fraud:       {int(y_test.sum()):,}"
    )

    print("\nPHASE 3 COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()