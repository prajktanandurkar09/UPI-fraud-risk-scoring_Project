from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "experiments" / "reports"


def create_temporal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create leakage-safe temporal features
    from TransactionDT.
    """

    if "TransactionDT" not in df.columns:
        raise ValueError(
            "TransactionDT column is required."
        )

    result = df.copy()

    transaction_time = result["TransactionDT"]

    # Dataset-relative time.
    result["transaction_day"] = (
        transaction_time // (24 * 60 * 60)
    )

    result["transaction_hour"] = (
        transaction_time // (60 * 60)
    ) % 24

    result["transaction_weekday"] = (
        result["transaction_day"] % 7
    )

    result["transaction_minute"] = (
        transaction_time // 60
    ) % 60

    # Cyclic encoding of hour.
    import numpy as np

    result["hour_sin"] = np.sin(
        2 * np.pi * result["transaction_hour"] / 24
    )

    result["hour_cos"] = np.cos(
        2 * np.pi * result["transaction_hour"] / 24
    )

    return result


def generate_feature_report(
    df: pd.DataFrame,
) -> None:
    """Generate a report describing dataset columns."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORT_DIR / "feature_inventory.txt"
    )

    lines = []

    lines.append(
        "UPI-SHIELD FEATURE INVENTORY"
    )
    lines.append(
        "=" * 60
    )
    lines.append("")
    lines.append(
        f"Rows: {len(df):,}"
    )
    lines.append(
        f"Columns: {len(df.columns):,}"
    )
    lines.append("")

    lines.append(
        "COLUMN LIST"
    )
    lines.append(
        "-" * 60
    )

    for index, column in enumerate(
        df.columns,
        start=1,
    ):
        dtype = str(df[column].dtype)
        missing = int(
            df[column].isna().sum()
        )
        unique = int(
            df[column].nunique(
                dropna=True
            )
        )

        lines.append(
            f"{index:03d}. "
            f"{column} | "
            f"dtype={dtype} | "
            f"unique={unique:,} | "
            f"missing={missing:,}"
        )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"Feature inventory saved to:\n"
        f"{report_path}"
    )


def main() -> None:
    """Inspect the training dataset."""

    train_path = (
        PROCESSED_DIR / "train.csv"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found:\n"
            f"{train_path}"
        )

    print("Loading training data...")

    train_df = pd.read_csv(
        train_path
    )

    print(
        f"Rows: {len(train_df):,}"
    )

    print(
        f"Columns: {len(train_df.columns):,}"
    )

    print(
        "\nCreating temporal features..."
    )

    featured_df = create_temporal_features(
        train_df
    )

    print(
        "Temporal features created:"
    )

    temporal_columns = [
        "transaction_day",
        "transaction_hour",
        "transaction_weekday",
        "transaction_minute",
        "hour_sin",
        "hour_cos",
    ]

    for column in temporal_columns:
        print(f"  [OK] {column}")

    generate_feature_report(
        featured_df
    )

    print(
        "\nPHASE 3 TEMPORAL FEATURE INSPECTION COMPLETE"
    )


if __name__ == "__main__":
    main()