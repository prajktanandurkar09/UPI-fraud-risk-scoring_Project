"""
UPI-SHIELD data split and leakage tests.

Verifies:
1. Processed datasets exist.
2. TransactionID does not overlap across splits.
3. TransactionDT remains chronological.
4. Target column is present.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)


def load_split(filename):
    """Load a processed dataset."""

    filepath = PROCESSED_DIR / filename

    assert filepath.exists(), (
        f"Missing processed dataset: {filename}"
    )

    return pd.read_csv(filepath)


def test_processed_datasets_exist():
    """Verify all chronological splits exist."""

    required_files = [
        "train.csv",
        "validation.csv",
        "test.csv",
    ]

    for filename in required_files:
        filepath = PROCESSED_DIR / filename

        assert filepath.exists(), (
            f"Missing file: {filename}"
        )


def test_transaction_id_no_overlap():
    """Verify TransactionID does not overlap."""

    train = load_split("train.csv")
    validation = load_split("validation.csv")
    test = load_split("test.csv")

    train_ids = set(train["TransactionID"])
    validation_ids = set(validation["TransactionID"])
    test_ids = set(test["TransactionID"])

    assert train_ids.isdisjoint(
        validation_ids
    )

    assert train_ids.isdisjoint(
        test_ids
    )

    assert validation_ids.isdisjoint(
        test_ids
    )


def test_temporal_order():
    """Verify chronological ordering of splits."""

    train = load_split("train.csv")
    validation = load_split("validation.csv")
    test = load_split("test.csv")

    train_max = train["TransactionDT"].max()
    validation_min = validation["TransactionDT"].min()

    validation_max = validation["TransactionDT"].max()
    test_min = test["TransactionDT"].min()

    assert train_max <= validation_min

    assert validation_max <= test_min


def test_target_column_exists():
    """Verify isFraud exists in every split."""

    for filename in [
        "train.csv",
        "validation.csv",
        "test.csv",
    ]:
        data = load_split(filename)

        assert "isFraud" in data.columns