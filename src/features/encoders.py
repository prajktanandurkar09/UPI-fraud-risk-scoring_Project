from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"


def build_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build the leakage-safe preprocessing pipeline.

    The preprocessor must be fitted ONLY on training data.
    """

    # Identify numeric columns.
    numeric_columns = (
        X_train
        .select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    # Identify categorical columns.
    categorical_columns = (
        X_train
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    # TransactionID is an identifier and must not
    # be used as a predictive feature.
    numeric_columns = [
        column
        for column in numeric_columns
        if column != "TransactionID"
    ]

    categorical_columns = [
        column
        for column in categorical_columns
        if column != "TransactionID"
    ]

    # Numeric preprocessing.
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    # Categorical preprocessing.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                ),
            ),
        ]
    )

    # Combine both pipelines.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def save_preprocessor(
    preprocessor: ColumnTransformer,
) -> None:
    """Save the fitted preprocessing pipeline."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        MODEL_DIR / "preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        path,
    )

    print(
        f"Preprocessor saved to:\n{path}"
    )