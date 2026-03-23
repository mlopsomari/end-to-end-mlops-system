from typing import Dict, Union, Tuple
import boto3
import pandas as pd
import os
import seaborn as sns
import logging
from shared.utils import configure_logging
import mlflow
from metaflow.exception import MetaflowException
from sklearn.model_selection import train_test_split

configure_logging()

def validate_dataset (data: pd.DataFrame) -> None:
    """
    Validate the `self.data` DataFrame before transforming it.

    Checks performed:
    1. `self.data` is a pandas DataFrame.
    2. `self.data` contains all required columns.
            """
    required_cols = {
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
    }

    # Validate DataFrame type
    logging.info("Validating: checking if self.data is a pandas DataFrame...")
    if not isinstance(data, pd.DataFrame):
        raise MetaflowException(
            "Validation failed: self.data is not a pandas DataFrame."
        )
    logging.info("✅ self.data is a valid pandas DataFrame.")

    # Validate required columns
    logging.info("Validating: checking required columns in DataFrame...")
    missing_columns = required_cols - set(data.columns)
    if missing_columns:
        raise MetaflowException(
            f"Validation failed: Missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )
    logging.info("✅ DataFrame contains all required columns.")


def split_data(penguins: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    """Create features and labels of train and test datasets"""

    # Create training and test splits
    penguins = penguins.dropna(subset=['species'])
    logging.info(msg="Creating features")
    X = penguins.drop('species', axis=1)
    logging.info(msg="Creating target")
    y = penguins['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    logging.info(msg="Test and train splits created")

    return X_train, X_test, y_train, y_test
