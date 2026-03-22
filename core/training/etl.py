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


def download_penguins() -> str | None:
    """
    Download the penguins dataset and store it in 'continuous_training/data'.

    Returns:
        str: Absolute file path to the saved CSV if successful or already present.
        None: If the download/save process failed.
    """





    # Determine project root relative to this file
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    data_dir = os.path.join(project_root, "data")
    file_path = os.path.join(data_dir, "penguins.csv")

    logging.info("Checking for existing dataset in %s", data_dir)

    # If dataset already exists, return immediately
    if os.path.exists(file_path):
        logging.info("Dataset already exists at %s", file_path)
        return file_path

    try:
        logging.info("Downloading the penguins dataset...")
        penguins = sns.load_dataset("penguins")

        if penguins is None or penguins.empty:
            raise ValueError("Downloaded dataset is empty or invalid.")

        os.makedirs(data_dir, exist_ok=True)

        logging.info("Saving dataset to %s", file_path)
        penguins.to_csv(file_path, index=False)

        logging.info("Download and save successful!")
        return file_path # ✅ return the full path here

    except Exception as e:
        logging.error("Failed to download or save penguins dataset: %s", e, exc_info=True)

        # Optional fallback: use existing data if available
        if os.path.exists(file_path):
            logging.warning("Using previously saved dataset at %s", file_path)
            return file_path

        logging.critical("No valid dataset available.")
        return None  # ❌ return None explicitly if it failed


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
