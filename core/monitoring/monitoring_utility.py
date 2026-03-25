from shared.utils import configure_logging
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import *
import logging
import mlflow
import pandas as pd
from typing import Tuple, Any
from evidently.ui.workspace import Project
from evidently import MulticlassClassification
import sqlalchemy
import tempfile
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient


configure_logging()


def get_evidently_html(evidently_object) -> str:
    """Returns the rendered EvidentlyAI report/metric as HTML

    Should be assigned to `self.html`, installing `metaflow-card-html` to be rendered
    """
    import tempfile

    with tempfile.NamedTemporaryFile() as tmp:
        evidently_object.save_html(tmp.name)
        with open(tmp.name) as fh:
            return fh.read()


def retrieve_data(data_collection_uri: str) -> pd.DataFrame:
    """Retrieve data inference data from RDS PostgreSQL database"""
    engine = None
    connection = None

    try:
        logging.info("Creating database engine...")
        engine = sqlalchemy.create_engine(data_collection_uri)

        logging.info("Retrieving all data...")

        query = sqlalchemy.text("""
            SELECT island, sex, bill_length_mm, bill_depth_mm, flipper_length_mm,
                   body_mass_g, classification, ground_truth
            FROM data
            ORDER BY date DESC;
        """)

        connection = engine.connect()
        data = pd.read_sql_query(query, connection)

        logging.info("Data retrieved successfully.")
        return data

    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
        raise

    finally:
        if connection is not None:
            connection.close()
            logging.info("Database connection closed.")

        if engine is not None:
            engine.dispose()
            logging.info("Engine disposed.")


def create_datasets(current_dataset: pd.DataFrame) -> Tuple[Dataset, Dataset]:
    """
    Create reference and current Evidently AI datasets for model monitoring.

    This function prepares two datasets for monitoring: a reference dataset from
    historical penguin data and a current dataset from recent inference results.
    Both datasets are configured with the same schema defining numerical and
    categorical features.
    """

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(
        name="penguin_classifier",
        alias="deployed"
    )

    logging.info("Retrieving latest validated URI...")

    # retrieve the MLFlow model uri for the deployed model
    model_uri = client.get_model_version_download_uri(
        name=model_version.name,
        version=model_version.version
    )

    # Use the model URI to define training dataset artifact URI
    dataset_uri = f"{model_uri}/artifacts/dataset.csv"

    with tempfile.TemporaryDirectory() as tmp_dir:
        logging.info("Downloading training data to temp dir: %s", tmp_dir)

        try:
            # Attempt to download the artifact
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=dataset_uri, dst_path=tmp_dir)

            data = pd.read_csv(local_path)
            logging.info("Training data successfully downloaded and loaded.")

        except MlflowException as e:
            logging.error("MLflow connection or artifact error: %s", e)
            raise

        except FileNotFoundError:
            logging.error("The file 'train.csv' was not found in the run artifacts.")
            raise

        except Exception as e:
            logging.error("An unexpected error occurred during data retrieval: %s", e)
            raise


    reference_dataset = data

    logging.info("Creating datasets")
    # Create a monitoring schema for the dataset
    schema = DataDefinition(
        numerical_columns=[
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ],
        categorical_columns=[
            "classification",
            "ground_truth",
            "island",
            "sex",
        ],
        classification=[MulticlassClassification(
            target="ground_truth",
            prediction_labels="classification",
        )]
    )

    # rename the species column as ground_truth for consistency with inference
    # data
    reference_dataset.rename(columns={"species":"ground_truth"}, inplace=True)
    # create a classification column to match evidently monitoring schema
    reference_dataset["classification"] = reference_dataset["ground_truth"]
    # create dataset object with the historical data
    ev_ref_data = Dataset.from_pandas(
        reference_dataset,
        data_definition=schema
    )
    # Create a dataset object with the inference data
    ev_curr_data = Dataset.from_pandas(
        current_dataset,
        data_definition=schema
    )

    # return a tuple of the historical and inference dataset objects
    return ev_ref_data, ev_curr_data


def run_report(ev_ref_data, ev_curr_data) -> Any:
    """
    Generate and store a data drift report in the Evidently AI workspace.

    This function creates data drift and classification reports comparing current inference data
    against reference data
    """
    logging.info("Running report...")
    # Create a report with pre-built evaluation templates for drift
    # and classifications tasks
    drift_report = Report([
    DataDriftPreset(
    threshold=0.1,
    drift_share=0.1,
        ),
    ClassificationPreset()]
    )
    # Execute the evaluations
    my_eval = drift_report.run(ev_curr_data, ev_ref_data)

    return my_eval