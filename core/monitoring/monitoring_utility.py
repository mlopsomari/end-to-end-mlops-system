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

    Parameters
    ----------
    None

    Returns
    -------
    ev_ref_data : Dataset
        The reference Evidently AI dataset created from the historical penguin
        data (penguins.csv). The 'species' column is renamed to 'ground_truth'
        and duplicated as 'classification' to match the monitoring schema.
    ev_curr_data : Dataset
        The current Evidently AI dataset created from the 100 most recent
        inference records retrieved from the database.

    Notes
    -----
    - Defines a schema with the following structure:
        * Numerical columns: bill_length_mm, bill_depth_mm, flipper_length_mm,
          body_mass_g
        * Categorical columns: classification, ground_truth, island, sex
    - Reference data is loaded from 'data/penguins.csv'
    - Current data retrieves the 100 most recent records from the database
    - Reference dataset has 'species' renamed to 'ground_truth' and copied to
      'classification' for consistency with inference data format
    - Both datasets use the same data definition schema for comparability
    - Logs the dataset creation process

    Examples
    --------
    ref_dataset, curr_dataset = create_datasets()
    print(ref_dataset.data.columns.tolist())
    ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g',
     'classification', 'ground_truth', 'island', 'sex']

    See Also
    --------
    retrieve_data : Retrieves current inference data from the database
    Dataset.from_pandas : Creates Evidently AI datasets from pandas DataFrames
    DataDefinition : Defines the schema for Evidently AI datasets
    """

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(
        name="penguin_classifier",
        alias="deployed"
    )

    logging.info("Retrieving latest validated URI...")

    model_uri = client.get_model_version_download_uri(
        name=model_version.name,
        version=model_version.version
    )

    dataset_uri = f"{model_uri}/artifacts/dataset.csv"

    with tempfile.TemporaryDirectory() as tmp_dir:
        logging.info("Downloading training data to temp dir: %s", tmp_dir)

        try:
            # Attempt to download the artifact
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=dataset_uri, dst_path=tmp_dir)

            # Verify if the file actually exists before reading
            data = pd.read_csv(local_path)
            logging.info("Training data successfully downloaded and loaded.")

        except MlflowException as e:
            logging.error("MLflow connection or artifact error: %s", e)
            # Handle specifically: e.g., retry logic or raising a custom error
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
    # create a classification column to match monitoring schema
    reference_dataset["classification"] = reference_dataset["ground_truth"]
    # retrieve inference data from database
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
    against reference data, stores the results in the workspace, and adds a
    visualization panel to the project dashboard.

    Parameters
    ----------
    ev_ref_data : Dataset
        The reference Evidently AI dataset used as the baseline for drift
        detection. Typically contains historical or training data.
    ev_curr_data : Dataset
        The current Evidently AI dataset containing recent inference data
        to be compared against the reference data.
    ws : Workspace
        The Evidently AI workspace instance where the report will be stored.
    project : Project
        The Evidently AI project instance where the dashboard panel will be
        added and configuration saved.

    Returns
    -------
    None
        This function does not return a value. Results are stored in the
        workspace and project configuration is updated.

    Notes
    -----
    - Creates a data drift report with a threshold of 0.1 (10%)
    - Compares current data against reference data to detect distribution shifts
    - Adds the evaluation results to the workspace under the specified project
    - Creates a dashboard panel showing row count metrics over time:
        * Title: "Row count"
        * Visualization type: Counter plot
        * Aggregation: Sum
        * Panel size: Half width
    - Saves the updated project configuration with the new dashboard panel
    - Logs the report generation process

    Examples
    --------
    ref_data, curr_data = create_datasets()
    workspace, project = utility()
    run_report(ref_data, curr_data, workspace, project)
    # Generates drift report and updates dashboard

    See Also
    --------
    create_datasets : Creates the reference and current datasets for reporting
    utility : Initializes the workspace and project
    Report : Evidently AI report class for generating analyses
    DataDriftPreset : Preset configuration for data drift detection
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

    #Add the evaluation to the current project

    return my_eval