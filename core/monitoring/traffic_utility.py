import pandas as pd
import numpy as np
import sqlite3
from mlflow.exceptions import RestException
from shared.utils import configure_logging
import logging
from typing import Any
from datetime import datetime, timezone
import uuid
import sqlalchemy
import mlflow
from mlflow.tracking import MlflowClient
import tempfile
from mlflow.exceptions import MlflowException
from mlflow.deployments import get_deploy_client

configure_logging()

def apply_drift(mlflow_tracking_uri: str) -> pd.DataFrame:

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    model_version = client.get_model_version_by_alias(
        name="penguin_classifier",
        alias="validated"
    )

    logging.info("Retrieving latest validated URI...")

    model_uri = client.get_model_version_download_uri(
        name=model_version.name,
        version=model_version.version)

    logging.info("Retrieving model version")

    # Get artifact root URI from the run

    run_id = model_version.run_id

    logging.info("Retrieving training data using run_id: %s", run_id)


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
            # Handle specifically: e.g., retry logic or raising a custom error
            raise

        except FileNotFoundError:
            logging.error("The file 'train.csv' was not found in the run artifacts.")
            raise

        except Exception as e:
            logging.error("An unexpected error occurred during data retrieval: %s", e)
            raise

    if "species" in data.columns:
        data.pop("species")

    rng = np.random.default_rng()

    data["body_mass_g"] += rng.uniform(
        1,
        3 * data["body_mass_g"].std(),
        size=len(data),
    )

    return data


def generate_traffic(
        data: pd.DataFrame,
        samples: int,
        data_collection_uri: str,

) -> None:
    """
    Generate synthetic traffic by sampling data and creating classifications.

    This function repeatedly samples batches of data, generates classifications
    for each batch, and stores the results in a PostgreSQL database. It continues
    until the specified number of samples have been processed.
    """

    try:
        # create a variable to keep count of classifications generated
        classifications_generated = 0

        while classifications_generated < samples:
            # instantiate dictionary called payload
            payload = {}
            #sample from the drift dataset replacing NaN values with none to ensure
            #compatibility with JSON objects
            batch = data.sample(n=10).replace([np.nan, np.inf, -np.inf], None)
            # generate a dictionary of inputs to send to the endpoint using a list
            # comprehension:

            # 1. iterate through each row of the dataframe
            # 2. convert series to a python dictionary into a list of tuples
            # 3. For each row generate in dataframe generated a dictionary containing
            # column:value for each row entry

            payload["inputs"] = [
                {
                    k: (None if pd.isna(v) else v)
                    for k, v in row.to_dict().items()
                }
                for _, row in batch.iterrows()
            ]

            # send inputs to the endpoint
            classifications = generate_classifications(payload)
            print(f"Classifications type: {type(classifications)}, value: {classifications}")
            logging.info(f"Generated classifications %s", classifications)
            try:

                # capture the input data and classifications and save it to the database
                capture_traffic(batch, classifications, data_collection_uri)
            except sqlite3.Error:
                logging.exception(
                    "There was an error connecting to the database. ",
                )
            # update the count of classifications generated
            classifications_generated += len(batch)
            logging.info("Generated %s classifications",classifications_generated)

    except Exception:
        logging.exception("There was an error sending traffic to the endpoint.")

def generate_classifications(payload: dict[str, Any]) \
        -> dict[str, Any] | list[Any] | None:
    """
    Send a classification request to the inference endpoint and return the results.
    """

    try:

        client = get_deploy_client("sagemaker:/us-east-1")
        response = client.predict(deployment_name="PenguinsEndpoint", inputs=payload["inputs"])

    except RestException as e:
        logging.exception("There was an error sending traffic to the endpoint. %s", e )

    except MlflowException as e:
        logging.exception("There was an error sending traffic to the endpoint: %s", e)

    except Exception as e:
        logging.exception("There was an error sending traffic to the endpoint: %s", e)

    return response


def capture_traffic(model_input: pd.DataFrame,
                    classifications: dict, data_uri: str) -> None:
    logging.info("Storing input payload and predictions in the database...")
    engine = None
    try:
        data = model_input.copy()
        data["date"] = datetime.now(timezone.utc)
        data["classification"] = None
        data["ground_truth"] = None

        if classifications is not None and len(classifications) > 0:
            data["classification"] = [item['classification'] for item in classifications['predictions']]

        data["uuid"] = [str(uuid.uuid4()) for _ in range(len(data))]

        print(f"Attempting to insert {len(data)} rows into database...")
        engine = sqlalchemy.create_engine(data_uri)
        data.to_sql("data", engine, if_exists="append", index=False)
        print(f"Successfully inserted {len(data)} rows.")

    except Exception as e:
        print(f"ERROR: {e}")
        logging.exception(
            "There was an error saving the input request and output prediction "
            "in the database.",
        )
    finally:
        if engine is not None:
            engine.dispose()
