import mlflow
from shared.utils import configure_logging
import logging
import os
from typing import Optional, Dict, Union




configure_logging()






def setup_tracking(
    mlflow_tracking_uri: Optional[str],
    experiment_name: Optional[str],
    parent_run_name: str,
) -> Dict[str, Union[str, None]]:
    """
    Configure MLflow tracking and start a parent run.
    Returns a dictionary of MLflow tracking artifacts.
    """
    try:
        # Determine tracking URI
        mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Determine experiment
        experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

        # Start the run
        with mlflow.start_run(run_name=parent_run_name) as run:
            parent_run_id = run.info.run_id

        logging.info(
            "MLFLOW_RUN_ID environment variable set as: %s", parent_run_id
        )

        return {
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "experiment_name": experiment_name,
            "parent_run_name": parent_run_name,
            "parent_run_id": parent_run_id,
        }

    except Exception as e:
        message = f"Failed to connect to MLflow server {mlflow_tracking_uri}."
        raise RuntimeError(message) from e
