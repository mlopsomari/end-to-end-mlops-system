from mlflow.tracking import MlflowClient
from typing import Tuple
import logging
import pandas as pd
import sqlalchemy
from sklearn.model_selection import train_test_split
from shared.utils import configure_logging
import mlflow
from datetime import datetime, timezone
from mlflow.exceptions import MlflowException
from mlflow.deployments import get_deploy_client

configure_logging()



def get_last_train_time(mlflow_tracking_uri) -> Tuple[datetime, datetime, str]:

    logging.info("Getting last training time")

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = MlflowClient()

    latest_version = client.get_model_version_by_alias(
        name="penguin_classifier",
        alias="validated",
    )

    print(f"Latest version: {latest_version.version}")

    # 2. Get the run ID associated with that model version
    run_id = latest_version.run_id

    # 3. Fetch the run and timestamps
    run = client.get_run(run_id)

    start_time = datetime.fromtimestamp(run.info.start_time / 1000.0)
    end_time   = datetime.fromtimestamp(run.info.end_time   / 1000.0)

    logging.info("Training started: %s", start_time)
    logging.info("Training ended: %s  ", end_time)
    return start_time, end_time, latest_version.version


def check_drift(version: str, mlflow_tracking_uri: str) -> bool:

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    model_version = client.get_model_version(
        name="penguin_classifier",
        version=version,
    )
    drift = model_version.tags.get("drift_detected")
    return drift

def transform_data(inference_data):

    ground_truth = inference_data["ground_truth"]
    features = inference_data.drop(columns=["ground_truth"])

    X_train, X_test, y_train, y_test = train_test_split(
        features, ground_truth, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test



def retrieve_pipeline(mlflow_tracking_uri, experiment_name):

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)


    loaded_pyfunc = mlflow.pyfunc.load_model(
        "models:/penguin_classifier@deployed"
    )

    fitted_pipeline = loaded_pyfunc._model_impl.python_model.model

    return fitted_pipeline


def retrain(X_train,
            X_test,
            y_train,
            y_test,
            fitted_pipeline,
            mlflow_tracking_uri,
            experiment_name):

    from sklearn.base import clone
    from datetime import datetime, timezone
    from sklearn.metrics import accuracy_score


    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_id = f"retraining-penguins-classifier{datetime.now(timezone.utc).strftime('%d-%m-%Y_%H-%M-%S')}"

    with mlflow.start_run(run_name=run_id):

        base_pipeline = clone(fitted_pipeline)

        new_model = base_pipeline.fit(X_train, y_train)

        test_accuracy = accuracy_score(y_test, new_model.predict(X_test))

        mlflow.log_metric("test_accuracy", test_accuracy)


    return new_model, run_id, test_accuracy





def retrieve_data(data_collection_uri: str, start_time: str) -> pd.DataFrame:
    """
    Retrieve all data from the inference database after a specified time.

    Connects to the PostgreSQL database, executes a query to fetch all records
    from the data table where date is after the specified start_time, and
    returns the results as a pandas DataFrame. Records are ordered by date
    in descending order (newest first).

    Parameters
    ----------
    data_collection_uri : str
        URI of the data collection table.
    start_time : str
        ISO format datetime string (UTC timezone) to filter records after this time.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all records with the following columns:
        - uuid : unique identifier for each record
        - island : island location
        - sex : sex of the penguin
        - bill_length_mm : bill length in millimeters
        - bill_depth_mm : bill depth in millimeters
        - flipper_length_mm : flipper length in millimeters
        - body_mass_g : body mass in grams
        - classification : model prediction
        - ground_truth : actual classification label

        Records are sorted by date in descending order.

    Notes
    -----
    The function establishes a connection to the PostgreSQL database, retrieves
    only data after start_time, and properly closes the connection before returning.

    Examples
    --------
    from datetime import datetime, timezone
    start = datetime.now(timezone.utc).isoformat()
    data = retrieve_data(start)
    print(data.head())
    """
    # Establish connection to the PostgreSQL database using SQLAlchemy
    engine = sqlalchemy.create_engine(data_collection_uri)

    logging.info("Retrieving data from: %s", data_collection_uri)
    logging.info("Retrieving records after: %s", start_time)

    # SQL query to select all relevant columns from the data table
    # Only retrieve records where date is after the specified start_time
    # Results are ordered by date in descending order (newest first)
    query = (
        "SELECT island, sex, bill_length_mm, bill_depth_mm, flipper_length_mm, "
        "body_mass_g, ground_truth FROM data "
        "WHERE date > %s "
        "ORDER BY date DESC;"
    )

    # Execute query and load results into a pandas DataFrame
    data = pd.read_sql_query(query, engine, params=(start_time,))

    samples = len(data)

    logging.info("Retrieved data of %s records", samples)

    # Dispose of the engine to free resources
    engine.dispose()

    return data




