import pandas as pd
import random
import logging
from shared.utils import configure_logging
import sqlalchemy


configure_logging()


def retrieve_data(data_collection_uri: str) -> pd.DataFrame:
    """
    Retrieve all unlabeled data from the inference database.

    Connects to the PostgreSQL database, executes a query to fetch all records
    from the data table where ground_truth is NULL (unlabeled data), and
    returns the results as a pandas DataFrame. Records are ordered by date
    in descending order (newest first).

    Parameters
    ----------
    data_collection_uri : str
    URI of the data collection table.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all unlabeled records with the following columns:
        - uuid : unique identifier for each record
        - island : island location
        - sex : sex of the penguin
        - bill_length_mm : bill length in millimeters
        - bill_depth_mm : bill depth in millimeters
        - flipper_length_mm : flipper length in millimeters
        - body_mass_g : body mass in grams
        - classification : model prediction
        - ground_truth : actual classification label (NULL for unlabeled data)

        Records are sorted by date in descending order.

    Notes
    -----
    The function establishes a connection to the PostgreSQL database, retrieves
    only data where ground_truth is NULL (hasn't been labeled yet), and
    properly closes the connection before returning.

    Examples
    --------
    data = retrieve_data()
    print(data.head())
    print(data['ground_truth'].isna().all())  # Should return True
    """
    # Establish connection to the PostgreSQL database using SQLAlchemy
    engine = sqlalchemy.create_engine(data_collection_uri)

    logging.info("Retrieving data....")

    # SQL query to select all relevant columns from the data table
    # Only retrieve records where ground_truth is NULL (unlabeled)
    # Results are ordered by date in descending order (newest first)
    query = (
        "SELECT uuid, island, sex, bill_length_mm, bill_depth_mm, flipper_length_mm, "
        "body_mass_g, classification, ground_truth FROM data "
        "WHERE ground_truth IS NULL "
        "ORDER BY date DESC;"
    )

    # Execute query and load results into a pandas DataFrame
    data = pd.read_sql_query(query, engine)

    logging.info("Retrieved data")

    # Dispose of the engine to free resources
    engine.dispose()

    return data


def get_label(prediction: str, prediction_quality: float) -> str:
    """
    Generate a simulated ground truth label based on prediction quality.

    This function creates synthetic ground truth labels for testing and simulation
    purposes. It returns the predicted label with a probability equal to the
    prediction quality, otherwise returns a random label from the possible classes.

    Parameters
    ----------
    prediction : str
        The model's predicted label. Should be one of the valid penguin species:
        "Adelie", "Chinstrap", or "Gentoo".
    prediction_quality : float
        The desired accuracy rate for the predictions, expressed as a probability
        between 0.0 and 1.0. A value of 0.8 means the ground truth will match
        the prediction 80% of the time.

    Returns
    -------
    str
        A ground truth label. Returns the input prediction with probability equal
        to `prediction_quality`, otherwise returns a randomly selected label from
        ["Adelie", "Chinstrap", "Gentoo"].

    Notes
    -----
    - This function is intended for simulation and testing purposes only
    - Uses random selection, so results are non-deterministic
    - Does not validate that the prediction is a valid penguin species
    - The random label may coincidentally match the prediction even when
      the quality check fails

    Examples
    --------
    # With high prediction quality, usually returns the prediction
    label = get_label("Adelie", prediction_quality=0.9)
    # Returns "Adelie" 90% of the time, random species 10% of the time

    # With low prediction quality, often returns a different label
    label = get_label("Chinstrap", prediction_quality=0.3)
    # Returns "Chinstrap" 30% of the time, random species 70% of the time

    # Simulate perfect predictions
    label = get_label("Gentoo", prediction_quality=1.0)
    print(label)
    'Gentoo'

    See Also
    --------
    random.choice : Used to select random labels when prediction quality check fails
    """

    return (
        prediction
        if random.random() < prediction_quality
        else random.choice(["Adelie", "Chinstrap", "Gentoo"])
        )


def label_data(data: pd.DataFrame, data_uri: str) -> int | None:

    if data.empty:
        return 0

    logging.info("Generating ground truth labels...")

    engine = None
    try:
        engine = sqlalchemy.create_engine(data_uri)

        with engine.connect() as connection:
            for _, row in data.iterrows():
                uuid = row["uuid"]
                label = get_label(row["classification"], 0.8)
                update_query = sqlalchemy.text(
                    "UPDATE data SET ground_truth = :label WHERE uuid = :uuid"
                )
                connection.execute(update_query, {"label": label, "uuid": uuid})
            connection.commit()

        logging.info("Generated ground truth labels")

    except Exception as e:
        print(f"ERROR: {e}")
        logging.exception("There was an error updating ground truth labels.")

    finally:
        if engine is not None:
            engine.dispose()

    return None










