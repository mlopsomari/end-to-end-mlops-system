import logging
import mlflow
import joblib
import uuid
import pandas as pd
from datetime import datetime, timezone
import sqlalchemy





class PyfuncModel(mlflow.pyfunc.PythonModel):

    def __init__(self,
                 data_collection_uri: str | None = None,
                 data_capture: bool | None = True,
    ) -> None:
        self.data_collection_uri = data_collection_uri
        self.data_capture = data_capture
        self.model = None



    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None  :

        # 1. Get the directory path for the model artifact
        model_file_path = context.artifacts["model"]

        # 2. Load the model object using joblib
        self.model = joblib.load(model_file_path)

    def predict(self, context, model_input, params):


        if isinstance(model_input, list | dict):
            model_input = pd.DataFrame(model_input)

        logging.info(
            "Received prediction request with %d %s",
            len(model_input),
            "samples" if len(model_input) > 1 else "sample",
        )

        model_output = self.model.predict(model_input)

        result = [{"classification": pred} for pred in model_output]

        if (
                params
                and params.get("data_capture", False) is True
                or not params
                and self.data_capture
        ):
            self.store_data(model_input, result)

        return result

    def store_data(self, model_input: pd.DataFrame, model_output: list) -> None:
        logging.info("Storing input payload and predictions in the database...")
        try:
            data = model_input.copy()
            data["date"] = datetime.now(timezone.utc)
            data["classification"] = None
            data["ground_truth"] = None

            if model_output is not None and len(model_output) > 0:
                data["classification"] = [item["classification"] for item in model_output]

            data["uuid"] = [str(uuid.uuid4()) for _ in range(len(data))]

            engine = sqlalchemy.create_engine(self.data_collection_uri)
            data.to_sql("data", engine, if_exists="append", index=False)

        except Exception:
            logging.exception(
                "There was an error saving the input request and output prediction "
                "in the database.",
            )

        finally:
            if engine is not None:
                engine.dispose()
