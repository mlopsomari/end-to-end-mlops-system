from metaflow import (
    FlowSpec,
    S3,
    Parameter,
    card,
    project,
    pypi_base,
    step)
import pandas as pd
from shared.model import PyfuncModel
import logging
from shared.utils import configure_logging


configure_logging()



@project(name="continous_training")
@pypi_base(
        python="3.10.9",
        packages={'numpy': '1.26.4',
                'seaborn': '0.12.2',
                'mlflow': '3.1.4',
                'pandas': '2.3.1',
                'pyarrow': '14.0.2',
                'scikit-learn': '1.7.2',
                'evidently': '0.7.17',
                'sniffio': '1.3.0',
                'anyio': '4.4.0',
                'SQLAlchemy': '2.0.46',
                'psycopg2-binary': '2.9.11',
                'boto3': '1.42.59',
                  })
class Train(FlowSpec):
    """
    This pipeline trains, logs and deploys a random forrest classifier on the
    penguins dataset.
    """

    data_collection_uri = Parameter(
        "rds_uri",
        default=None,
        type= str,
        help="The URI for the Amazon RDS database used to capture inference traffic.",
        required=True,
        show_default=True,
    )
    mlflow_tracking_uri = Parameter(
        "mlflow_tracking_uri",
        default=None,
        type= str,
        help="The URI for MLflow Tracking Server",
        required=True,
        show_default=True,
    )
    mlflow_experiment_name = Parameter(
        "mlflow_experiment_name",
        default="continuous_training",
        type= str,
        help="The name of the MLflow experiment.",
        required=True,
        show_default=True,
    )

    s3_dataset_uri = Parameter(
        "s3_dataset_uri",
        type= str,
        help="The S3 URI for the dataset.",
        required=True,
    )

    @step
    def start(self):
        """Configure MLflow experiment and run"""
        from shared.mlflow_config import setup_tracking
        from datetime import datetime


        self.parent_run_name = f"penguins-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"

        mlflow_tracking = setup_tracking(
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            experiment_name=self.mlflow_experiment_name,
            parent_run_name=self.parent_run_name,
        )

        self.parent_run_id = mlflow_tracking["parent_run_id"]

        self.next(self.download_data)

    @step
    def download_data(self):
        """Download and persist the penguins dataset."""

        try:
            with S3() as s3:
                logging.info("Downloading penguins dataset from S3.")
                s3obj = s3.get(self.s3_dataset_uri)

                self.dataset = s3obj.blob
                logging.info("Dataset download successful!")
        except Exception as e:
            logging.error(e)

        self.next(self.validate_data)

    @step
    def validate_data(self):
        """Validate the penguins dataset."""

        from core.training.etl import validate_dataset
        from io import BytesIO

        logging.info("Converting Penguins dataset to Pandas DataFrame.")
        self.dataset = pd.read_csv(BytesIO(self.dataset))

        validate_dataset(self.dataset)

        self.next(self.transform_data)

    @step
    def transform_data(self):
        """Split data into train and test datasets"""

        from core.training.etl import split_data

        (self.X_train,
         self.X_test,
         self.y_train,
         self.y_test,
         ) = split_data(self.dataset)

        self.next(self.build_pipeline)

    @step
    def build_pipeline(self):
        """Build a scikit-learn pre-processing pipeline with Random Forest
        base estimator."""

        from core.training.training_utility import build_sklearn_pipeline

        self.base_pipeline = build_sklearn_pipeline(self.X_train)

        self.next(self.generate_param_combinations)

    @step
    def generate_param_combinations(self):
        """ Generate random parameter combinations for hyperparameter search."""

        from core.training.training_utility import build_param_combinations

        self.param_combinations = build_param_combinations()

        self.next(self.cross_validation, foreach="param_combinations")

    @step
    def cross_validation(self):
        """Evaluate a single parameter combination using cross-validation.
        This step runs in parallel for each parameter combination."""

        from core.training.training_utility import run_cross_validation

        params = self.input

        self.trial_results = run_cross_validation(
            params=params,
            base_pipeline=self.base_pipeline,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            parent_run_id=self.parent_run_id,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            experiment_name=self.mlflow_experiment_name,
        )

        self.next(self.collect_results)

    @step
    def collect_results(self, inputs):
        """  """

        from core.training.training_utility import select_best_model

        trial_results_list = [inp.trial_results for inp in inputs]

        self.merge_artifacts(
            inputs,
            exclude=['trial_results', 'param_combinations'],
        )

        self.best_model = select_best_model(trial_results_list=trial_results_list)

        self.next(self.train_best_model)

    @step
    def train_best_model(self):
        """Train best model on full dataset."""

        from core.training.training_utility import train_model

        self.best_estimator = train_model(
            base_pipeline=self.base_pipeline,
            params=self.best_model["parameters"],
            X_train=self.X_train,
            y_train=self.y_train,
        )

        self.next(self.evaluate_best_model)

    @step
    def evaluate_best_model(self):
        """Evaluate best model on test dataset."""

        from core.training.training_utility import evaluate_model

        self.test_accuracy = evaluate_model(
            best_estimator=self.best_estimator,
            X_test=self.X_test,
            y_test=self.y_test,
        )

        self.next(self.register_model)

    @step
    def register_model(self):
        """Register the model to MLflow."""

        from core.training.training_utility import register_model

        if self.test_accuracy["test_accuracy"] >= 0.9:

            register_model(
                model=self.best_estimator,
                params=self.best_model["parameters"],
                test_accuracy=self.test_accuracy["test_accuracy"],
                parent_run_id=self.parent_run_id,
                pyfunc_model=PyfuncModel(data_collection_uri=self.data_collection_uri, data_capture=True),
                mlflow_tracking_uri=self.mlflow_tracking_uri,
                mlflow_experiment_name=self.mlflow_experiment_name,
                dataset=self.dataset,
            )

        self.next(self.end)

    @step
    def end(self):
        """End training and evaluation."""
        logging.info("Training finished.")

if __name__ == "__main__":
    Train()









































