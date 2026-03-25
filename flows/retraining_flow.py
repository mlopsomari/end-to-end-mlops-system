from metaflow import (
    FlowSpec,
    Parameter,
    card,
    environment,
    project,
    pypi_base,
    resources,
    step,
)
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
                'boto3': '1.40.67'
                  })
class RetrainingFlow(FlowSpec):
    mlflow_tracking_uri = Parameter(
        "mlflow_tracking_uri",
        default=None,
        required=True,
        type=str,
    )

    data_collection_uri = Parameter(
        "rds_uri",
        default=None,
        required=True,
        type=str,
    )

    sagemaker_role = Parameter(
        "sagemaker_role",
        required=True,
        default=None,
        type=str,
    )

    @step
    def start(self):

        from core.monitoring.retraining_utility import get_last_train_time

        self.last_train_time, _, self.deployed_model_version= get_last_train_time(self.mlflow_tracking_uri)

        self.next(self.check_for_drift)

    @step
    def check_for_drift(self):
        """Pull inference generated since the model was last trained."""

        from core.monitoring.retraining_utility import check_drift

        self.drift = check_drift(self.deployed_model_version, self.mlflow_tracking_uri)

        self.next(self.retrieve_data)

    @step
    def retrieve_data(self):
        """Retrieve data from the data collection service."""

        if self.drift == True:

            from core.monitoring.retraining_utility import retrieve_data

            # retrieving production data from the data collection database
            self.inference_data = retrieve_data(
                self.data_collection_uri,
                self.last_train_time
            )
        self.next(self.data_transformation)

    @step
    def data_transformation(self):
        """Transform the data collection service."""

        if self.drift == True:

            from core.monitoring.retraining_utility import transform_data

            (self.X_train,
            self.X_test,
            self.y_train,
            self.y_test) = transform_data(self.inference_data)

        self.next(self.retrieve_pipeline)

    @step
    def retrieve_pipeline(self):
        """Retrieve pipeline used to fit the existing deployed model."""

        if self.drift == True:

            from core.monitoring.retraining_utility import retrieve_pipeline

            self.fitted_pipeline = retrieve_pipeline(
                self.mlflow_tracking_uri,
                experiment_name="continous-training",
            )

        self.next(self.retrain)


    @step
    def retrain(self):
        """Retrain model on new data"""

        if self.drift == True:

            from core.monitoring.retraining_utility import retrain

            self.new_model, self.run_id, self.test_accuracy = retrain(
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                self.fitted_pipeline,
                self.mlflow_tracking_uri,
                experiment_name="continous-training",
            )

        self.next(self.log_model)

    @step
    def log_model(self):
        """Log trained model to mlflow if it meets our quality
        threshold."""

        if self.drift == True:

            from core.training.training_utility import register_model
            from shared.model import PyfuncModel


            logging.info("Determining if model meets our quality threshold.")
            if self.test_accuracy > 0.6:

                self.quality_threshold_met = True

                logging.info("Model meets our quality threshold. %.4f",
                            self.test_accuracy)

                logging.info(msg="Logging model to mlflow model registry...")

                register_model(model=self.new_model,
                            params=self.new_model.get_params(),
                            test_accuracy=self.test_accuracy,
                            parent_run_id=self.run_id,
                            pyfunc_model=PyfuncModel(data_collection_uri=self.rds_uri, data_capture=True),
                            mlflow_tracking_uri=self.mlflow_tracking_uri,
                            mlflow_experiment_name="continous-training",
                            dataset=self.inference_data,
                )

                logging.info(msg="Model successfully logged to mlflow model registry.")

            else:

                self.quality_threshold_met = False
                logging.info(f"Model doesn't meet our quality threshold. %.4f",
                            self.test_accuracy)

        self.next(self.update_deployed_model)


    @step
    def update_deployed_model(self):
        """Update the deployed model"""

        if self.drift == True and self.quality_threshold_met == True:

            import mlflow
            from core.deployment.deployment_utility import deploy_to_sagemaker

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment("continous-training")

            logging.info("Updating deployed model...")
            self.deployment = deploy_to_sagemaker(self.mlflow_tracking_uri, self.sagemaker_role)
            logging.info("Deployment successful!!!")

        self.next(self.end)

    @step
    def end(self):
        """End retraining flow."""
        logging.info("Retraining pipeline completed.")


if __name__ == "__main__":
    RetrainingFlow()






