from metaflow import (
    FlowSpec,
    Parameter,
    card,
    environment,
    project,
    pypi_base,
    resources,
    step,
    S3,
)
import logging
from datetime import datetime
from shared.utils import configure_logging
from mlflow.tracking.client import MlflowClient


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
class Monitoring(FlowSpec):

    mlflow_tracking_uri = Parameter(
        "mlflow_tracking_uri",
        default=None,
        help="MLflow tracking URI",
        type=str,
    )

    data_collection_uri = Parameter(
        "rds_uri",
        default=None,
        required=True,
        type=str,
    )

    @step
    def start(self):

        self.next(self.retrieve_data)

    @step
    def retrieve_data(self):

        from core.monitoring.monitoring_utility import retrieve_data

        self.current_data = retrieve_data(
            data_collection_uri=self.data_collection_uri
        )

        self.next(self.create_evidently_data)

    @step
    def create_evidently_data(self):

        import mlflow
        from core.monitoring.monitoring_utility import create_datasets

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        (self.current_evidently_data,
         self.reference_evidently_data) = create_datasets(
            current_dataset=self.current_data,
        )

        self.next(self.report)

    @card(type='html')
    @step
    def report(self):
        from core.monitoring.monitoring_utility import run_report, get_evidently_html
        import tempfile
        import json
        import os
        from metaflow import S3
        from datetime import datetime

        eval_report = run_report(
            ev_ref_data=self.reference_evidently_data,
            ev_curr_data=self.current_evidently_data,
        )

        # Capture HTML for the card
        self.html = get_evidently_html(eval_report)

        # Build the S3 key
        now = datetime.now()
        current_datetime_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"drift_report_{current_datetime_string}.json"
        s3_key = f"drift_reports/{file_name}"

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
                tmp_path = tmp.name

            # Write report to temp file
            eval_report.save_json(tmp_path)

            # Extract drift metrics from the saved JSON before uploading
            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.drift_threshold = data["metrics"][0]["config"]["drift_share"]
            self.drift_value = data["metrics"][0]["value"]["share"]
            self.drift = (
                    self.drift_threshold is not None
                    and self.drift_value is not None
                    and self.drift_value >= self.drift_threshold
            )

            # Upload to S3
            with S3(s3root='s3://continous-training-103441327704') as s3:
                s3.put_files([(s3_key, tmp_path)])

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        self.next(self.log_drift)

    @step
    def log_drift(self):

        import mlflow
        from datetime import datetime, timezone

        logging.info("Drift detected: %s", self.drift)

        if self.drift:
            logging.info("Tagging deployed model with 'drift_detected'")

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

            client = MlflowClient()

            model_version = client.get_model_version_by_alias(
                name="penguin_classifier",
                alias="deployed"
            )

            client.set_model_version_tag(
                name=model_version.name,
                version=model_version.version,
                key="drift_detected",
                value=True,
            )

            client.set_model_version_tag(
                name=model_version.name,
                version=model_version.version,
                key="drift_value",
                value=self.drift_value,
            )

            client.set_model_version_tag(
                name=model_version.name,
                version=model_version.version,
                key="drift_detected_at",
                value=datetime.now(timezone.utc).isoformat())

            logging.info("Model tagged with 'drift_detected'")

        self.next(self.end)

    @step
    def end(self):

        logging.info("Drift present... %s", self.drift)

        logging.info("Finished")

if __name__ == "__main__":
    Monitoring()






