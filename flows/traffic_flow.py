import logging

from metaflow import (
    FlowSpec,
    Parameter,
    project,
    pypi_base,
    step,
)
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
                "psycopg2-binary": "2.9.9",
                'SQLAlchemy': '2.0.46',
                'boto3': '1.40.67',
                  })
class Traffic(FlowSpec):

    mlflow_tracking_uri = Parameter(
        name="mlflow_tracking_uri",
        default="http://localhost:5000",
        help="The MLflow tracking uri",
        type=str,
        required=True
    )

    flow_type = Parameter(
        name="flow_type",
        default="traffic",
        required=True,
        type=str,
    )

    data_collection_uri = Parameter(
        name="rds_uri",
        default=None,
        type=str,
        required=True,
    )

    @step
    def start(self):
        """ """
        if self.flow_type not in ["traffic", "labelling"]:
            raise ValueError("Traffic flow type must be traffic or labelling")

        else:
            logging.info('Traffic flow type is: %s', self.flow_type)


        self.next(self.apply_drift)


    @step
    def apply_drift(self):

        if self.flow_type == "traffic":

            from core.monitoring.traffic_utility import apply_drift


            self.drift_data = apply_drift(self.mlflow_tracking_uri)

        self.next(self.traffic)


    @step
    def traffic(self):

        if self.flow_type == "traffic":

            from core.monitoring.traffic_utility import generate_traffic

            generate_traffic(
                data=self.drift_data,
                samples=300,
                data_collection_uri=self.data_collection_uri,
            )

        self.next(self.labelling)

    @step
    def labelling(self):

        from core.monitoring.labelling_utility import retrieve_data, label_data

        self.captured_data = retrieve_data(self.data_collection_uri)
        label_data(self.captured_data, self.data_collection_uri)

        self.next(self.end)

    @step
    def end(self):

        if self.flow_type == "traffic":
            logging.info("Traffic flow ended")

        if self.flow_type == "labelling":
            logging.info("Labelling flow labelled")


if __name__ == "__main__":
    Traffic()




