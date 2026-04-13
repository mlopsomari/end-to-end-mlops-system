from metaflow import (
    FlowSpec,
    card,
    step,
    Parameter,
    environment,
    project,
    pypi_base,
    resources,)
import os
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
class Deploy(FlowSpec):
    """
    This pipeline deploys the model trained to a sagemaker endpoint
    """

    mlflow_tracking_uri = Parameter(
        "mlflow_tracking_uri",
        default=None,
        type= str,
        help="The URI for MLflow Tracking Server",
        required=True,
    )
    sagemaker_role = Parameter(
        "sagemaker_role",
        default=None,
        type= str,
        required=True,
        help="""The name of an IAM role granting the SageMaker 
        service permissions to access the specified Docker image
        and S3 bucket containing MLflow model artifacts""",
    )

    ecr_image_uri = Parameter(
        "ecr_image_uri",
        default=None,
        type= str,
        required=True,
        help="The ECR Image URI",
    )

    @step
    def start(self):
        import mlflow
        from core.deployment.deployment_utility import deploy_to_sagemaker

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        logging.info(msg="Deploying model to Sagemaker...")

        self.deployment = deploy_to_sagemaker(self.mlflow_tracking_uri,
                                              self.sagemaker_role,
                                              self.ecr_image_uri)

        logging.info(msg="Deployment successful!!!")

        self.next(self.end)

    @step
    def end(self):
        logging.info(msg="Deployment pipeline successfully completed!!!")


if __name__ == '__main__':
    Deploy()









