import os
import mlflow
import logging
from mlflow import MlflowClient
from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException
from shared.utils import configure_logging
from datetime import datetime, timezone


configure_logging()


def deploy_to_sagemaker(mlflow_tracking_uri, sagemaker_role):
    """Deploy the model to SageMaker.

    This function creates a new SageMaker model, endpoint configuration, and
    endpoint to serve the latest version of the model.

    Raises:
        Exception: If the deployment fails, causing Metaflow to stop execution.
    """

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    model_version = client.get_model_version_by_alias(
        name="penguin_classifier",
        alias="validated"
    )

    logging.info("Retrieving latest validated URI...")

    model_uri = client.get_model_version_download_uri(
        name=model_version.name,
        version=model_version.version
    )

    logging.info("Model URI: %s", model_uri)

    deployment_configuration = {
        "instance_type": "ml.c5.large",
        "instance_count": 1,
        "execution_role_arn": sagemaker_role,
        "image_url": "<ecr-image>",
        "synchronous": True,
        # We want to archive resources associated with the endpoint that become
        # inactive as the result of updating an existing deployment.
        "archive": True,
        # Notice how we are storing the version number as a tag.
        "tags": {"version": model_version.version},
    }

    # If the data capture destination is defined, we can configure the SageMaker
    # endpoint to capture data.

    deployment_target_uri = "sagemaker:/us-east-1"

    logging.info("Deployment target URI: %s", deployment_target_uri)

    deployment_client = get_deploy_client(deployment_target_uri)

    deployment_name = "PenguinsEndpoint"

    try:
        # Let's return the deployment with the name of the endpoint we want to
        # create. If the endpoint doesn't exist, this function will raise an
        # exception.
        try:

            logging.info("Checking for existing deployment:...")

            deployment_client.get_deployment(name=deployment_name)

            logging.info("Existing deployment found")

            deployment = deployment_client.update_deployment(
                name=deployment_name,
                model_uri=model_uri,
                flavor="python_function",
                config=deployment_configuration,
            )
        except Exception:

            logging.info("No running deployment found, creating new deployment...")

            deployment = deployment_client.create_deployment(
                name=deployment_name,
                model_uri=model_uri,
                flavor="python_function",
                config=deployment_configuration,
            )

        logging.info("Model deployed successfully to SageMaker")

        logging.info("Updating MLFlow model alias: validated -> deployed...")

        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_version.name,
            alias="deployed",
            version=f"{model_version.version}",
        )

        client.set_model_version_tag(
            name=model_version.name,
            version=model_version.version,
            key="deployed_to production",
            value=True,
        )

        client.set_model_version_tag(
            name=model_version.name,
            version=model_version.version,
            key="date_deployed_to production",
            value=datetime.now(timezone.utc).isoformat(),
        )

        logging.info("MLFlow model aliases now: %s", model_version.aliases )

        return None

    except MlflowException as e:
        # Log the error and re-raise to stop Metaflow execution
        logging.error("Failed to deploy model to SageMaker: %s", str(e))
        logging.error("Deployment failed. Please check your MLflow and SageMaker configuration.")

        # Re-raise the exception to stop Metaflow execution
        raise Exception(f"SageMaker deployment failed: {str(e)}") from e

    except Exception as e:
        # Catch any other unexpected exceptions
        logging.error("Unexpected error during SageMaker deployment: %s", str(e))

        # Re-raise to stop Metaflow execution
        raise Exception(f"Unexpected deployment error: {str(e)}") from e






