from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import  ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.base import clone, BaseEstimator
from typing import Any, Union
import pandas as pd
import random
import logging
from shared.utils import configure_logging
import tempfile
from pathlib import Path
from mlflow.tracking import MlflowClient

configure_logging()

def build_sklearn_pipeline(X_train: Union[pd.DataFrame, np.ndarray]) -> Pipeline:
    """Build preprocessing pipeline and base Random Forest model"""

    logging.info("Building sklearn pipeline")
    # Define preprocessing steps
    numerical_features = X_train.select_dtypes(include=np.number).columns
    categorical_features = X_train.select_dtypes(include='object').columns


    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    logging.info("Numerical and categorical transformation pipelines created")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full pipeline with preprocessor and model

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))])

    logging.info("Random forrest classifier with preprocessor created")

    return pipeline

def build_param_combinations():
    """Creates a set of random forest model parameters"""

    logging.info(f"Generating parameter combinations...")

    # Define parameter distributions
    param_distributions = {
        'classifier__n_estimators': [50, 100, 150, 200, 250, 300],
        'classifier__max_depth': [None, 5, 10, 15, 20, 25, 30],
        'classifier__min_samples_split': [2, 3, 4, 5, 8, 10],
        'classifier__min_samples_leaf': [1, 2, 3, 4, 5],
        'classifier__max_features': ['sqrt', 'log2', None, 0.5, 0.7],
        'classifier__bootstrap': [True, False]
    }

    # Generate random parameter combinations
    random.seed(42)
    param_combinations = [
        {**{param_name: random.choice(param_values)
            for param_name, param_values in param_distributions.items()},
            "trial_id": i}
        for i in range(6)
    ]

    logging.info("Parameter combinations created %s", param_combinations)

    return param_combinations


def run_cross_validation(params: dict[str, Any],
                         base_pipeline: Pipeline,
                         X_train: Union[pd.DataFrame, np.ndarray],
                         y_train : Union[pd.Series, np.ndarray],
                         X_test: Union[pd.DataFrame, np.ndarray],
                         y_test: Union[pd.Series, np.ndarray],
                         parent_run_id: str,
                         mlflow_tracking_uri: str,
                         experiment_name: str,
                        ) -> dict[str, Any]:
    """Runs cross validation for given parameters and tracks training with MLflow"""


    trial_id = params.pop("trial_id")
    run_name = f"Trial {trial_id}"

    pipeline = clone(base_pipeline)
    pipeline.set_params(**params)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(nested=True, run_name=run_name):

            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='accuracy')
            pipeline.fit(X_train, y_train)
            test_accuracy = accuracy_score(y_test, pipeline.predict(X_test))

            mlflow.log_metric("mean_cv_score", cv_scores.mean())
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_params(params)

    return {
        "trial_id": trial_id,
        "parameters": params,
        "mean_cv_score": cv_scores.mean(),
        "test_accuracy": test_accuracy,
    }

def select_best_model(trial_results_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Select best model based on cross validation score"""
    logging.info("Collecting results from all trials...")

    # Sort results by performance (descending)
    all_results = sorted(trial_results_list, key=lambda x: x["test_accuracy"], reverse=True)
    best_result: dict[str, Any] = all_results[0]

    logging.info("Best parameters: %s", best_result["parameters"])
    logging.info("Best CV score: %.4f", best_result["mean_cv_score"])
    logging.info("Best test accuracy: %.4f", best_result["test_accuracy"])

    return best_result


def train_model(
    base_pipeline: Pipeline,
    params: dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
):
    """Train a model with given parameters"""
    pipeline = clone(base_pipeline)
    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)

    return pipeline

def evaluate_model(best_estimator: BaseEstimator,
                   X_test: Union[pd.DataFrame, np.ndarray],
                   y_test: Union[pd.Series, np.ndarray],
) -> dict[str, int]:
    """Evaluate a model"""

    logging.info("Evaluating best model...")

    test_accuracy = accuracy_score(y_test, best_estimator.predict(X_test))

    logging.info("Test accuracy: %.4f", test_accuracy)

    return {"test_accuracy": test_accuracy}


def register_model(
        model: BaseEstimator,
        params: dict[str, Any],
        test_accuracy: float,
        parent_run_id: str,
        pyfunc_model: mlflow.pyfunc.PythonModel,
        mlflow_tracking_uri: str,
        mlflow_experiment_name: str,
        dataset: pd.DataFrame,
) -> None:
    """Register a model with given parameters"""

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    client = MlflowClient()
    with (
        mlflow.start_run(run_id=parent_run_id),
        tempfile.TemporaryDirectory() as directory,
    ):
        # Log parameters and metrics for the best model
        logging.info("Logging metrics and parameters for best model from trials")
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_params(params)

        # Log best model
        logging.info("Logging best model from trials")
        model_info = mlflow.pyfunc.log_model(
            python_model=pyfunc_model,
            registered_model_name="penguin_classifier",
            name="penguin_rnd_frst_classifier",
            infer_code_paths=[(Path(__file__).parent / "flows/model.py").as_posix()],
            artifacts=get_model_artifacts(directory, model, dataset),
            pip_requirements=get_pip_requirements(),
            signature=get_model_signature()
        )

    # uri = model_info.model_uri
    version = model_info.registered_model_version
    client.set_registered_model_alias(
        name="penguin_classifier",
        alias="validated",
        version=version,
    )




def get_model_artifacts(directory: str, model: BaseEstimator, dataframe: pd.DataFrame):
    """Return the list of artifacts that will be included with model.

    The model must preprocess the raw input data before making a prediction, so we
    need to include the Scikit-Learn transformers as part of the model package.
    """
    import joblib

    # Save the model inside the supplied directory.
    model_path = (Path(directory) / "model.joblib").as_posix()
    joblib.dump(model, model_path)

    dataset_path = (Path(directory) / "dataset.csv").as_posix()
    dataframe.to_csv(dataset_path, index=False)

    return {
        "model": model_path,
        "dataset": dataset_path,
    }

def get_model_signature():
    """Return the model's signature.

    The signature defines the expected format for model inputs and outputs. This
    definition serves as a uniform interface for appropriate and accurate use of
    a model.
        """
    from mlflow.models import infer_signature

    return infer_signature(
        model_input={
        'bill_length_mm': 38.0,
        'bill_depth_mm': 18.3,
        'flipper_length_mm': 188.0,
        'body_mass_g': 4200.0,
        'island': 'Torgersen',
        'sex': 'Male',
        },
        model_output={"prediction": "Adelie"},
        params={"data_capture_uri": "database.db",
                "data_capture": False},
        )

def get_pip_requirements():
    """Return the dictionary of pip requirements."""

    return [
        "scikit-learn==1.7.2",
        "pandas==1.5.3",
        "numpy==1.23.5"
    ]