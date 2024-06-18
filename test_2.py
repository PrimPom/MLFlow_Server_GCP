#https://github.com/mlflow/mlflow/issues/7009
import os
import subprocess
import sys
from datetime import datetime
import mlflow
import requests


# Authentication function

def authenticate_mlflow():
    """
    Authenticate to mlflow cluster
    """
    if os.getenv("OAUTH_CLIENT_ID") != None:
        try:
            auth = subprocess.run("gcloud auth activate-service-account --key-file=" + SERVICE_ACCOUNT_JSON, shell=True,
                                  check=True, capture_output=True)
            print("Authenticated service account")
        except subprocess.CalledProcessError:
            print("Authentication with service account " + SERVICE_ACCOUNT_JSON + " failed")
            sys.exit(1)
        try:
            stream = subprocess.run("gcloud auth print-identity-token --audiences={OAUTH_CLIENT_ID}".format(
                OAUTH_CLIENT_ID=os.getenv("OAUTH_CLIENT_ID")), shell=True, text=True, check=True, capture_output=True)
            # set token
            token = stream.stdout.rstrip()
            os.environ["MLFLOW_TRACKING_TOKEN"] = token
        except subprocess.CalledProcessError:
            print("Could not fetch token for OAuth client " + os.getenv("OAUTH_CLIENT_ID"))
            sys.exit(1)

        # set tracking uri
        mlflow.set_tracking_uri(TRACKING_URI)
        print("Authenticated MLFlow")

        # Create an experiment if it doesn't exist
        experiment_name = "Experiment with auth 2"
        if not mlflow.get_experiment_by_name(name=experiment_name):
            mlflow.create_experiment(
                name=experiment_name
            )
        exp = mlflow.get_experiment_by_name(experiment_name)

        return exp
    else:
        print("Missing OAuth client ID environmental variable \"OAUTH_CLIENT_ID\"to complete authentication")

    return None


# set constants
TRACKING_URI = "https://mlflow.manipai.dev"
SERVICE_ACCOUNT_JSON = "private.json"
os.environ["OAUTH_CLIENT_ID"] = "953272985429-8mbakf4goo5d0j18qe08nvuq90mt0908.apps.googleusercontent.com"

# authenticate and run requests
exp = authenticate_mlflow()

# token = os.getenv("MLFLOW_TRACKING_TOKEN")

# Define the run name and tags for the experiment
run_name = datetime.now().strftime("%Y-%m-%d_%H:%M")
tags = {
    "env": "test",
    "data_date": "2023-11-24",
    "model_type": "ElasticNet",
    "experiment_description": "Tutorial MLFlow experiment"
    # ... other tags ...
}

# Start the MLflow run
with mlflow.start_run(
        experiment_id=exp.experiment_id,
        run_name=run_name,
        tags=tags
):
    # Log the hyperparameters used in the model
    mlflow.log_param("alpha", 12)
    mlflow.log_param("l1_ratio", 12)

    # Log the metrics
    mlflow.log_metric("rmse", 12)
    mlflow.log_metric("r2", 12)
    mlflow.log_metric("mae", 12)

    mlflow.log_artifact()

