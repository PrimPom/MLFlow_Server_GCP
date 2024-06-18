"Simple script to log a experiment on MLFlow Server"

# Import libraries
import pandas as pd
import numpy as np
from datetime import datetime
import google.oauth2


import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from google.auth.transport.requests import Request
from google.oauth2 import id_token
import requests
import os


credential_path = "private.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def make_iap_request(url, client_id, method="GET", **kwargs):
    """Makes a request to an application protected by Identity-Aware Proxy.

    Args:
      url: The Identity-Aware Proxy-protected URL to fetch.
      client_id: The client ID used by Identity-Aware Proxy.
      method: The request method to use
              ('GET', 'OPTIONS', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE')
      **kwargs: Any of the parameters defined for the request function:
                https://github.com/requests/requests/blob/master/requests/api.py
                If no timeout is provided, it is set to 90 by default.

    Returns:
      The page body, or raises an exception if the page couldn't be retrieved.
    """
    # Set the default timeout, if missing
    if "timeout" not in kwargs:
        kwargs["timeout"] = 90

    # Obtain an OpenID Connect (OIDC) token from metadata server or using service
    # account.
    open_id_connect_token = id_token.fetch_id_token(Request(), client_id)

    # Fetch the Identity-Aware Proxy-protected URL, including an
    # Authorization header containing "Bearer " followed by a
    # Google-issued OpenID Connect token for the service account.
    resp = requests.request(
        method,
        url,
        headers={"Authorization": "Bearer {}".format(open_id_connect_token)},
        **kwargs
    )
    if resp.status_code == 403:
        raise Exception(
            "Service account does not have permission to "
            "access the IAP-protected application."
        )
    elif resp.status_code != 200:
        raise Exception(
            "Bad response from application: {!r} / {!r} / {!r}".format(
                resp.status_code, resp.headers, resp.text
            )
        )
    else:
        return resp.text


response = make_iap_request(url="https://mlflow.manipai.dev", client_id="953272985429-8mbakf4goo5d0j18qe08nvuq90mt0908.apps.googleusercontent.com", method="GET")
print(response)




# Define the tracking URI for the MLflow experiment
TRACKING_URI = "https://mlflow.manipai.dev"

# Read the wine-quality dataset from a CSV file
csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
data = pd.read_csv(csv_url, sep=";")

# Split the data into training and testing sets
train, test = train_test_split(data)

# Extract the features and target variable from the data
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# Define hyperparameters for the Elastic Net model
alpha = 0.5
l1_ratio = 0.5
random_state = 42
max_iter = 1000

# Create an Elastic Net model with the defined hyperparameters
lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=max_iter)

# Fit the model to the training data
lr.fit(train_x, train_y)

# Make predictions on the testing data
predictions = lr.predict(test_x)

# Evaluate the model's performance
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

(rmse, mae, r2) = eval_metrics(test_y, predictions)

# Print the evaluation metrics
print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}, random_state={random_state}, max_iter={max_iter}):")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")
from google.auth.transport.requests import Request
from google.oauth2 import id_token
import requests
import os


credential_path = "private.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def make_iap_request(url, client_id, method="GET", **kwargs):
    """Makes a request to an application protected by Identity-Aware Proxy.

    Args:
      url: The Identity-Aware Proxy-protected URL to fetch.
      client_id: The client ID used by Identity-Aware Proxy.
      method: The request method to use
              ('GET', 'OPTIONS', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE')
      **kwargs: Any of the parameters defined for the request function:
                https://github.com/requests/requests/blob/master/requests/api.py
                If no timeout is provided, it is set to 90 by default.

    Returns:
      The page body, or raises an exception if the page couldn't be retrieved.
    """
    # Set the default timeout, if missing
    if "timeout" not in kwargs:
        kwargs["timeout"] = 90

    # Obtain an OpenID Connect (OIDC) token from metadata server or using service
    # account.
    open_id_connect_token = id_token.fetch_id_token(Request(), client_id)

    # Fetch the Identity-Aware Proxy-protected URL, including an
    # Authorization header containing "Bearer " followed by a
    # Google-issued OpenID Connect token for the service account.
    resp = requests.request(
        method,
        url,
        headers={"Authorization": "Bearer {}".format(open_id_connect_token)},
        **kwargs
    )
    if resp.status_code == 403:
        raise Exception(
            "Service account does not have permission to "
            "access the IAP-protected application."
        )
    elif resp.status_code != 200:
        raise Exception(
            "Bad response from application: {!r} / {!r} / {!r}".format(
                resp.status_code, resp.headers, resp.text
            )
        )
    else:
        return resp.url


response = make_iap_request(url="https://mlflow.manipai.dev", client_id="953272985429-8mbakf4goo5d0j18qe08nvuq90mt0908.apps.googleusercontent.com", method="GET")



# Set the tracking URI for the MLflow experiment
mlflow.set_tracking_uri(response)

# Create an experiment if it doesn't exist
experiment_name = "Experiment with auth"
if not mlflow.get_experiment_by_name(name=experiment_name):
    mlflow.create_experiment(
        name=experiment_name
    )
experiment = mlflow.get_experiment_by_name(experiment_name)

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
    experiment_id=experiment.experiment_id, 
    run_name=run_name, 
    tags=tags
):
    
    # Log the hyperparameters used in the model
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("max_iter", max_iter)
    
    # Log the metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    # LOg images

    
    # Log model:
    signature = infer_signature(train_x, predictions)
    mlflow.sklearn.log_model(lr, "model", signature=signature)