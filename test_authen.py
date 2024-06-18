
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
        return resp


response = make_iap_request(url="https://mlflow.manipai.dev", client_id="953272985429-8mbakf4goo5d0j18qe08nvuq90mt0908.apps.googleusercontent.com", method="GET")

print(response)

