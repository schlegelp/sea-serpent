import json
import os
import requests
import logging
from urllib.parse import urlparse

# from seatable_api import Account
from .patch import Account, SeaTableAPI

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def _secret_root():
    return os.path.expanduser("~/.seaserpent/secrets")


def _ensure_secret_dir():
    secret_dir = _secret_root()
    os.makedirs(secret_dir, mode=0o700, exist_ok=True)
    try:
        os.chmod(secret_dir, 0o700)
    except OSError:
        pass
    return secret_dir


def _normalize_server_host(server):
    if not server:
        return None
    parsed = urlparse(server)
    host = parsed.netloc or parsed.path
    if host.endswith("/"):
        host = host[:-1]
    return host.replace(":", "_")


def _instance_secret_path(server):
    host = _normalize_server_host(server)
    if not host:
        return None
    return os.path.join(_secret_root(), f"instance.{host}.json")


def _general_secret_path():
    return os.path.join(_secret_root(), "seatable_secret.json")


def save_secret(token, server=None):
    path = _instance_secret_path(server) if server else _general_secret_path()
    _ensure_secret_dir()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token": token}, f, indent=2)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError:
        raise OSError(f"Unable to write SeaTable secret to {path}")
    return path


def load_secret(server=None):
    def _load(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("token")
        except (OSError, json.JSONDecodeError, AttributeError):
            return None

    if server:
        instance_path = _instance_secret_path(server)
        if instance_path and os.path.isfile(instance_path):
            token = _load(instance_path)
            if token:
                return token

    general_path = _general_secret_path()
    if os.path.isfile(general_path):
        token = _load(general_path)
        if token:
            return token

    return None


def resolve_auth_token(auth_token=None, server=None):
    if auth_token:
        return auth_token
    token = load_secret(server)
    if token:
        return token
    return os.environ.get("SEATABLE_TOKEN")


def get_auth_token(username, password, server="https://cloud.seatable.io", save_to_secret=True):
    """Retrieve your auth token.

    Parameters
    ----------
    username :  str
                Login name or email address.
    password :  str
                Your password.
    server :    str
                URL to your seacloud server.
    save_to_secret : bool
                If True, save the returned token to the appropriate secret
                file under ~/.seaserpent/secrets/.

    Returns
    -------
    dict
                Dictionary {'token': 'YOURTOKEN'}

    """
    while server.endswith("/"):
        server = server[:-1]

    url = f"{server}/api2/auth-token/"
    post = {"username": username, "password": password}

    r = requests.post(url, data=post)
    r.raise_for_status()

    token = r.json()
    if save_to_secret:
        path = save_secret(token["token"], server=server)
        print(f"Saved SeaTable auth token to {path}")

    return token


def set_auth_token(token, server=None, save_to_secret=True):
    """Store a manually obtained auth token.

    Parameters
    ----------
    token :     str
                Your SeaTable auth token from the web UI.
    server :    str, optional
                URL to the SeaTable instance this token belongs to.
                If provided, the token is saved to the matching instance
                secret file.
    save_to_secret : bool
                If True, write the token to the secret store in
                ~/.seaserpent/secrets/.

    Returns
    -------
    str
                The token that was saved.
    """
    if save_to_secret:
        path = save_secret(token, server=server)
        print(f"Saved SeaTable auth token to {path}")
    return token


def get_base_from_token(base_token, server):
    """Authenticate a base directly with a fine-grained base-level API token.

    Unlike the account flow, this skips workspace enumeration and the
    temp-api-token exchange. It calls ``/api/v2.1/dtable/app-access-token/``
    directly, which is the endpoint that accepts base-level (fine-grained)
    API tokens.

    Parameters
    ----------
    base_token :        str
                        A fine-grained, base-level SeaTable API token (read or
                        read+write). Not to be confused with the account token.
    server :            str
                        URL to the SeaTable instance.

    Returns
    -------
    SeaTableAPI
                        An authenticated base object. After ``auth()`` it
                        exposes ``dtable_name``, ``workspace_id`` and
                        ``dtable_uuid``.

    """
    base = SeaTableAPI(base_token, server)
    base.auth()
    return base


def get_account(auth_token=None, server=None):
    """Initialize SeaTable Account.

    Parameters
    ----------
    auth_token :        str, optional
                        Your user's auth token (not the base token). Can either
                        be provided explicitly, stored in a secret file, or set
                        as ``SEATABLE_TOKEN`` environment variable.
    server :            str, optional
                        Must be provided explicitly or set as ``SEATABLE_SERVER``
                        environment variable.

    Returns
    -------
    Account

    """
    if not server:
        server = os.environ.get("SEATABLE_SERVER")

    if not auth_token:
        auth_token = resolve_auth_token(server=server)

    if not server:
        raise ValueError(
            "Must provide `server` or set `SEATABLE_SERVER` environment variable"
        )
    if not auth_token:
        raise ValueError(
            "Must provide `auth_token`, a secret file, or "
            "set `SEATABLE_TOKEN` environment variable"
        )

    account = Account(None, None, server)
    account.token = auth_token

    return account
