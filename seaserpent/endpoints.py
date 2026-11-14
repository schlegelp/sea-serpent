"""Server addresses and the SQL call, for old and new SeaTable servers.

SeaTable 5.0 added the API gateway (``/api-gateway/``). Old servers (and old
cloud.seatable.io setups) only have ``/dtable-server/`` and ``/dtable-db/``.
From 5.3 the old addresses are on their way out. The server tells us which
one to use when a base logs in (``use_api_gateway`` from
``/api/v2.1/dtable/app-access-token/``), and ``seatable_api.SeaTableAPI``
keeps that answer. So we go by that flag rather than by version numbers.

All requests are sent with the base's own headers (``Token <base JWT>``),
which both the old addresses and the gateway accept.
"""
import http.cookiejar
import threading

import requests

from seatable_api.exception import AuthExpiredError

from .patch import convert_db_rows

#: One requests.Session per thread (requests doesn't promise a Session is
#: thread-safe), so repeated calls, including the pages that `Table._query_all`
#: fetches in parallel, reuse connections instead of opening new ones.
_local = threading.local()


def _session():
    s = getattr(_local, "session", None)
    if s is None:
        s = _local.session = requests.Session()
        # Keep no cookies: auth is by header only
        s.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
    return s


class APIError(requests.HTTPError, ConnectionError):
    """A request to the server failed.

    Both a ``requests.HTTPError`` (what sea-serpent raised for these calls
    before) and a ``ConnectionError`` (what seatable-api raises), so existing
    ``except`` clauses for either keep working.
    """


def uses_gateway(base):
    """Whether this base talks to the server through the API gateway."""
    return bool(base.use_api_gateway)


def dtable_url(base, path=""):
    """Address of a per-base endpoint.

    Parameters
    ----------
    base :  seatable_api.SeaTableAPI
            An authenticated base.
    path :  str
            The part after ``.../dtables/<uuid>/``, e.g. ``"related-users/"``.

    Returns
    -------
    str
            ``<server>/api-gateway/api/v2/dtables/<uuid>/<path>`` on new
            servers, ``<dtable-server>/api/v1/dtables/<uuid>/<path>`` on old
            ones.

    """
    path = path.lstrip("/")
    if uses_gateway(base):
        return f"{base.api_gateway.api_gateway_url}/api/v2/dtables/{base.dtable_uuid}/{path}"
    return f"{base.dtable_server_url}/api/v1/dtables/{base.dtable_uuid}/{path}"


def sql_url(base):
    """Address for SQL queries."""
    if uses_gateway(base):
        return dtable_url(base, "sql")
    return f"{base.dtable_db_url}/api/v1/query/{base.dtable_uuid}/"


def request(base, method, url, timeout=None, **kwargs):
    """Send a request with the base's headers and return the parsed JSON.

    Raises ``AuthExpiredError`` for an expired token (as seatable-api does)
    and `APIError` for any other failed request or a reply that isn't JSON.
    """
    r = _session().request(
        method,
        url,
        headers=base.headers,
        timeout=timeout or base.timeout,
        **kwargs,
    )
    if r.status_code >= 400:
        if r.status_code == 403 and "Token expired" in r.text:
            raise AuthExpiredError
        raise APIError(f"{r.status_code} for {method} {r.url}: {r.text[:500]}", response=r)
    try:
        return r.json()
    except ValueError:
        raise APIError(
            f"Reply to {method} {r.url} is not JSON: {r.text[:200]!r}", response=r
        ) from None


def run_sql(base, sql, convert=True, timeout=None):
    """Run an SQL query against a base.

    We send the request ourselves instead of using ``base.query()``:
    ``seatable-api`` converts the results differently depending on its version
    and on whether the gateway is used (from 4.0.0 the gateway path lets the
    server convert and drops the column info). Here the server always returns
    raw values keyed by column key, plus the column info, and we convert them
    with our own :func:`seaserpent.patch.convert_db_rows`. That gives the same
    output on every server and ``seatable-api`` version.

    Parameters
    ----------
    base :      seatable_api.SeaTableAPI
                An authenticated base.
    sql :       str
                The query.
    convert :   bool
                If True, turn column keys into names, select option ids into
                option names, and format dates. If False, return the raw rows
                (keyed by column key).

    Returns
    -------
    list of dict

    """
    if not sql:
        raise ValueError("sql can not be empty.")
    # Ask for raw rows keyed by column key; convert_db_rows needs those
    data = request(
        base, "POST", sql_url(base), json={"sql": sql, "convert_keys": False},
        timeout=timeout,
    )
    if not isinstance(data, dict):
        raise APIError(f"Unexpected SQL reply: {str(data)[:200]}")
    if not data.get("success"):
        raise Exception(data.get("error_message"))
    results = data.get("results") or []
    if convert:
        if "metadata" not in data:
            raise APIError("SQL reply has no column info, so rows can't be converted")
        return convert_db_rows(data["metadata"], results)
    return results
