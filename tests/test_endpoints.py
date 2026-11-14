"""Offline tests: which address is used for old and new servers."""
from types import SimpleNamespace

from seaserpent import endpoints


def old_base():
    # What seatable-api keeps after logging in to a 3.x server
    return SimpleNamespace(
        server_url="https://example.org",
        dtable_server_url="https://example.org/dtable-server",
        dtable_db_url="https://example.org/dtable-db",
        dtable_uuid="abc",
        use_api_gateway=None,
        api_gateway=None,
    )


def new_base():
    # 6.2.1 reports dtable_server = <server>/api-gateway/ and no dtable_db
    return SimpleNamespace(
        server_url="https://example.org",
        dtable_server_url="https://example.org/api-gateway",
        dtable_db_url="",
        dtable_uuid="abc",
        use_api_gateway=True,
        api_gateway=SimpleNamespace(api_gateway_url="https://example.org/api-gateway"),
    )


def test_old_server_addresses():
    b = old_base()
    assert not endpoints.uses_gateway(b)
    assert (
        endpoints.dtable_url(b, "related-users/")
        == "https://example.org/dtable-server/api/v1/dtables/abc/related-users/"
    )
    assert endpoints.sql_url(b) == "https://example.org/dtable-db/api/v1/query/abc/"


def test_new_server_addresses():
    b = new_base()
    assert endpoints.uses_gateway(b)
    assert (
        endpoints.dtable_url(b, "/operations/")
        == "https://example.org/api-gateway/api/v2/dtables/abc/operations/"
    )
    assert endpoints.sql_url(b) == "https://example.org/api-gateway/api/v2/dtables/abc/sql"


def test_convert_db_rows_mixed_offsets():
    """From SeaTable 4.4 SQL dates carry +01:00 or Z, sometimes both in one column."""
    from seaserpent.patch import convert_db_rows

    meta = [{"key": "d1", "name": "when", "type": "date",
             "data": {"format": "YYYY-MM-DD HH:mm"}}]
    rows = [{"d1": "2022-09-14T10:30:00+01:00"}, {"d1": "2022-11-22T10:30:00Z"}, {"d1": None}]
    out = convert_db_rows(meta, rows)
    assert [r["when"] for r in out] == ["2022-09-14 10:30:00", "2022-11-22 10:30:00", None]


class FakeResponse:
    def __init__(self, status, body, url="https://example.org/x"):
        self.status_code, self.text, self.url = status, body, url

    def json(self):
        import json
        return json.loads(self.text)


def fake_session(monkeypatch, response, seen=None):
    def request(method, url, **kwargs):
        if seen is not None:
            seen.append(kwargs)
        return response
    monkeypatch.setattr(endpoints._session(), "request", request)


def test_request_errors(monkeypatch):
    import pytest
    import requests
    from seatable_api.exception import AuthExpiredError

    b = new_base()
    b.headers, b.timeout = {}, 30
    fake_session(monkeypatch, FakeResponse(404, "not found"))
    with pytest.raises(requests.HTTPError):  # what sea-serpent raised before
        endpoints.request(b, "GET", "https://example.org/x")
    with pytest.raises(ConnectionError):  # what seatable-api raises
        endpoints.request(b, "GET", "https://example.org/x")
    fake_session(monkeypatch, FakeResponse(403, '{"error_msg": "Token expired."}'))
    with pytest.raises(AuthExpiredError):
        endpoints.request(b, "GET", "https://example.org/x")
    fake_session(monkeypatch, FakeResponse(200, "<html>proxy page</html>"))
    with pytest.raises(endpoints.APIError, match="not JSON"):
        endpoints.request(b, "GET", "https://example.org/x")


def test_run_sql_asks_for_raw_keys(monkeypatch):
    import pytest

    b = new_base()
    b.headers, b.timeout = {}, 30
    seen = []
    body = '{"success": true, "results": [{"0000": "a"}], "metadata": [{"key": "0000", "name": "id", "type": "text"}]}'
    fake_session(monkeypatch, FakeResponse(200, body), seen)
    assert endpoints.run_sql(b, "SELECT id FROM t") == [{"id": "a"}]
    assert seen[0]["json"]["convert_keys"] is False
    # No column info -> refuse to convert rather than return wrong rows
    fake_session(monkeypatch, FakeResponse(200, '{"success": true, "results": [{"id": "a"}]}'))
    with pytest.raises(endpoints.APIError):
        endpoints.run_sql(b, "SELECT id FROM t")
    assert endpoints.run_sql(b, "SELECT id FROM t", convert=False) == [{"id": "a"}]
