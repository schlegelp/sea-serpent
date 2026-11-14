"""Read-only tests. Safe against any server (light load: a few requests)."""
import pandas as pd

from seaserpent import endpoints


def test_server_info(read_table):
    info = read_table.server_info
    assert "version" in info


def test_shape_and_columns(read_table):
    n, m = read_table.shape
    assert n >= 0 and m == len(read_table.columns) > 0


def test_head(read_table):
    df = read_table.head(3)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == list(read_table.columns)


def test_query_raw_and_converted(read_table):
    col = read_table.columns[0]
    conv = read_table.query(f"SELECT `{col}` FROM {read_table.name} LIMIT 2")
    raw = read_table.query(f"SELECT `{col}` FROM {read_table.name} LIMIT 2", convert=False)
    assert len(conv) == len(raw)
    if conv:
        assert col in conv[0]  # converted rows use column names ...
        assert read_table[col].key in raw[0]  # ... raw rows use column keys


def test_to_frame_matches_shape(read_table):
    df = read_table.to_frame()
    assert df.shape == read_table.shape


def test_paging_across_pages(read_table, monkeypatch):
    """Force small pages (about 5 per table) so the paging code runs,
    sequentially and in parallel."""
    import seaserpent.base as sb

    n = read_table.shape[0]
    monkeypatch.setattr(sb, "MAX_ROWS_PER_QUERY", max(7, n // 4 + 1))
    for workers in (1, 3):
        rows = read_table.query("SELECT _id", no_limit=True, workers=workers)
        assert len(rows) == n
        assert len({r["_id"] for r in rows}) == n


def test_collaborators(read_table):
    c = read_table.collaborators
    assert isinstance(c, pd.DataFrame)
    assert "name" in c.columns


def test_fetch_logs(read_table):
    logs = read_table.fetch_logs(max_entries=5, progress=False)
    assert isinstance(logs, pd.DataFrame)


def test_fetch_row_logs(read_table):
    rid = read_table.query(f"SELECT _id FROM {read_table.name} LIMIT 1")[0]["_id"]
    logs = read_table.fetch_row_logs(rid, max_entries=3, progress=False)
    assert isinstance(logs, pd.DataFrame)


def test_gateway_flag_matches_server(read_table):
    """6.x servers must be reached through the gateway, 3.x without it."""
    major = int(read_table.server_info["version"].split(".")[0])
    assert endpoints.uses_gateway(read_table.base) == (major >= 5)
