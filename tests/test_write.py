"""Write tests. Only for a throwaway base on a test server (see conftest.py)."""
import datetime as dt
import time

import numpy as np
import pandas as pd
import pytest

import seaserpent as ss
from seaserpent import endpoints


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "id": [f"n{i}" for i in range(12)],
            "count": np.arange(12, dtype=int),
            "score": np.linspace(0, 1, 12),
            "flag": [i % 2 == 0 for i in range(12)],
            "kind": pd.Categorical(["a", "b", "c"] * 4),
            "tags": [["x"], ["y"], ["x", "y"], None] * 3,
        }
    )


def test_from_frame_roundtrip(make_table, df):
    t = make_table(df)
    assert t.shape == (12, 6)
    got = t.to_frame().sort_values("count").reset_index(drop=True)
    assert got["id"].tolist() == df["id"].tolist()
    assert got["count"].tolist() == df["count"].tolist()
    assert np.allclose(got["score"].astype(float), df["score"])
    assert got["flag"].tolist() == df["flag"].tolist()
    assert got["kind"].astype(str).tolist() == df["kind"].astype(str).tolist()
    # to_frame() turns multiple-select into text on purpose; check the lists via SQL
    rows = t.query(f"SELECT id, tags FROM {t.name}", no_limit=True)
    by_id = {r["id"]: r["tags"] for r in rows}
    tags = [sorted(by_id[i]) if by_id[i] else None for i in df["id"]]
    want = [sorted(v) if isinstance(v, list) else None for v in df["tags"]]
    assert tags == want


def test_setitem_and_loc(make_table, df):
    t = make_table(df)
    t["count"] = list(range(100, 112))
    assert sorted(t["count"].values.astype(int).tolist()) == list(range(100, 112))
    t.loc[t["id"] == "n3", "score"] = 9.5
    assert t.query(f"SELECT score FROM {t.name} WHERE id = 'n3'")[0]["score"] == 9.5


def test_append_and_delete_rows(make_table, df):
    t = make_table(df)
    t.append(pd.DataFrame({"id": ["extra1", "extra2"], "count": [50, 51]}))
    assert t.shape[0] == 14
    ids = [r["_id"] for r in t.query(f"SELECT _id FROM {t.name} WHERE count >= 50")]
    # Row IDs as a pandas Series (string dtype under pandas 3)
    t.delete_rows(pd.Series(ids, dtype="string"), skip_confirmation=True)
    assert t.shape[0] == 12
    # An empty selection is refused rather than taken as "no row IDs"
    with pytest.raises(TypeError):
        t.delete_rows(pd.Series([], dtype=object), skip_confirmation=True)
    # Row IDs with a gap are refused, not turned into the text "<NA>"
    with pytest.raises(ValueError, match="missing"):
        t.delete_rows(pd.Series([ids[0], None], dtype="string"), skip_confirmation=True)


def test_column_operations(make_table, df):
    t = make_table(df)
    t.add_column("notes", str)
    t.add_column("when", "date")
    t.add_column("choice", "single_select",
                 col_options=[{"name": "p", "color": "#aaa", "textColor": "#000000"}])
    t.fetch_meta()
    assert {"notes", "when", "choice"} <= set(t.columns)

    t["notes"].rename("comments")
    t["comments"].resize(300)
    t["comments"].freeze()
    t["comments"].unfreeze()
    t["choice"].add_options([{"name": "q", "color": "#bbb", "textColor": "#000000"}])
    assert {"p", "q"} <= set(t["choice"].options)
    t["comments"].delete()
    t.fetch_meta()
    assert "comments" not in t.columns


def test_dates_roundtrip(make_table, df):
    """Dates written and read back unchanged (SQL returns them with +01:00/Z)."""
    t = make_table(df)
    t.add_column("when", "date")
    t.fetch_meta()
    # Pick one winter and one summer day so both UTC offsets show up
    days = ["2026-01-15", "2026-07-15"] * 6
    t["when"] = days
    got = t.query(f"SELECT id, `when` FROM {t.name}", no_limit=True)
    by_id = {r["id"]: r["when"] for r in got}
    assert [by_id[i] for i in df["id"]] == days
    # Dates from a pandas text column with gaps (pd.NA / NaN for missing)
    t["when"] = pd.Series(["2026-03-01"] + [None] * 11, dtype="string")
    got = t.query(f"SELECT id, `when` FROM {t.name}", no_limit=True)
    by_id = {r["id"]: r["when"] for r in got}
    assert by_id["n0"] == "2026-03-01" and by_id["n1"] is None
    raw = t.query(f"SELECT `when` FROM {t.name} LIMIT 2", convert=False)
    print("raw date values from the server:", raw)


def test_link_and_linked_column(make_table, df):
    a = make_table(df)
    b = make_table(pd.DataFrame({"id": [f"n{i}" for i in range(0, 12, 2)],
                                 "label": [f"L{i}" for i in range(6)]}))
    a.link(b.name, link_on="id", link_on_other="id", link_col="to_b")
    a.add_linked_column("b_label", link_col="to_b", link_on="label")
    a.fetch_meta()
    assert {"to_b", "b_label"} <= set(a.columns)
    rows = a.query(f"SELECT id, to_b FROM {a.name}", no_limit=True)
    linked = {r["id"] for r in rows if r["to_b"]}
    assert linked == set(b["id"].values.astype(str))
    print("link cell as returned by SQL:", rows[0]["to_b"])


def test_copy_table_with_views(make_table, df, server, rw_token):
    src = make_table(df)
    # Give the source an extra view with a sort and a filter
    src.base.add_view(src.name, "by count")
    count_key = src["count"].key
    endpoints.request(
        src.base, "PUT",
        endpoints.dtable_url(src.base, "views/by%20count/"),
        params={"table_name": src.name},
        json={"sorts": [{"column_key": count_key, "sort_type": "down"}],
              "filters": [{"column_key": src["id"].key, "filter_predicate": "contains",
                           "filter_term": "n1"}]},
    )
    src.fetch_meta()

    copy_name = f"{src.name}_copy"
    dst = ss.Table.from_frame(src, table_name=copy_name, base=src.base,
                              auth_token=rw_token, server=server)
    try:
        dst.fetch_meta()
        assert "by count" in dst.views
        view = next(v for v in dst.meta["views"] if v["name"] == "by count")
        assert view["sorts"][0]["sort_type"] == "down"
        assert view["sorts"][0]["column_key"] == dst["count"].key
        assert view["filters"][0]["filter_term"] == "n1"
        assert view["filters"][0]["column_key"] == dst["id"].key
        # n1, n10, n11, sorted by count, largest first
        assert dst.get_view("by count")["count"].astype(int).tolist() == [11, 10, 1]
    finally:
        dst.delete(skip_confirmation=True)


def test_logs_after_writes(make_table, df):
    t = make_table(df)
    t.loc[t["id"] == "n1", "count"] = 77
    rid = t.query(f"SELECT _id FROM {t.name} WHERE id = 'n1'")[0]["_id"]
    # The server lists new edits in its logs only after a short delay
    logs = row_logs = []
    for _ in range(30):
        if not len(logs):
            logs = t.fetch_logs(max_entries=5, progress=False)
        if not len(row_logs):
            row_logs = t.fetch_row_logs(rid, max_entries=5, progress=False)
        if len(logs) and len(row_logs):
            break
        time.sleep(1)
    assert len(logs) > 0
    assert len(row_logs) > 0


def test_delete_table(server, rw_token, base_uuid):
    name = f"sstest_del_{dt.datetime.now():%H%M%S%f}"
    t = ss.Table.new(name, base=base_uuid, auth_token=rw_token, server=server)
    t.delete(skip_confirmation=True)
    names = [x["name"] for x in t.base.get_metadata()["tables"]]
    assert name not in names


def test_logs_empty_history(make_table, df, monkeypatch):
    """No history (e.g. all older than the server keeps) gives an empty frame."""
    t = make_table(df)
    rid = t.query(f"SELECT _id FROM {t.name} LIMIT 1")[0]["_id"]
    from seaserpent import base as sb
    monkeypatch.setattr(sb, "api_request",
                        lambda *a, **k: {"operations": [], "activities": []})
    assert t.fetch_logs(max_entries=5, progress=False).empty
    assert t.fetch_logs(max_entries=5, unpack=False, progress=False).empty
    assert t.fetch_row_logs(rid, max_entries=5, progress=False).empty
