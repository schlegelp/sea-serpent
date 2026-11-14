"""Write tests. Only for a throwaway base on a test server (see conftest.py)."""
import datetime as dt
import urllib.parse
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
    set_view(src, "by count",
             [{"column": "id", "filter_predicate": "contains", "filter_term": "n1"}],
             sorts=[{"column_key": src["count"].key, "sort_type": "down"}])

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


def set_view(t, name, filters=(), conjunction="And", **settings):
    """Create view `name` on table `t` (if new) and set its filters, given by
    column name, plus any other view settings (e.g. ``sorts``)."""
    if name not in t.views:
        t.base.add_view(t.name, name)
    filters = [
        {**{k: v for k, v in f.items() if k != "column"}, "column_key": t[f["column"]].key}
        for f in filters
    ]
    endpoints.request(
        t.base, "PUT",
        endpoints.dtable_url(t.base, f"views/{urllib.parse.quote(name, safe='')}/"),
        params={"table_name": t.name},
        json={"filters": filters, "filter_conjunction": conjunction, **settings},
    )
    t.fetch_meta()


@pytest.mark.parametrize(
    "predicate, term, expected",
    [
        ("equal", 4, [4]),
        ("not_equal", "4", [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]),
        ("less", 3, [0, 1, 2]),
        ("greater", "8", [9, 10, 11]),  # the web page stores some terms as text
        ("less_or_equal", "2", [0, 1, 2]),
        ("greater_or_equal", 10, [10, 11]),
        ("greater", 8.5, [9, 10, 11]),
        ("greater", "", list(range(12))),  # no value set: filter is ignored
    ],
)
def test_get_view_number_filters(make_table, df, predicate, term, expected):
    t = make_table(df)
    set_view(t, "v", [{"column": "count", "filter_predicate": predicate, "filter_term": term}])
    got = t.get_view("v")["count"].astype(int)
    assert sorted(got.tolist()) == expected


def test_get_view_combined_number_filters(make_table, df):
    t = make_table(df)
    between = [
        {"column": "count", "filter_predicate": "greater_or_equal", "filter_term": 3},
        {"column": "count", "filter_predicate": "less", "filter_term": "6"},
    ]
    set_view(t, "between", between)
    assert sorted(t.get_view("between")["count"].astype(int)) == [3, 4, 5]
    outside = [
        {"column": "count", "filter_predicate": "less", "filter_term": 2},
        {"column": "count", "filter_predicate": "greater", "filter_term": 9},
    ]
    set_view(t, "outside", outside, "Or")
    assert sorted(t.get_view("outside")["count"].astype(int)) == [0, 1, 10, 11]


def test_get_view_without_filters(make_table, df):
    t = make_table(df)
    got = t.get_view(t.views[0])  # the default view has no filters
    assert sorted(got["count"].astype(int)) == list(range(12))


def test_filters_on_awkward_column_names(make_table):
    """Column names that are SQL keywords or contain spaces must be quoted."""
    t = make_table(pd.DataFrame({
        "id": [f"n{i}" for i in range(6)],
        "group": [1, 2, 3, 4, 5, 6],
        "cell type": ["a", "b", "a", "b", None, "a"],
    }))
    assert len(t.loc[t["group"] > 3]) == 3
    assert len(t.loc[t["group"] == 2]) == 1
    assert len(t.loc[t["cell type"] == "a"]) == 3
    assert len(t.loc[t["cell type"].isin(["a", "b"])]) == 5
    assert len(t.loc[t["cell type"].isnull()]) == 1
    assert len(t.loc[t["cell type"].contains("b")]) == 2
    assert 4 in t["group"]
    set_view(t, "v", [{"column": "group", "filter_predicate": "less_or_equal", "filter_term": "2"}])
    assert sorted(t.get_view("v")["group"].astype(int)) == [1, 2]


@pytest.fixture
def tagged():
    tags = [["x"], ["y"], ["x", "y"], None, ["it's"], ["x", "it's"]]
    return pd.DataFrame({"id": [f"n{i}" for i in range(6)], "tags": tags})


def tag_ids(t, *names):
    """Option IDs (what a view stores) for the given option names."""
    opts = {o["name"]: o["id"] for o in t["tags"].meta["data"]["options"]}
    return [opts[n] for n in names]


@pytest.mark.parametrize(
    "predicate, names, expected",
    [
        ("has_any_of", ["y", "it's"], ["n1", "n2", "n4", "n5"]),
        ("has_all_of", ["x", "y"], ["n2"]),
        ("has_none_of", ["x"], ["n1", "n3", "n4"]),  # includes the empty row
        ("is_exactly", ["x"], ["n0"]),
        ("is_exactly", ["x", "it's"], ["n5"]),
    ],
)
def test_get_view_multi_select_filters(make_table, tagged, predicate, names, expected):
    t = make_table(tagged)
    set_view(t, "v", [{"column": "tags", "filter_predicate": predicate,
                       "filter_term": tag_ids(t, *names)}])
    assert sorted(t.get_view("v")["id"]) == expected


def test_get_view_multi_select_filters_kept_apart(make_table, tagged):
    """Two 'any of' filters joined by AND: each needs a match on its own."""
    t = make_table(tagged)
    set_view(t, "v", [
        {"column": "tags", "filter_predicate": "has_any_of", "filter_term": tag_ids(t, "x")},
        {"column": "tags", "filter_predicate": "has_any_of", "filter_term": tag_ids(t, "it's")},
    ], "And")
    assert sorted(t.get_view("v")["id"]) == ["n5"]


def test_get_view_skips_filters_without_value(make_table, tagged):
    t = make_table(tagged)
    set_view(t, "v", [
        {"column": "id", "filter_predicate": "contains", "filter_term": ""},
        {"column": "id", "filter_predicate": "is", "filter_term": ""},
        {"column": "tags", "filter_predicate": "has_any_of", "filter_term": []},
        {"column": "id", "filter_predicate": "is_not", "filter_term": "n0"},
    ], "And")
    assert sorted(t.get_view("v")["id"]) == ["n1", "n2", "n3", "n4", "n5"]


def test_get_view_deleted_option(make_table, tagged):
    """A filter that points at an option ID that no longer exists matches nothing."""
    t = make_table(tagged)
    set_view(t, "v", [{"column": "tags", "filter_predicate": "has_any_of",
                       "filter_term": ["999999"]}])
    assert len(t.get_view("v")) == 0


def test_get_view_sorted_by_hidden_column(make_table, df):
    t = make_table(df)
    set_view(t, "v", [{"column": "count", "filter_predicate": "less", "filter_term": 4}],
             hidden_columns=[t["count"].key],
             sorts=[{"column_key": t["count"].key, "sort_type": "down"}])
    got = t.get_view("v")
    assert "count" not in got.columns
    assert got["id"].tolist() == ["n3", "n2", "n1", "n0"]


def test_multi_select_column_methods(make_table, tagged):
    t = make_table(tagged)
    tags = t["tags"]

    def ids(f):
        return sorted(t.loc[f, ["id"]]["id"])

    assert ids(tags.has_any_of(["y", "it's"])) == ["n1", "n2", "n4", "n5"]
    assert ids(tags.has_all_of(["x", "y"])) == ["n2"]
    assert ids(tags.has_none_of("x")) == ["n1", "n3", "n4"]
    assert ids(tags.is_exactly(["x", "it's"])) == ["n5"]
    assert ids(~tags.has_any_of(["x"])) == ids(tags.has_none_of(["x"]))
    with pytest.raises(ValueError):
        ~tags.is_exactly(["x"])
    with pytest.raises(TypeError):
        t["id"].has_any_of(["x"])


def test_text_values_with_quotes(make_table):
    t = make_table(pd.DataFrame({"id": ["it's", "plain", "a=b"], "n": [1, 2, 3]}))
    assert t.loc[t["id"] == "it's", ["n"]]["n"].tolist() == [1]
    assert t.loc[t["id"] != "a=b", ["n"]]["n"].astype(int).tolist() == [1, 2]
    assert sorted(t.loc[t["id"].isin(np.array(["it's", "plain"])), ["n"]]["n"].astype(int)) == [1, 2]
    assert t.loc[t["id"].contains("t's"), ["n"]]["n"].tolist() == [1]
    assert "it's" in t["id"]


def test_get_view_is_filters_follow_conjunction(make_table, df):
    t = make_table(df)
    two_is = [{"column": "id", "filter_predicate": "is", "filter_term": "n1"},
              {"column": "id", "filter_predicate": "is", "filter_term": "n2"}]
    set_view(t, "and", two_is, "And")
    assert len(t.get_view("and")) == 0  # one cell can't be both
    set_view(t, "or", two_is, "Or")
    assert sorted(t.get_view("or")["id"]) == ["n1", "n2"]
    two_not = [{"column": "id", "filter_predicate": "is_not", "filter_term": "n1"},
               {"column": "id", "filter_predicate": "is_not", "filter_term": "n2"}]
    set_view(t, "or_not", two_not, "Or")
    assert len(t.get_view("or_not")) == 12  # every row differs from n1 or from n2


def test_get_view_select_contains_no_match(make_table, df):
    t = make_table(df)
    set_view(t, "v", [{"column": "kind", "filter_predicate": "contains", "filter_term": "zzz"}])
    assert len(t.get_view("v")) == 0
    set_view(t, "w", [{"column": "kind", "filter_predicate": "does_not_contain", "filter_term": "zzz"}])
    assert len(t.get_view("w")) == 12


def test_get_view_unsupported_filters_raise_value_error(make_table, df):
    t = make_table(df)
    set_view(t, "bad_number", [{"column": "count", "filter_predicate": "equal", "filter_term": "1:30"}])
    with pytest.raises(ValueError, match="1:30"):
        t.get_view("bad_number")
    # has_any_of on a column that isn't multiple-select (e.g. collaborators)
    set_view(t, "bad_type", [{"column": "id", "filter_predicate": "has_any_of", "filter_term": ["n1"]}])
    with pytest.raises(ValueError, match="Unsupported"):
        t.get_view("bad_type")


def test_awkward_table_name(server, rw_token, base_uuid):
    name = f"sstest group {dt.datetime.now():%H%M%S%f}"
    t = ss.Table.from_frame(pd.DataFrame({"id": ["a", "b"], "n": [1, 2]}),
                            table_name=name, base=base_uuid, auth_token=rw_token, server=server)
    t.progress = False
    try:
        assert t.shape == (2, 2)
        assert len(t.head()) == 2
        assert len(t.to_frame()) == 2
        assert t.loc[t["n"] > 1, ["id"]]["id"].tolist() == ["b"]
        assert "a" in t["id"]
    finally:
        t.delete(skip_confirmation=True)
