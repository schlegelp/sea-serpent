"""Offline tests for data clean-up."""
import pandas as pd

from seaserpent.utils import process_records


def test_numbers_sent_as_text():
    """dtable-db sends some auto-numbers as text ("20870") and some as numbers."""
    records = [{"serial_id": 13765}, {"serial_id": "20870"}, {"serial_id": None}]
    df = process_records(records, dtypes={"serial_id": pd.Int64Dtype()})
    assert df["serial_id"].dtype == pd.Int64Dtype()
    assert df["serial_id"].tolist()[:2] == [13765, 20870]


def test_text_that_isnt_a_number_is_kept():
    records = [{"n": 1}, {"n": "ID-2"}]
    df = process_records(records, dtypes={"n": pd.Int64Dtype()})
    assert df["n"].tolist() == [1, "ID-2"]


def test_large_numbers_sent_as_text_stay_exact():
    records = [{"n": "9007199254740993"}, {"n": 1}, {"n": None}]
    df = process_records(records, dtypes={"n": pd.Int64Dtype()})
    assert df["n"].tolist()[:2] == [9007199254740993, 1]


def test_invert_ignores_operators_inside_values():
    import pytest
    from seaserpent.base import Filter

    assert (~Filter("`a` LIKE '%x = y%'")).query == "`a` NOT LIKE '%x = y%'"
    assert (~Filter("`a` = 'rock AND roll'")).query == "`a` != 'rock AND roll'"
    assert (~Filter("`my IN col` IS NULL")).query == "`my IN col` IS NOT NULL"
    assert (~Filter("`t` HAS ANY OF ('R AND B')")).query == "`t` HAS NONE OF ('R AND B')"
    with pytest.raises(ValueError):
        ~Filter("`t` IS EXACTLY ('a')")
    with pytest.raises(ValueError):
        ~Filter("(`a` = 1) AND (`b` = 2)")
