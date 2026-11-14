import re

import seatable_api

import pandas as pd

#: Trailing UTC offset in an ISO timestamp: "Z", "+01:00", "-0500".
_TZ_OFFSET = re.compile(r'(?:[Zz]|[+-]\d{2}:?\d{2})$')


def convert_db_rows(metadata, results, utc=False):
    """Convert dtable-db rows data to readable rows data.

    This is a modified version of the original conversion function that uses
    pandas.to_datetime which deals correctly with time-zoned dates (plus a
    few other edits). Date columns are converted in one vectorised pass per
    column rather than cell by cell, which is ~10x faster overall.

    Parameters
    ----------
    metadata :  list
                The table's meta data.
    results :   list
                The raw query results.
    utc :       bool
                If True all dates will be converted to UTC+0. If False, dates
                are returned "as is".

    Returns
    -------
    list
                List of records.

    """
    if not results:
        return []
    converted_results = []
    column_map = {column['key']: column for column in metadata}
    select_map = {}
    for column in metadata:
        column_type = column['type']
        if column_type in ('single-select', 'multiple-select'):
            column_data = column['data']
            if not column_data:
                continue
            column_key = column['key']
            column_options = column['data']['options']
            select_map[column_key] = {
                select['id']: select['name'] for select in column_options}

    # Pre-convert each date column in a single vectorised pass. Calling
    # pd.to_datetime() once per *cell* makes pandas re-infer the datetime
    # format from scratch every time (~130us/cell) - on a 58k-row table that
    # was 7.5s, i.e. 94% of this function's runtime.
    date_cache = {}
    for column in metadata:
        if column['type'] != 'date':
            continue
        column_key = column['key']
        try:
            fmt = ('%Y-%m-%d' if column['data']['format'] == 'YYYY-MM-DD'
                   else '%Y-%m-%d %H:%M:%S')
        except Exception:
            continue  # no usable format -> leave the raw value alone

        raw = [r.get(column_key) for r in results]
        if utc:
            values = raw
        else:
            # One column can carry several UTC offsets (this data mixes
            # "+01:00" and "Z"). Parsing those together with utc=False makes
            # pandas bail out to object dtype - it emits a FutureWarning today
            # and will raise outright in a future release. With utc=False we
            # only ever render local wall-clock time, for which the offset is
            # irrelevant, so strip it and parse tz-naive: identical output, no
            # warning, and it stays on the fast datetime64 path.
            values = [_TZ_OFFSET.sub('', v) if isinstance(v, str) else v
                      for v in raw]

        try:
            # format='ISO8601' covers every shape dtable-db emits (with or
            # without "T", date-only, fractional seconds) and stops pandas
            # trying to infer one, which warns when the column holds anything
            # it cannot make sense of.
            parsed = pd.to_datetime(pd.Series(values, dtype='object'),
                                    utc=utc, format='ISO8601', errors='coerce')
        except Exception:
            parsed = None

        if parsed is None:
            formatted = [None] * len(raw)
        else:
            try:
                formatted = parsed.dt.strftime(fmt).tolist()
            except AttributeError:
                # Defensive: some input still landed us on object dtype.
                # Parsing is already done, so only formatting is left.
                formatted = [v.strftime(fmt) if isinstance(v, pd.Timestamp)
                             else None for v in parsed]

        # Retry anything ISO8601 could not handle one value at a time, the way
        # the old per-cell code did (scalar parsing falls back to dateutil and
        # does not warn). Normally this loop finds nothing to do.
        for i, (v, f) in enumerate(zip(raw, formatted)):
            if v and (f is None or f != f):
                try:
                    formatted[i] = pd.to_datetime(v, utc=utc).strftime(fmt)
                except Exception:
                    pass

        # NaT/failure -> keep the original value, as the old try/except did
        date_cache[column_key] = [v if (f is None or f != f) else f
                                  for v, f in zip(raw, formatted)]

    for i, result in enumerate(results):
        item = {}
        for column_key, value in result.items():
            if column_key in column_map:
                column = column_map[column_key]
                column_name = column['name']
                column_type = column['type']

                # Pass through empty values
                # Note that this also catches empty strings ('') and `False`
                if not value:
                    item[column_name] = value
                    continue

                s_map = select_map.get(column_key)
                if column_type == 'single-select' and s_map:
                    item[column_name] = s_map.get(value, value)
                elif column_type == 'multiple-select' and s_map:
                    item[column_name] = [s_map.get(s, s) for s in value]
                elif column_type == 'date':
                    cached = date_cache.get(column_key)
                    item[column_name] = value if cached is None else cached[i]
                else:
                    item[column_name] = value
            else:
                item[column_key] = value
        converted_results.append(item)

    return converted_results


# sea-serpent itself sends SQL through `endpoints.run_sql`, which always uses
# the function above. These patches only matter for code that calls
# `base.query()` from seatable-api directly.
seatable_api.utils.convert_db_rows = convert_db_rows
seatable_api.main.convert_db_rows = convert_db_rows

Account = seatable_api.main.Account
SeaTableAPI = seatable_api.main.SeaTableAPI
