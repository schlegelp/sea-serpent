import seatable_api

import pandas as pd


def convert_db_rows(metadata, results, utc=False):
    """Convert dtable-db rows data to readable rows data.

    This is a modified version of the original conversion function that uses
    pandas.to_datetime which deals correctly with time-zoned dates (plus a
    few other edits).

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

    for result in results:
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
                    try:
                        date_value = pd.to_datetime(value, utc=utc)
                        date_format = column['data']['format']
                        if date_format == 'YYYY-MM-DD':
                            value = date_value.strftime('%Y-%m-%d')
                        else:
                            value = date_value.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception as e:
                        pass
                    item[column_name] = value
                else:
                    item[column_name] = value
            else:
                item[column_key] = value
        converted_results.append(item)

    return converted_results


seatable_api.utils.convert_db_rows = convert_db_rows
seatable_api.main.convert_db_rows = convert_db_rows

Account = seatable_api.main.Account
SeaTableAPI = seatable_api.main.SeaTableAPI
