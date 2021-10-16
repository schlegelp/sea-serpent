import pandas as pd
import numpy as np

import requests
import numbers

def process_records(records, columns=None):
    """Turn records into pandas DataFrame.

    Parameters
    ----------
    records :   list
                The records (e.g. from an SQL query).
    columns :   list
                Columns. If provided will filter records to only these columns
                but also fill missing columns with ``None``.

    Returns
    -------
    DataFrame

    """
    assert isinstance(records, list)

    if not isinstance(columns, type(None)):
        records = [{c: r.get(c, None) for c in columns} for r in records]

    df = pd.DataFrame.from_records(records)

    if not isinstance(columns, type(None)):
        df = df[columns]

    return df


def get_auth_token(username, password, server='https://cloud.seatable.io'):
    """Retriever your user's auth token."""
    while server.endswith('/'):
        server = server[:-1]

    url = f'{server}/api2/auth-token/'
    post = {'username': username, 'password': password}

    r = requests.post(url, data=post)

    r.raise_for_status()

    return r.json()


def is_iterable(x):
    """Return True if `x` is iterable (container)."""
    if isinstance(x, (list, np.ndarray, set, tuple)):
        return True
    return False


def validate_comparison(column, other, allow_iterable=False):
    """Raise TypeError if comparison not allowed."""
    if not is_iterable(other):
        if column.dtype in ('text', 'long-text'):
            if isinstance(other, str):
                return True
        elif column.dtype == 'number':
            if isinstance(other, numbers.Number):
                return True
        elif column.dtype == 'checkbox':
            if isinstance(other, bool):
                return True
        else:
            # For unknown column types assume we're good
            return True
    elif allow_iterable:
        # Check individual items in container
        return [validate_comparison(column, o, allow_iterable=False) for o in other]

    # If we haven't returned yet, throw hissy fit
    raise TypeError(f'Unable to compare column of type "{column.dtype}" with "{type(other)}"')
