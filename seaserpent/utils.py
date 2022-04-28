import functools
import numbers
import os
import requests
import sys
import logging

import datetime as dt
import numpy as np
import pandas as pd

from seatable_api import Account
from seatable_api.constants import ColumnTypes


logger = logging.getLogger(__name__)
logger.setLevel('INFO')

COLUMN_TYPES = {
    (int, float, 'i', 'u', 'int', 'f', 'float', 'number'): ColumnTypes.NUMBER,  # number
    (str, 'S', 'str', 'text'): ColumnTypes.TEXT,                                # text
    ('long text', ): ColumnTypes.LONG_TEXT,                                     # long text
    (bool, 'b', 'bool', 'checkbox'): ColumnTypes.CHECKBOX,                      # checkbox
    ('date', ): ColumnTypes.DATE,                                               # date & time
    ('select', 'single_select',
     pd.CategoricalDtype): ColumnTypes.SINGLE_SELECT,                           # single select
    ('multiple_select', ): ColumnTypes.MULTIPLE_SELECT,                         # multiple select
    ('image', ): ColumnTypes.IMAGE,                                             # image
    ('file', ): ColumnTypes.FILE,                                               # file
    ('collaborator', ): ColumnTypes.COLLABORATOR,                               # collaborator
    ('link', ): ColumnTypes.LINK,                                               # link to other records
    ('link-formula', ): ColumnTypes.LINK_FORMULA,                               # pull values from other table via link
    ('formula', ): ColumnTypes.FORMULA,                                         # formula
    ('creator', ): ColumnTypes.CREATOR,                                         # creator
    ('ctime', ): ColumnTypes.CTIME,                                             # create time
    ('last_modifier', ): ColumnTypes.LAST_MODIFIER,                             # last modifier
    ('mtime', ): ColumnTypes.MTIME,                                             # modify time
    ('location', 'geolocation'): ColumnTypes.GEOLOCATION,                       # geolocation
    ('auto_number', ): ColumnTypes.AUTO_NUMBER,                                 # auto munber
    ('url', ): ColumnTypes.URL,                                                 # URL
}

NUMERIC_TYPES = (numbers.Number, int, float,
                 np.int0, np.int8, np.int16, np.int32, np.int64, np.integer,
                 np.uint, np.uint0, np.uint8, np.uint16, np.uint32, np.uint64,
                 np.float16, np.float32, np.float64, np.float128, np.floating)


def map_columntype(x):
    """Map `x` to SeaTable column type."""
    for k, v in COLUMN_TYPES.items():
        if x == v:
            return v
        if x in k:
            return v

    type_str = [f'{" or ".join([str(i) for i in k])} = {str(v).split(".")[-1]}' for k, v in COLUMN_TYPES.items()]
    type_str = "\n".join(type_str)
    raise ValueError(f'Unknown column type "{x}". Try one of the following:\n {type_str}')


def process_records(records, columns=None, row_id_index=True, dtypes=None):
    """Turn records into pandas DataFrame.

    Parameters
    ----------
    records :       list
                    The records (e.g. from an SQL query).
    columns :       list
                    Columns. If provided will filter records to only these
                    columns but also fill missing columns with ``None``.
    row_id_index :  bool
                    Whether to use row IDs as index (if present). If False,
                    will keep as `_row_id` column.
    dtypes :        dict, optional
                    Optional SeaTable data types as strings. If provides, will
                    perform some clean-up operations.

    Returns
    -------
    DataFrame

    """
    assert isinstance(records, list)

    if not isinstance(columns, type(None)):
        # If row ID is requested as index, make sure it's among columns
        if row_id_index and '_id' not in columns and '_id' in records[0]:
            columns = list(columns) + ['_id']
        records = [{c: r.get(c, None) for c in columns} for r in records]

    df = pd.DataFrame.from_records(records)

    if row_id_index and '_id' in df.columns:
        df.set_index('_id', drop=True, inplace=True)
        df.index.name = 'row_id'

    if not isinstance(columns, type(None)):
        df = df[[c for c in columns if c in df.columns]]

    # Try some clean-up operations
    if isinstance(dtypes, dict):
        for c, dt in dtypes.items():
            # Skip non-existing columns
            if c not in df:
                continue

            if dt == 'checkbox':
                df[c] = df[c].astype(bool, copy=False, errors='ignore')
            elif dt == 'number':
                # Manually cleared cells will unfortunately return an empty
                # str ('') as value instead of just no value at all...
                if df[c].dtype == 'O':
                    # Set empty strings to None
                    df.loc[df[c] == '', c] = None
                    # Try to convert to float
                    df[c] = df[c].astype(float, copy=False, errors='ignore')
            elif dt == 'date':
                df[c] = pd.to_datetime(df[c])
            elif dt in ('text', 'long text'):
                # Manually cleared cells will return an empty
                # str ('') as value instead of just no value at all...
                df.loc[df[c] == '', c] = None
                df.loc[df[c] == 'None', c] = None

                # If not any ``None`` in there, convert to proper string column
                if not any(pd.isnull(df[c].values)):
                    df[c] = df[c].values.astype(str)

    return df


def get_auth_token(username, password, server='https://cloud.seatable.io'):
    """Retrieve your auth token.

    Parameters
    ----------
    username :  str
                Login name or email address.
    password :  str
                Your password.
    server :    str
                URL to your seacloud server.

    Returns
    -------
    dict
                Dictionary {'token': 'YOURTOKEN'}

    """
    while server.endswith('/'):
        server = server[:-1]

    url = f'{server}/api2/auth-token/'
    post = {'username': username, 'password': password}

    r = requests.post(url, data=post)

    r.raise_for_status()

    return r.json()


def is_hashable(x):
    """Test if `x` is hashable."""
    try:
        hash(x)
        return True
    except TypeError:
        return False
    except BaseException:
        raise


def is_iterable(x):
    """Return True if `x` is iterable (container)."""
    if isinstance(x, (list, np.ndarray, set, tuple)):
        return True
    return False


def make_iterable(x):
    """Force x into iterable."""
    if not is_iterable(x):
        x = [x]
    return np.asarray(x)


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


def find_base(base=None, required_table=None, auth_token=None, server=None):
    """Find a base matching some parameters.

    Parameters
    ----------
    base :              str | int, optional
                        Name, ID or UUID of the base containing ``table``. If not
                        provided will try to find a base containing ``table``.
    required_table :    str, optional
                        Name of a table that is required to be in the base.
    auth_token :        str, optional
                        Your user's auth token (not the base token). Can either
                        provided explicitly or be set as ``SEATABLE_TOKEN``
                        environment variable.
    server :            str, optional
                        Must be provided explicitly or set as ``SEATABLE_SERVER``
                        environment variable.

    Returns
    -------
    base :              seatable_api.main.SeaTableAPI
    auth_token :        str
    server :            str

    """
    if isinstance(required_table, type(None)) and isinstance(base, type(None)):
        raise ValueError('`base` and `required_table` must not both be `None`')

    if not server:
        server = os.environ.get('SEATABLE_SERVER')

    if not auth_token:
        auth_token = os.environ.get('SEATABLE_TOKEN')

    if not server:
        raise ValueError('Must provide `server` or set `SEATABLE_SERVER` '
                         'environment variable')
    if not auth_token:
        raise ValueError('Must provide either `auth_token` or '
                         'set `SEATABLE_TOKEN` environment variable')

    # Initialize Account
    account = Account(None, None, server)
    account.token = auth_token

    # Now find the base
    workspaces = account.list_workspaces()['workspace_list']

    # Find candidate bases
    cand_bases = []
    for w in workspaces:
        for b in w.get('table_list', []):
            # If base is not None, skip if this base is not a match
            if not isinstance(base, type(None)):
                if not (b.get('id', None) == base or b['name'] == base or b['uuid'] == base):
                    continue
            cand_bases.append([b['workspace_id'], b['name']])
        for b in w.get('shared_table_list', []):
            # If base is not None, skip if this base is not a match
            if not isinstance(base, type(None)):
                if not (b.get('id', None) == base or b['name'] == base or b['uuid'] == base):
                    continue
            cand_bases.append([b['workspace_id'], b['name']])

    # Complain if no matches
    if not len(cand_bases):
        if isinstance(base, type(None)):
            raise ValueError('Did not find a single base.')
        else:
            raise ValueError(f'Did not find a base "{base}"')

    # Filter for table if required
    bases = []
    if not isinstance(required_table, type(None)):
        for bs in cand_bases:
            bb = account.get_base(*bs)
            for t in bb.get_metadata().get('tables', []):
                if t.get('name', None) == required_table or t.get('_id', None) == required_table:
                    bases.append(bs)
                    break
    else:
        bases = cand_bases

    if len(bases) == 0:
        raise ValueError(f'Did not find a matching base')
    elif len(bases) > 1:
        raise ValueError(f'Found multiple matching bases. Please be more specific.')

    # Initialize the base
    base = account.get_base(*bases[0])

    return base, auth_token, server


def write_access(func):
    """Decorator to check for write access to Table."""
    @functools.wraps(func)
    def inner(*args, **kwargs):
        self = args[0]

        if 'Column' in str(type(self)):
            table = self.table
        else:
            table = self

        if table.read_only:
            raise ValueError('Table is read-only to prevent accidental edits. '
                             'Please initialize with `read_only=False` to allow '
                             'writing to it.')
        return func(*args, **kwargs)
    return inner


def suppress_print(func):
    """Suppress any print statements during the function.

    This is unfortunately necessary because seatable_python (in version 2.5.1
    at least) has some bug in how dates are parsed and just prints an error
    for every single row.
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        # Set stdout to nothing
        sys.stdout = open(os.devnull, 'w')
        try:
            res = func(*args, **kwargs)
        except BaseException:
            raise
        finally:
            sys.stdout = sys.__stdout__

        return res
    return inner


def validate_dtype(table, column, values):
    """Assert that datatype of `values` matches that of `column`."""
    dtype = table.dtypes[column]

    if isinstance(values, pd.DatetimeIndex):
        values = values.values

    # Some shortcuts if `values` is an array
    if isinstance(values, np.ndarray):
        # Shortcut for dates
        if dtype == 'date' and values.dtype.type == np.datetime64:
            return
        elif dtype in ('number', ) and values.dtype.type in NUMERIC_TYPES:
            return
        elif dtype in ('text',
                       'long text',
                       'single-select',
                       'multiple-select') and values.dtype.kind == 'U':
            return

    # If list or unknown datattype, bite the bullet and check every value
    for v in values:
        # `None`/`NaN` effectively leads to removal of the value
        if pd.isnull(v):
            continue

        ok = True
        if dtype in ('text', 'long text', 'single-select', 'multiple-select'):
            if not isinstance(v, str):
                ok = False
        elif dtype in ('number', ):
            if not isinstance(v, numbers.Number):
                ok = False
        elif dtype == 'date':
            if not isinstance(v, (numbers.Number, str, dt.datetime, np.datetime64)):
                ok = False

        if not ok:
            raise TypeError(f'Trying to write {type(v)} to "{dtype}" column.')


def validate_table(table):
    """Make sure table can be uploaded.

    In particular, we're making sure that 64bit integers/floats are either
    safely converted to 32bit or, failing that, turned into strings.
    """
    # Avoid unnecessary copies
    is_copy = False
    # For each column check...
    for col in table.columns:
        # ... if it's 64 bit integers/floats
        if table.dtypes[col] == (np.int64, np.float64):
            # Make copy if not already happened
            if not is_copy:
                table = table.copy()
                is_copy = False
            # If too large/small for 32 bits
            if table[col].max() > 2_147_483_647 or table[col].min() < -2_147_483_648:
                table[col] = table[col].astype(str)
            else:
                table[col] = table[col].astype(np.int32)

        if table.dtypes[col] in (float, np.float32, np.float64):
            if any(np.isinf(table[col].values)):
                raise ValueError(f'Column "{col}" contains non-finite values.')

    return table


def validate_values(values, col=None):
    """Similar to validate_table but for a 1d array."""
    # Note that any list of numeric, non-decimal values will automatically
    # get the np.int64 datatype -> hence we won't warn if we safely downcast

    if isinstance(values, pd.DatetimeIndex):
        values = values.values
    else:
        values = np.asarray(values)

    if values.dtype == np.int64:
        # If too large/small for 32 bits
        if values.max() > 2_147_483_647 or values.min() < -2_147_483_648:
            values = values.astype(float)
        else:
            values = values.astype(np.int32)
    elif values.dtype in (float, np.float32, np.float64):
        if any(np.isinf(values)):
            raise ValueError('At least some values are non-finite.')

    # Dates must be given as strings "YEAR-MONTH-DAY HOUR:MINUTE:SECONDS"
    if col.dtype == 'date':
        # If object type make sure each value is converted correctly
        if values.dtype == 'O':
            # Do not use np.isnan here!
            not_nan = values != None

            for i, v in enumerate(values):
                # Skip empty rows
                if v == None or v == np.nan:
                    continue

                if isinstance(v, str):
                    # For strings we won't do any extra checking
                    continue
                elif isinstance(v, dt.date):
                    values[i] = dt.date.strftime(v, '%Y-%m-%d %H:%M:%S')
                elif isinstance(v, np.datetime64):
                    values[i] = np.datetime_as_string(v, unit='m').replace('T', ' ')
                else:
                    raise TypeError('Dates must be given as string(s) (e.g. '
                                    '"2021-10-01 10:10"), or as datetime or '
                                    f'numpy.datetime64 objects. Got "{v}" '
                                    f'({type(v)}).')
        elif values.dtype.type in (np.datetime64, pd.Timestamp):
            # We need strings formatted like this "2012-10-01 14:01"
            # Note that we also need to cater for empty columns
            is_date = ~np.isnan(values)
            dates = values[is_date]
            dates_str = np.datetime_as_string(dates, unit='m')
            dates_str = [d.replace('T', ' ') for d in dates_str]

            # Rewrite values
            values = np.zeros(len(values), dtype='O')
            values[is_date] = dates_str
            values[~is_date] = None
        # If non-string (i.e. numeric) array
        elif values.dtype.kind != 'U':
            raise TypeError('Dates must be given as string(s) (e.g. '
                            '"2021-10-01 10:10"), or as datetime or '
                            f'numpy.datetime64 objects. Got {values.dtype}.')
    elif col.dtype in ('single-select', 'multiple-select'):
        options = [o['name'] for o in col.meta['data']['options']]
        not_options = ~np.isin(values, options)
        if any(not_options):
            miss = np.unique(values[not_options]).astype(str).tolist()
            logger.warning('Some of the values to write are not currently '
                           f'among options for column "{col.name}" ({col.dtype}):'
                           f' {", ".join(miss)}.\nThese will be added '
                           'automatically but you will need to refresh the '
                           'website for them to show up.')

    return values.tolist()


def make_records(table):
    """Create clean records from given table.

    In brief:
      - create records
      - drop any `null`, `None` or `NaNs`

    Parameters
    ----------
    table :     pd.DataFrame

    Returns
    -------
    records :   list of dicts

    """
    records = table.to_dict(orient='records')

    records = [{k: v for k, v in r.items() if not pd.isnull(v)} for r in records]

    return records


def flatten(x):
    """Flatten list of lists."""
    if not isinstance(x, list):
        return [x]
    l = []
    for v in x:
        l += flatten(v)
    return l
