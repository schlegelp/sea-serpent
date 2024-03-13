import logging
import copy
import json
import jwt
import numbers
import requests
import time
import warnings

import datetime as dt
import numpy as np
import pandas as pd

from functools import partial
#from seatable_api.main import SeaTableAPI
from seatable_api.constants import ColumnTypes
from tqdm.auto import trange, tqdm

from .utils import (process_records, make_records,
                    is_iterable, make_iterable, is_hashable,
                    map_columntype, find_base, write_access,
                    validate_dtype, validate_comparison, validate_table,
                    validate_values, flatten, check_token, dict_replace,
                    is_equal_array, is_array_like)
from .patch import SeaTableAPI, Account

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(levelname)-5s : %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)


NoneType = type(None)


class Table:
    """A remote data table.

    Parameters
    ----------
    table :         str | int
                    Name or index of the table. If index, you must provide a
                    base.
    base :          str | int | SeaTableAPI, optional
                    Name, ID or UUID of the base containing ``table``. Providing
                    the base can greatly speed up initialization. If not
                    provided will try to find a base containing ``table``.
    auth_token :    str, optional
                    Your user's auth token (not the base token). Can either
                    provided explicitly or be set as ``SEATABLE_TOKEN``
                    environment variable.
    server :        str, optional
                    Must be provided explicitly or set as ``SEATABLE_SERVER``
                    environment variable.
    read_only :     bool
                    Whether to allow writes to table. The main purpose of this
                    is to avoid accidental edits when all you want is reading
                    data.
    max_operations : int
                    How many operations (i.e. writes) we can do at a time. The
                    limit is set by the server, I think.
    sanitize :      bool
                    Whether to sanitize downloaded data. Specifically this will:
                      - turn empty strings (``''``) into ``None``
                      - turn date columns into datetime objects
                      - turn checked columns into boolean arrays
    progress :      bool
                    Whether to show progress bars.

    """

    def __init__(self, table, base=None, auth_token=None, server=None,
                 read_only=True, max_operations=1000, sanitize=True,
                 progress=True):
        # If the table is given as index (i.e. first table in base X), `base`
        # must not be `None`
        if isinstance(table, int) and isinstance(base, type(None)):
            raise ValueError('Must provide a `base` when giving `table` index '
                             'instead of name.')

        # Find base and table
        if not isinstance(base, SeaTableAPI):
            (self.workspace_id,
             self.base_name,
             self.auth_token,
             self.server) = find_base(base=base,
                                      server=server,
                                      auth_token=auth_token,
                                      required_table=table if not isinstance(table, int) else None)
            # This sets/refreshes self.base
            self.auth()
        else:
            self.base = base
            self.server = base.server_url
            self.base_name = base.dtable_name
            self.workspace_id = base.workspace_id
            self.auth_token = None

        if isinstance(table, int):
            tables = self.base.get_metadata().get('tables', [])
            if not len(tables):
                raise ValueError('Base does not contain any tables.')
            table = tables[0]['name']

        # Pull meta data for base and table
        self._stale = True
        self._meta = None
        self.name = table
        self.fetch_meta()

        if self.name == self.id:
            self.name = self.meta['name']

        self.loc = LocIndexer(self)
        # Haven't decided if the below would actually be useful
        #self.iloc = iLocIndexer(self)

        # Whether to show progress bars
        self.progress = progress

        # Whether table is read-only (default)
        self.read_only = read_only

        # Whether to clean-up downloaded data
        self.sanitize = sanitize

        # Maximum number of operations (e.g. edits) per batch
        self.max_operations = max_operations

        # Some parameters for bundling updates
        self._hold = False
        self._queue = []

    def __array__(self, dtype=None):
         return np.array(self.values, dtype=dtype)

    def __dir__(self):
        """Custom __dir__ to make columns searchable."""
        return list(set(super().__dir__() + list(self.columns)))

    def __len__(self):
        """Length of table."""
        return self.shape[0]

    def __getattr__(self, name):
        if name not in self.columns and name != '_id':
            # Update meta data
            _ = self.fetch_meta()

            # If name still not in columns
            if name not in self.columns:
                raise AttributeError(f'Table has no "{name}" column')
        return Column(name=name, table=self)

    def __getitem__(self, key):
        if isinstance(key, Filter):
            return self.loc[key]

        # If single string assume this is a column and return the promise
        if is_hashable(key):
            if key not in self.columns and key not in ('_id', ):
                self.fetch_meta()
                if key not in self.columns:
                    raise AttributeError(f'Table has no "{key}" column')
            return Column(name=key, table=self)

        if isinstance(key, slice):
            columns = self.columns[key]
        elif is_iterable(key):
            self._check_columns(key)
            columns = key
        else:
            raise KeyError(key)

        query = create_query(self, columns=columns, where=None, limit=None)
        records = self.query(query, no_limit=True)
        return process_records(records, columns=columns,
                               dtypes=self.dtypes.to_dict() if self.sanitize else None)

    @write_access
    @check_token
    def __setitem__(self, key, values):
        if not is_hashable(key) or isinstance(key, tuple):
            raise KeyError('Key must be hashable (i.e. a single column). Use '
                           '.loc indexer to set values for a specific slice.')

        # Update meta data for self
        _ = self.fetch_meta()

        if key not in self.columns:
            raise KeyError('Column must exists, use `add_column()` method to '
                           f'create "{key}" before setting its values')

        if isinstance(values, (pd.Series, Column)):
            values = values.values

        if not is_iterable(values):
            values = [values] * len(self)
        elif len(values) != len(self):
            raise ValueError(f'Length of values ({len(values)}) does not '
                             f'match length of keys ({len(self)})')

        # Validate datatype
        validate_dtype(self, key, values)

        # This checks for potential int64 -> int32 issues
        values = validate_values(values, col=self[key])

        # Fetch the IDs
        row_ids = self.query('SELECT _id', no_limit=True)

        records = [{'row_id': r['_id'],
                    'row': {key: v if not pd.isnull(v) else None}} for r, v in zip(row_ids, values)]

        if not self._hold:
            r = batch_upload(partial(self.base.batch_update_rows, self.name),
                             records, batch_size=self.max_operations,
                             progress=self.progress)

            if 'success' in r:
                logger.info('Write successful!')
        else:
            self._queue += records

    def __repr__(self):
        shape = self.shape
        return f'SeaTable <"{self.name}" ({self.base.dtable_name}), {shape[0]} rows, {shape[1]} columns>'

    @property
    def collaborators(self):
        """List of all user with access to this table (cached)."""
        if isinstance(getattr(self, '_collaborators', None), type(None)):
            url = (f"{self.server}/dtable-server/api/v1/dtables/"
                   f"{self.base.dtable_uuid}/related-users/")
            r = requests.get(url, headers=self.base.headers)
            r.raise_for_status()

            self._collaborators = pd.DataFrame(r.json()['user_list'])
            col = self._collaborators.pop("name")  # bring name column to front
            self._collaborators.insert(0, col.name, col)

        return self._collaborators

    @property
    def meta(self):
        """Meta data for this table."""
        if not getattr(self, '_meta', None) or getattr(self, '_stale', True):
            self.fetch_meta()
            self._stale = False
        return self._meta

    @meta.setter
    def meta(self, value):
        if not isinstance(value, dict):
            raise TypeError(f'`meta` must be dict, got {type(value)}')
        self._meta = value

    @property
    def columns(self):
        """Table columns."""
        return np.array([c['name'] for c in self.meta['columns']])

    @property
    def row_ids(self):
        """Row IDs."""
        return self['_id'].values.astype(str)

    @property
    def dtypes(self):
        """Column data types."""
        return pd.Series([c['type'] for c in self.meta['columns']],
                         index=self.columns)

    @property
    def id(self):
        """ID of the table."""
        return self.meta['_id']

    @property
    def server_info(self):
        """Server info."""
        r = requests.get(f'{self.server}server-info')
        r.raise_for_status()
        return r.json()

    @property
    def shape(self):
        """Shape of table."""
        n_rows = self.query('SELECT COUNT(*)',
                            no_limit=False)[0].get('COUNT(*)', 'NA')
        return (n_rows, len(self.columns))

    @property
    def values(self):
        """Values."""
        return self.to_frame().values

    @property
    def views(self):
        """Available views for this table."""
        return [v['name'] for v in self.meta['views']]

    @classmethod
    def from_frame(cls, df, table_name, base, id_col=0, auth_token=None, server=None):
        """Create a new table from dataframe.

        Parameters
        ----------
        df :            pandas.DataFrame | seaserpent.Table
                        Table to export to SeaTable. For pandas.DataFramess the
                        data types are inferred as follows:
                          - object or string -> text
                          - int, float -> number
                          - bool -> check box
                          - categorical -> single-select
                          - lists (object) -> multiple-select
        table_name :    str
                        Name of the new table.
        base :          str | int
                        Name or ID of base.
        id_col :        str | int
                        Name or index of the ID column to use. Ignored if input
                        is `seaserpent.Table`

        Returns
        -------
        Table
                        Will be initialized with `read_only=False`.

        """
        if isinstance(df, Table):
            return cls._from_ss_table(df,
                                      table_name=table_name,
                                      base=base,
                                      auth_token=auth_token,
                                      server=server)

        # Some sanity checks
        if len(df.columns) < len(np.unique(df.columns)):
            raise ValueError('Table must not contain duplicate column names')

        if isinstance(id_col, int):
            id_col = df.columns[id_col]

        if id_col not in df.columns:
            raise ValueError(f'ID column "{id_col}" not among columns.')

        # Validate table
        df = validate_table(df)

        if df[id_col].dtype in (object, 'string[python]'):
            id_col_dtype = 'text'
        else:
            id_col_dtype = map_columntype(df[id_col].dtype.kind).name.lower()
        columns = [{'column_name': id_col,
                    'column_type': id_col_dtype}]

        table = cls.new(table_name=table_name, base=base, columns=columns,
                        auth_token=auth_token, server=server)
        logger.info('New table created.')

        # Create the columns (infer data types)
        for c in df.columns:
            # Skip ID column
            if c == id_col:
                continue

            col = df[c]

            if col.dtype in (object, 'string[python]'):
                # If all non-null values are lists
                if all([isinstance(v, list) or pd.isnull(v) for v in col.values]):
                    dtype = 'multiple_select'
                else:
                    dtype = str
            elif isinstance(col.dtype, pd.CategoricalDtype):
                dtype = 'single_select'
            else:
                dtype = col.dtype.kind

            if dtype == 'single_select':
                options = [{
                            'name': str(o),
                            'color': '#aaa',
                            'textColor': '#000000'
                            } for o in col.dtype.categories]
            elif dtype == 'multiple_select':
                vals = []
                for v in col.values:
                    if isinstance(v, list):
                        vals += v
                options = [{
                            'name': str(o),
                            'color': '#aaa',
                            'textColor': '#000000'
                            } for o in list(set(vals))]
            else:
                options = None

            table.add_column(col_name=c, col_type=dtype, col_options=options)

        logger.info('New columns added.')

        # Add the actual data
        table.append(df)
        logger.info('Data uploaded.')

        return table

    @classmethod
    def _from_ss_table(cls, df, table_name, base, auth_token=None, server=None):
        """Create Table from other Table.

        Entry point for this is usually `Table.from_table`.

        """
        assert isinstance(df, Table)

        if 'link' in df.dtypes.values:
            logger.warning('Table contains `link` columns which will not be'
                           'copied.')

        columns = []
        for c in df.meta['columns']:
            # Skip link columns
            if c['type'] == 'link':
                continue

            columns.append({'column_name': c['name'],
                            'column_type': c['type']})
            if 'data' in c:
                columns[-1]['column_data'] = c['data']

        table = cls.new(table_name=table_name, base=base,
                        columns=columns,
                        auth_token=auth_token, server=server)
        logger.info('New table created.')

        # Resize columns
        for col in df.meta['columns']:
            table[col['name']].resize(col['width'])

        # Add the actual data
        table.append(df.to_frame())
        logger.info('Data uploaded.')

        # Add views
        if df.meta.get('views', None):
            # Map between old and new col keys in the views
            views = copy.deepcopy(df.meta['views'])
            old2new = dict(zip([c['key'] for c in df.meta['columns']],
                               [c['key'] for c in table.meta['columns']]))
            for view in views:
                dict_replace(view, 'column_key', old2new)
                view['hidden_columns'] = [old2new[c] for c in view['hidden_columns']]

            # Now add & modify the views
            existing_views = [v['name'] for v in table.meta['views']]
            for view in views:
                # Generate view if it doesn't exist yet
                if view['name'] not in existing_views:
                    url = (f"{table.server}/dtable-server/api/v1/dtables/"
                           f"{table.base.dtable_uuid}/views/"
                           f"?table_name={table_name}")
                    r = requests.post(url,
                                      headers=table.base.headers,
                                      json={'name': view['name']})
                    r.raise_for_status()

                    # Data contains the ID of the view
                    _ = r.json()

                # Now actually set the view
                url = (f"{table.server}/dtable-server/api/v1/dtables/"
                       f"{table.base.dtable_uuid}/views/{view['name']}"
                       f"?table_name={table_name}")

                # These are the parameters we can send
                # Note: colorbys don't seem to work (= ignored by API endpoint)
                params = ('is_locked', 'filters', 'filter_conjunction',
                          'hidden_columns', 'sorts', 'groupbys', 'colorbys')
                data = {p: view[p] for p in params if p in view}

                r = requests.put(url,
                                 headers=table.base.headers,
                                 json=data)
                r.raise_for_status()

                data = r.json()

            logger.info('Views added.')

        return table

    @classmethod
    def new(cls, table_name, base, columns=None, auth_token=None, server=None):
        """Create a new table.

        Parameters
        ----------
        table_name :    str
                        Name of the new table.
        base :          str | int
                        Name or ID of base.
        columns :       list, optional
                        If provided, must be a list of dicts:

                        [{'column_name': 'First column',
                          'column_type': 'number',
                          'column_data': None}]

                        If not provided, the table will be initialized with a
                        single "Name" column.

        Returns
        -------
        Table
                        Will be initialized with `read_only=False`.

        """
        # Find the base
        (workspace_id,
         base_name,
         auth_token,
         server) = find_base(base=base,
                             server=server,
                             auth_token=auth_token)
        account = Account(None, None, server)
        account.token = auth_token
        # Initialize the base
        base = account.get_base(workspace_id, base_name)

        existing_tables = base.get_metadata()['tables']
        existing_names = [t['name'] for t in existing_tables]

        if table_name in existing_names:
            raise ValueError(f'Base already contains a table named "{table_name}"')

        base.add_table(table_name, lang='en', columns=columns)

        return cls(table=table_name, base=base, read_only=False,
                   auth_token=auth_token, server=server)

    def _check_columns(self, columns):
        """Check if `columns` exist."""
        if not is_iterable(columns):
            columns = [columns]
        columns = np.asarray(columns)
        miss = columns[~np.isin(columns, self.columns) & (columns != '_id')]
        if any(miss):
            raise KeyError(f'"{miss}" not among columns')

    def _col_ids_to_names(self, ids):
        """Map column IDs to names. Returns `None` if id not found."""
        id_map = {c['key']: c['name'] for c in self.meta['columns']}

        if not is_iterable(ids):
            return id_map.get(ids)

        return [id_map.get(i, None) for i in ids]

    def _token_time_left(self):
        """Time left before token expires [s]."""
        decoded = jwt.decode(self.base.jwt_token, algorithms=['HS256'],
                             options={"verify_signature": False})

        return int(decoded['exp']) - int(time.time())

    def auth(self):
        """Authenticate."""
        account = Account(None, None, self.server)
        account.token = self.auth_token

        # Initialize the base
        self.base = account.get_base(self.workspace_id, self.base_name)

    @write_access
    @check_token
    def add_column(self, col_name, col_type, col_data=None, col_options=None):
        """Add new column to table.

        Parameters
        ----------
        col_name :  str
                    Name of the new column.
        col_type :  str | type
                    The type of the new column:
                      - a column type, e.g. `seatable_api.constants.ColumnTypes.NUMBER`
                      - 'number' for "number"
                      - `bool` for "checkbox"
                      - `str` for "text"
                      - "long_text" for "longtext"
                      - "link" for "link"
        col_data :  dict, optional
                    Config info of column. Required for link-type columns,
                    optional for other type columns.
        col_options : records, optional
                    Column options for single and multiple select columns.
                    Format must be something like:

                     [{"name": "ddd", "color": "#aaa", "textColor": "#000000"}]

        """
        # Make sure meta data is up-to-date
        self.fetch_meta()

        if col_name in self.columns:
            raise ValueError(f'Column "{col_name}" already exists.')

        col_type = map_columntype(col_type)

        resp = self.base.insert_column(table_name=self.name,
                                       column_name=col_name,
                                       column_type=col_type,
                                       column_data=col_data)

        if col_options and col_type in (ColumnTypes.SINGLE_SELECT,
                                        ColumnTypes.MULTIPLE_SELECT):
            self.base.add_column_options(self.name, col_name, col_options)

        # Make sure meta is updated before next use
        self._stale = True

        if not resp.get('name'):
            raise ValueError(f'Error writing to table: {resp}')
        logger.info(f'Column "{col_name}" ({col_type}) added.')

    @write_access
    @check_token
    def add_linked_column(self, col_name, link_col, link_on, formula='lookup'):
        """Add linked column to table.

        Parameters
        ----------
        col_name :  str
                    Name of the new column.
        link_col :  str
                    Column containing the links to other table.
        link_on :   str
                    Column in other table to link on.
        formula :   str
                    Formula to use for pulling data from other table:
                      - "lookup" returns values of the linked record(s)
                      - "count_links" counts number of linked records
                      - "rollup-average": average across linked records
                      - "rollup-sum": sum across linked records
                      - "rollup-concatenate": concatenate across linked records
                      - "findmax": maximum across linked records
                      - "findmin": minimum across linked records

        See Also
        --------
        Table.link
                    For creating links between two tables.

        """
        ALLOWED_FORMULAS = ('lookup', 'count_links', 'rollup-avg', 'findmax',
                            'findmin', 'rollup-sum', 'rollup-conc')
        if formula not in ALLOWED_FORMULAS:
            raise ValueError(f'Unrecognized formula "{formula}"')

        # Make sure meta data is up-to-date
        self.fetch_meta()

        if col_name in self.columns:
            raise ValueError(f'Column "{col_name}" already exists.')

        if link_col not in self.columns:
            raise ValueError(f'Link column "{link_col}" does not exist.')
        elif self[link_col].dtype != 'link':
            raise TypeError(f'Link column must be type "link", not "{self[link_col].dtype}"')

        # Prepare column data
        col_data = {}
        col_data['formula'] = formula.split('-')[0]
        col_data['link_column'] = link_col
        col_data['level1_linked_column'] = link_on
        if formula.startswith('rollup'):
            col_data['summary_method'] = formula.split('-')[1]

        _ = self.base.insert_column(table_name=self.name,
                                    column_name=col_name,
                                    column_type=ColumnTypes.LINK_FORMULA,
                                    column_data=col_data)

        # Make sure meta is updated before next use
        self._stale = True

        logger.info(f'Linked column "{col_name}" added.')

    @write_access
    @check_token
    def append(self, other):
        """Append rows of `other` to the end of this table.

        Columns in `other` that are not also in the table are ignored.

        Parameters
        ----------
        other :     pandas.DataFrame

        """
        if isinstance(other, Table):
            other = other.to_frame()

        if not isinstance(other, pd.DataFrame):
            raise TypeError(f'`other` must be DataFrame, got "{type(other)}"')

        other = other[other.columns[np.isin(other.columns, self.columns)]].copy()

        if not other.shape[1]:
            raise ValueError('None of the columns in `other` are in table')

        for col in other.columns:
            # Validate datatype
            validate_dtype(self, col, other[col].values)

            # This checks for potential int64 -> int32 issues
            other[col] = validate_values(other[col].values,
                                         col=self[col])

        records = make_records(other)

        r = batch_upload(partial(self.base.batch_append_rows, self.name),
                         records, desc='Appending',
                         batch_size=self.max_operations,
                         progress=self.progress)

        if 'success' in r:
            logger.info('Rows successfully added!')

    @write_access
    @check_token
    def delete_rows(self, rows, skip_confirmation=False):
        """Delete given rows.

        Parameters
        ----------
        rows :      int | str | iterable | Filter
                    Can be either:
                     - single integer or list thereof is intepreted as indices
                       IMPORTANT: this expects Python indices, i.e. the
                       first row has index index 0 (not 1)
                     - single str or list thereof is intepreted as row ID(s)
                     - an array of booleans is interpreted as mask and rows where
                       the value is True will be deleted
                     - a Filter query
        skip_confirmation : bool
                    If True, will skip confirmation.

        """
        if isinstance(rows, (int, str)):
            rows = [rows]

        if isinstance(rows, Filter):
            rows = self.loc[rows, '_id'].values.astype(str)

        # Pandas Boolean arrays are a specia datatype which is annoying
        # because they get converted to `object` by np.asarray by default
        if isinstance(rows, pd.core.arrays.boolean.BooleanArray):
            rows = np.asarray(rows, dtype=bool)
        else:
            rows = np.asarray(rows)

        if rows.dtype.kind == 'U':
            miss = ~np.isin(rows, self.row_ids)
            if any(miss):
                raise ValueError('Some of the provided row IDs do not appear '
                                 f'to exist: {rows[miss]}')
            row_ids = rows
        elif rows.dtype in (np.int64, np.int32):
            row_ids = self.row_ids[rows]
        elif rows.dtype == bool:
            if len(rows) != len(self):
                raise ValueError(f'Length of boolean array ({len(rows)}) does not '
                                 f'match that of table ({len(self)})')
            row_ids = self.row_ids[rows]
        else:
            raise TypeError('Unable to determine which rows to delete from data '
                            f'of type "{rows.dtype}"')

        if not skip_confirmation:
            if input(f'Delete {len(row_ids)} rows in table "{self.name}" '
                     f'in base "{self.base.dtable_name}"? [y/n]').lower() != 'y':
                return

        r = batch_upload(partial(self.base.batch_delete_rows, self.name),
                         row_ids.tolist(),
                         desc='Deleting',
                         batch_param='row_ids',
                         batch_size=self.max_operations,
                         progress=self.progress)

        if 'success' in r:
            logger.info(f'Successfully deleted {len(row_ids)} rows!')

    @write_access
    @check_token
    def delete(self, skip_confirmation=False):
        """Danger! Delete this table.

        Parameters
        ----------
        skip_confirmation : bool
                            If True, will skip confirmation.

        """
        if not skip_confirmation:
            if input(f'Delete table "{self.name}" '
                     f'in "{self.base.dtable_name}"? [y/n]').lower() != 'y':
                return

        url = self.base._table_server_url()

        json_data = {
                    'table_name': self.name
                }
        r = requests.delete(url,
                            json=json_data,
                            headers=self.base.headers,
                            timeout=self.base.timeout)
        r.raise_for_status()

        if 'success' in r.content.decode():
            logger.info('Table successfully deleted!')
        else:
            logger.warning(f'Something went wrong: {r.content.decode()}')

    def time_machine(self, date, columns=None):
        """Recreate version of table at a given point in time.

        Important: this is work in progress and does currently not revert
        deleted or added columns/rows!

        Parameters
        ----------
        date :          dt.date | dt.datetime
                        Time to go back to.

        Returns
        -------
        pandas.DataFrame

        """
        # Convert date to datetime
        if isinstance(date, dt.date):
            date = dt.datetime(date.year, date.month, date.day)

        if date >= dt.datetime.now():
            raise ValueError('Time travel only works backwards, not into the '
                             'future!')

        logs = self.fetch_logs(max_time=date, unpack=True)

        if not columns:
            columns = self.columns

        if '_id' not in columns:
            columns = np.append(columns, '_id')

        # Grab the table
        table = self[columns]

        # Only keep the oldest log entry for each row/col
        logs = logs.drop_duplicates(['row_id', 'column'], keep='last')

        # Drop
        logs = logs[logs.row_id.isin(table.index.values) & logs.column.isin(table.columns)]

        for i, row in logs.iterrows():
            if isinstance(row.old_value, dict):
                row.old_value = row.old_value['text']

            table.loc[row.row_id, row.column] = row.old_value

        return table

    @check_token
    def fetch_logs(self, max_entries=25, max_time=None, unpack=True, progress=True):
        """Fetch activity logs for this table.

        Parameters
        ----------
        max_entries :   int
                        Maximum number of logs to return. Set to ``None`` to
                        fetch all logs (may take a while). Ignored if
                        ``max_time`` is given.
        max_time :      dt.date | dt.datetime
                        Max time to go back to. If ``max_time`` is given,
                        ``max_entries`` is ignored!
        unpack :        bool
                        If False, each row represents an operation on multiple
                        rows/values. If True, each row will represent an edit
                        to a single value. This will drop anything that isn't
                        a "modify_rows" operation!

        Returns
        -------
        pandas.DataFrame

        """
        # Convert date to datetime
        if isinstance(max_time, dt.date):
            max_time = dt.datetime(max_time.year, max_time.month, max_time.day)

        # Convert datetime to timestamp
        if isinstance(max_time, dt.datetime):
            total = (dt.datetime.now() - max_time).days
            max_time = int(dt.datetime.timestamp(max_time) * 1e3)
            now = int(dt.datetime.timestamp(dt.datetime.now()) * 1e3)
            max_entries = None
        elif max_entries:
            total = max_entries
        else:
            total = None

        entries = 0
        page = 1
        logs = []

        with tqdm(desc='Fetching logs',
                  total=total,
                  leave=False,
                  disable=not progress) as pbar:
            while True:
                url = (f"{self.server}/dtable-server/api/v1/dtables/"
                       f"{self.base.dtable_uuid}/operations/"
                       f"?page={page}&per_page=25")  # per_page seems to be hard-coded
                r = requests.get(url, headers=self.base.headers)
                r.raise_for_status()

                data = r.json()['operations']

                # Stop if no more pages
                if not data:
                    break

                # Create DataFrame
                logs.append(pd.DataFrame.from_records(data))

                # Parse op dictionary
                logs[-1]['operation'] = logs[-1].operation.map(json.loads)

                # Drop irrelevant entries
                table = logs[-1].operation.map(lambda x: x.get('table_id', None))

                # Get the last timestamp before we drop rows 
                last_op = logs[-1].op_time.values[-1]

                # Drop rows for other tables
                logs[-1] = logs[-1][table == self.id]

                entries += logs[-1].shape[0]

                if max_entries:
                    pbar.update(logs[-1].shape[0])
                    if entries >= max_entries:
                        break

                if max_time:
                    # Get the days we went back
                    days = (now - last_op) / 1e3 / 86_400
                    diff = int(days - pbar.n)
                    if diff:
                        pbar.update(diff)
                    if last_op <= max_time:
                        break

                page += 1

        # Combine
        logs = pd.concat([ta for ta in logs if not ta.empty], axis=0)

        if max_time:
            logs = logs[logs.op_time >= max_time]
        elif max_entries:
            logs = logs.iloc[:max_entries].copy()

        # Some clean-up:
        # Extract/parse relevant values
        logs['op_time'] = (logs.op_time / 1e3).map(dt.datetime.fromtimestamp)
        logs['op_type'] = logs.operation.map(lambda x: x['op_type']).astype('category')

        users = self.collaborators.set_index('email').name.to_dict()
        logs.insert(0, 'user', logs.author.map(users))
        logs['user'] = logs['user'].astype('category')
        logs.drop('author', inplace=True, axis=1)

        logs['rows_modified'] = logs.operation.map(lambda x: len(x.get('row_ids', [])))
        logs.loc[logs.op_type == 'modify_row', 'rows_modified'] = 1

        col = logs.pop('operation')
        logs['details'] = col

        def clean_details(x):
            """Run some clean up on the details column."""
            # Pop some values we don't need (anymore)
            _ = x.pop('table_id', '')
            _ = x.pop('op_type', '')

            return x

        logs['details'] = logs.details.map(clean_details)

        logs = logs.reset_index(drop=True)

        if unpack:
            unpacked = []
            for row in logs[logs.op_type.isin(['modify_rows', 'modify_row'])].itertuples():
                user = row.user
                app = row.app
                op_time = row.op_time
                op_id = row.op_id
                op_type = row.op_type
                details = row.details

                # If single row turn it into a fake multi row edit
                if row.op_type == 'modify_row':
                    row_id = details['row_id']
                    details['row_ids'] = [row_id]
                    details['updated'] = {row_id: details['updated']}
                    details['old_rows'] = {row_id: details['old_row']}

                for id in details['row_ids']:
                    for col, new_value in details['updated'][id].items():
                        # Skip internal columns like '_last_modifier'
                        if col.startswith('_'):
                            continue
                        old_value = details['old_rows'][id].get(col, None)
                        unpacked.append([user, app, op_time, op_id,
                                         id, col, old_value, new_value])

            logs = pd.DataFrame(unpacked,
                                columns=['user', 'app', 'op_time', 'op_id',
                                         'row_id', 'column',
                                         'old_value', 'new_value'])

            logs['column'] = self._col_ids_to_names(logs.column.values)
            for c in ['user', 'app', 'column']:
                logs[c] = logs[c].astype('category')

        return logs

    @check_token
    def fetch_row_logs(self, ids, max_entries=25, max_time=None, unpack=True, progress=True):
        """Fetch activity logs for given row(s).

        Parameters
        ----------
        ids :           str | iterable | pandas.DataFrame
                        Row IDs to fetch logs for. Can be:
                          - a single row ID (e.g. "M9MzMmdRQdC5X-aP5qRSxg")
                          - list of the above
                          - a DataFrame where the index contains row IDs
        max_entries :   int
                        Maximum number of logs to return. Set to ``None`` to
                        fetch all logs (may take a while). Ignored if
                        ``max_time`` is given.
        max_time :      dt.date | dt.datetime
                        Max time to go back to. If ``max_time`` is given,
                        ``max_entries`` is ignored!
        unpack :        bool
                        If False, each row represents an operation on multiple
                        rows/values. If True, each row will represent an edit
                        to a single value. This will drop anything that isn't
                        a "modify_rows" operation!

        Returns
        -------
        pandas.DataFrame

        """
        # Convert date to datetime
        if isinstance(max_time, dt.date):
            max_time = dt.datetime(max_time.year, max_time.month, max_time.day)

        # Convert datetime to timestamp
        if isinstance(max_time, dt.datetime):
            total = (dt.datetime.now() - max_time).days
            max_time = int(dt.datetime.timestamp(max_time) * 1e3)
            now = int(dt.datetime.timestamp(dt.datetime.now()) * 1e3)
            max_entries = None
        elif max_entries:
            total = max_entries
        else:
            total = None

        if isinstance(ids, str):
            ids = [ids]
        elif isinstance(ids, pd.DataFrame):
            ids = ids.index.values

        logs = []
        for i in tqdm(ids, leave=False, disable=not progress, desc='Rows'):
            with tqdm(desc='Fetching logs',
                      total=total,
                      leave=False,
                      disable=not progress) as pbar:
                entries = 0
                page = 1
                while True:
                    url = (f"{self.server}/dtable-server/api/v1/dtables/"
                           f"{self.base.dtable_uuid}/activities/"
                           f"?row_id={i}&page={page}&per_page=25")  # per_page seems to be hard-coded
                    r = requests.get(url, headers=self.base.headers)
                    r.raise_for_status()

                    data = r.json()['activities']

                    # Stop if no more pages
                    if not data:
                        break

                    # Create DataFrame
                    logs.append(pd.DataFrame.from_records(data))

                    # Parse op dictionary
                    logs[-1]['activities'] = logs[-1].operation.map(json.loads)

                    # Drop irrelevant entries
                    table = logs[-1].operation.map(lambda x: x.get('table_id', None))
                    logs[-1] = logs[-1][table == self.id]

                    entries += logs[-1].shape[0]

                    if max_entries:
                        pbar.update(logs[-1].shape[0])
                        if entries >= max_entries:
                            break

                    if max_time:
                        # Get the days we went back
                        days = (now - logs[-1].op_time.values[-1]) / 1e3 / 86_400
                        diff = int(days - pbar.n)
                        if diff:
                            pbar.update(diff)
                        if logs[-1].op_time.values[-1] <= max_time:
                            break

                    page += 1

        return logs

        # Combine
        logs = pd.concat(logs, axis=0)

        if max_time:
            logs = logs[logs.op_time >= max_time]
        elif max_entries:
            logs = logs.iloc[:max_entries].copy()

        # Some clean-up:
        # Extract/parse relevant values
        logs['op_time'] = (logs.op_time / 1e3).map(dt.datetime.fromtimestamp)
        logs['op_type'] = logs.operation.map(lambda x: x['op_type']).astype('category')

        users = self.collaborators.set_index('email').name.to_dict()
        logs.insert(0, 'user', logs.author.map(users))
        logs['user'] = logs['user'].astype('category')
        logs.drop('author', inplace=True, axis=1)

        logs['rows_modified'] = logs.operation.map(lambda x: len(x.get('row_ids', [])))
        logs.loc[logs.op_type == 'modify_row', 'rows_modified'] = 1

        col = logs.pop('operation')
        logs['details'] = col

        def clean_details(x):
            """Run some clean up on the details column."""
            # Pop some values we don't need (anymore)
            _ = x.pop('table_id', '')
            _ = x.pop('op_type', '')

            return x

        logs['details'] = logs.details.map(clean_details)

        logs = logs.reset_index(drop=True)

        if unpack:
            unpacked = []
            for row in logs[logs.op_type.isin(['modify_rows', 'modify_row'])].itertuples():
                user = row.user
                app = row.app
                op_time = row.op_time
                op_id = row.op_id
                details = row.details

                # If single row turn it into a fake multi row edit
                if row.op_type == 'modify_row':
                    row_id = details['row_id']
                    details['row_ids'] = [row_id]
                    details['updated'] = {row_id: details['updated']}
                    details['old_rows'] = {row_id: details['old_row']}

                for id in details['row_ids']:
                    for col, new_value in details['updated'][id].items():
                        # Skip internal columns like '_last_modifier'
                        if col.startswith('_'):
                            continue
                        old_value = details['old_rows'][id].get(col, None)
                        unpacked.append([user, app, op_time, op_id,
                                         id, col, old_value, new_value])

            logs = pd.DataFrame(unpacked,
                                columns=['user', 'app', 'op_time', 'op_id',
                                         'row_id', 'column',
                                         'old_value', 'new_value'])

            logs['column'] = self._col_ids_to_names(logs.column.values)
            for c in ['user', 'app', 'column']:
                logs[c] = logs[c].astype('category')

        return logs

    @check_token
    def fetch_meta(self):
        """Fetch/update meta data for table and base."""
        self.base_meta = self.base.get_metadata()

        meta = [t for t in self.base_meta['tables'] if t['name'] == self.name or t['_id'] == self.name]

        if len(meta) == 0:
            raise ValueError(f'No table with name "{self.name}" in base')
        elif len(meta) > 1:
            raise ValueError(f'Multiple tables with name "{self.name}" in base')

        meta = meta[0]

        # Check for duplicate columns
        seen = []
        for col in meta.get('columns', []):
            if col['name'] in seen:
                raise ValueError(f'Table {self.name} contains duplicate '
                                 f'column(s): {col["name"]}')
            seen.append(col['name'])

        self._meta = meta
        return self._meta

    @check_token
    def get_view(self, view, hide_cols=True, sort=True):
        """Download given view of tthe table.

        Applies filters, sorts and hidden columns. Groupings are ignored.

        Parameters
        ----------
        view :      str | int
                    Name or index of the view.
        hide_cols : bool
                    Whether to exclude columns hidden in this view.
        sort :      bool
                    Whether to apply same sort as in view.

        Returns
        -------
        pd.DataFrame

        """
        if isinstance(view, str):
            if view not in self.views:
                raise ValueError(f'"{view}" not found')
            view = [v for v in self.meta['views'] if v['name'] == view]
            if len(view) > 1:
                raise ValueError(f'Found multiple views with name "{view}". '
                                 'Consider using an index instead.')
            view = view[0]
        elif isinstance(view, int):
            view = self.meta['views'][view]
        else:
            raise TypeError(f'Expected `view` to be str or int, got "{type(view)}"')

        # Create filters:
        # We need to group the filters by field and predicate to avoid producing
        # queries like "(fieldA != 'value1') AND (fieldB != 'value2')"
        # Instead we need to combine them to
        # "(fieldA not in ('value1', 'value2')"
        view['filters_grp'] = {}
        for f in view['filters']:
            # Translate col IDs to names while we're at it
            col_name = [c['name'] for c in self.meta['columns'] if c['key'] == f['column_key']][0]
            if col_name not in view['filters_grp']:
                view['filters_grp'][col_name] = {}

            pred = f['filter_predicate']
            if pred not in view['filters_grp'][col_name]:
                view['filters_grp'][col_name][pred] = []
            view['filters_grp'][col_name][pred].append(f['filter_term'])

        filters = []
        for col_name in view['filters_grp']:
            # Get the column with this name
            col = self[col_name]

            # Map IDs to actual values (if applicable)
            for predicate in view['filters_grp'][col_name]:
                terms = view['filters_grp'][col_name][predicate]

                if predicate == 'is' and len(terms) > 1:
                    predicate = 'is_any_of'
                elif predicate == 'is_not' and len(terms) > 1:
                    predicate = 'is_none_of'

                terms = flatten(terms)
                terms = col._ids_to_values(terms)

                # Flatten a potential list of lists
                if predicate == 'is':
                    filters.append(col == terms[0])
                elif predicate == 'is_not':
                    filters.append(col != terms[0])
                elif predicate == 'is_not_empty':
                    filters.append(col.notnull())
                elif predicate == 'is_empty':
                    filters.append(col.isnull())
                elif predicate == 'is_none_of':
                    filters.append(~col.isin(terms))
                elif predicate == 'is_any_of':
                    filters.append(col.isin(terms))
                elif predicate == 'contains':
                    for t in terms:
                        filters.append(col.contains(t))
                elif predicate == 'does_not_contain':
                    for t in terms:
                        filters.append(~col.contains(t))
                else:
                    raise ValueError(f'Unsupported filter predicate: "{f["filter_predicate"]}"')

        conj = view.get('filter_conjunction', 'AND')
        comb = Filter(f' {conj} '.join([f'({f.query})' for f in filters]))

        if hide_cols:
            hidden_cols = self._col_ids_to_names(view.get('hidden_columns', []))
            cols = [c for c in self.columns if c not in hidden_cols]

            data = self.loc[comb, cols]
        else:
            data = self.loc[comb]

        if sort and view.get('sorts', None):
            cols = self._col_ids_to_names([s['column_key'] for s in view['sorts']])
            asc = [True if s['sort_type'] == 'up' else False for s in view['sorts']]

            data = data.sort_values(cols, ascending=asc)

        return data

    def head(self, n=5):
        """Return top N rows as pandas DataFrame."""
        data = self.base.query(f'SELECT * FROM {self.name} LIMIT {n}')
        return process_records(data, columns=self.columns,
                               dtypes=self.dtypes.to_dict() if self.sanitize else None)

    @write_access
    @check_token
    def link(self, other_table, link_on=None, link_on_other=None,
             link_col=None, multi_match=True):
        """Link rows in this table to rows in other table.

        Parameters
        ----------
        other_table :   str
                        Name of other table (must be in the same base).
        link_on :       str, optional
                        Column in this table that we will link on. If ``None``
                        will use the index column.
        link_on_other : str, optional
                        Column in other table to link on. If ``None`` will use
                        the index column.
        link_col :      str, optional
                        Column used for storing the links. Will be created if it
                        doesn't exist and updated if it does. If ``None``,
                        will use `link_{other_table}` as name!
        multi_match :   bool
                        Whether to allow matching a row in this table to multiple
                        rows in other table. If False, will pick the first row
                        that matches and ignore any other matches.

        See Also
        --------
        Table.add_linked_column
                        For a column utilizing the links to pull records from
                        other table.

        """
        other = Table(other_table, base=self.base)

        if link_on_other and link_on_other not in other.columns:
            raise ValueError(f'Column to link on "{link_on_other}" does not '
                             'exist in other table.')
        elif not link_on_other:
            link_on_other = other.columns[0]

        # Update meta data for self
        _ = self.fetch_meta()

        if link_on and link_on not in self.columns:
            raise ValueError(f'Column to link on "{link_on}" does not exist in table.')
        elif not link_on:
            link_on = self.columns[0]

        if not link_col:
            link_col = f'link_{other_table}'

        # Get the data we need to merge on
        link_data = self[link_on].to_series()
        link_data_other = other[link_on_other].to_series()

        # Turn other into a dict of {value: [row_id1, row_id2, ...]}
        link_dict_other = link_data_other.reset_index(drop=False).groupby(link_data_other.name).row_id.apply(list).to_dict()

        # Map row IDs in this DataFrame to those in others
        # {row_id: [other_row_id1, other_row_id2], ...}
        other_rows_ids_map = {}
        for row_id, value in zip(link_data.index.values, link_data.values):
            if value in link_dict_other:
                other_rows_ids_map[row_id] = link_dict_other[value]

        # If no multimatch allowed, drop anything but the first match
        if not multi_match:
            other_rows_ids_map = {k: v[:1] for k, v in other_rows_ids_map.items()}

        # Add link column if doesn't already exists:
        if link_col not in self.columns:
            self.add_column(col_name=link_col, col_type='link',
                            col_data={'table': self.name,
                                      'other_table': other_table,
                                      'is_multiple': False})

        # Add obsolete links that need to be removed (= set them to [])
        other_rows_ids_map.update({k: [] for k in link_data.index.values if k not in other_rows_ids_map})

        link_id = self[link_col].meta['data']['link_id']
        table_id = self.id
        other_table_id = other.id
        row_id_list = list(other_rows_ids_map.keys())

        # Update links in batches
        for i in trange(0, len(row_id_list), self.max_operations,
                        disable=len(row_id_list) < self.max_operations,
                        desc='Linking'):
            batch_row_id_list = row_id_list[i: i + self.max_operations]
            batch_other_rows_ids_map = {k: v for k, v in other_rows_ids_map.items() if k in batch_row_id_list}

            self.base.batch_update_links(link_id,
                                         table_id,
                                         other_table_id,
                                         batch_row_id_list,
                                         batch_other_rows_ids_map)

    def to_frame(self, row_id_index=True):
        """Download entire table as pandas DataFrame."""
        data = self.base.query(f'SELECT * FROM {self.name} LIMIT {self.shape[0]}')
        return process_records(data,
                               columns=self.columns,
                               row_id_index=row_id_index,
                               dtypes=self.dtypes.to_dict() if self.sanitize else None)

    @check_token
    def query(self, query, no_limit=False, convert=True):
        """Run SQL query against this table.

        Parameters
        ----------
        query :     str | Filter
                    The SQL query. The "FROM {TABLENAME}" will be automatically
                    added to the end of the query if not already present.
        no_limit :  bool
                    By default, the SeaTable API limits queries to 100 rows. If
                    True, we will override this by adding `LIMIT {table.shape[0]}`
                    to the query.
        convert :   bool
                    Whether to process raw values into something more readable.
                    Importantly this parses single/multi-select values and
                    date columns into datetime objects.

        Returns
        -------
        list
                    Records (list of dicts) contain the results of the query.

        """
        if isinstance(query, Filter):
            query = f'SELECT * from {self.name} WHERE {query.query}'

        if 'from' not in query.lower():
            query = f'{query} FROM {self.name}'
        if no_limit and 'LIMIT' not in query:
            query += f' LIMIT {self.shape[0]}'
        logger.debug(f'Running SQL query: {query}')
        return self.base.query(query, convert=convert)


class Column:
    """Class representing a table column."""

    def __init__(self, name, table):
        self.name = name
        self.table = table

    def __array__(self, dtype=None):
         return np.array(self.values, dtype=dtype)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Column <column={self.name}, table="{self.table.name}", datatype={self.dtype}>'

    def __len__(self):
        """Number of rows in column."""
        return len(self.table)

    def __eq__(self, other):
        _ = validate_comparison(self, other)
        if isinstance(other, str):
            # Escape the value
            return Filter(f"{self.name} = '{other}'")
        else:
            # This is assuming other is a number
            return Filter(f"{self.name} = {other}")

    def __ne__(self, other):
        return Filter((self == other).query.replace('=', '!='))

    def __ge__(self, other):
        return Filter((self > other).query.replace('>', '>='))

    def __gt__(self, other):
        _ = validate_comparison(self, other)
        if isinstance(other, numbers.Number):
            return Filter(f"{self.name} > {other}")
        raise TypeError(f"'>' not supported between column and {type(other)}")

    def __le__(self, other):
        return Filter((self < other).query.replace('<', '<='))

    def __lt__(self, other):
        _ = validate_comparison(self, other)
        if isinstance(other, numbers.Number):
            return Filter(f"{self.name} < {other}")
        raise TypeError(f"'>' not supported between column and {type(other)}")

    def __and__(self, other):
        if isinstance(other, Column):
            if other.dtype == 'checkbox':
                return Filter(f'{self.name} AND {other.name}')
        elif isinstance(other, Filter):
            if self.dtype == 'checkbox':
                return Filter(f'{self.name}') & other
            else:
                raise TypeError('Unable to construct filter from column of '
                                f'type {self.dtype}')

        raise TypeError(f'Unable to combine Column and "{type(other)}"')

    def __or__(self, other):
        if isinstance(other, Column):
            if other.dtype == 'checkbox':
                return Filter(f'{self.name} OR {other.name}')
        elif isinstance(other, Filter):
            if self.dtype == 'checkbox':
                return Filter(f'{self.name}') | other
            else:
                raise TypeError('Unable to construct filter from column of '
                                f'type {self.dtype}')

        raise TypeError(f'Unable to combine Column and "{type(other)}"')

    def __contains__(self, value):
        # Add quotation marks to string
        if isinstance(value, str):
            value = f"'{value}'"
        q = (f"SELECT {self.name} "
             f"FROM {self.table.name} "
             f"WHERE {self.name} = {value} "
             "LIMIT 1")
        return any(self.table.query(q))

    @property
    def key(self):
        """Unique identifier of this column."""
        keys = []
        for col in self.table.meta['columns']:
            if col['name'] == self.name:
                keys.append(col['key'])

        if len(keys) > 1:
            raise ValueError(f'Found multiple columns with name "{self.name}"')

        return keys[0]

    @property
    def dtype(self):
        return self.meta['type']

    @property
    def meta(self):
        """Meta data for this column."""
        if self.name == '_id':
            return {'type': str, 'key': None}
        else:
            return [c for c in self.table.meta['columns'] if c['name'] == self.name][0]

    @property
    def options(self):
        """Options for single- or multi-select columns."""
        if 'select' not in self.dtype:
            raise TypeError('`options` only exists for single- or multi-select '
                            'columns')
        return np.array([r['name'] for r in self.meta['data']['options']])

    @property
    def values(self):
        return self.to_series().values

    def _ids_to_values(self, ids):
        """Map ID(s) (e.g. for single-select Columns) to values.

        Will simply pass-through if column doesn't use IDs.

        Parameters
        ----------
        ids :   str | iterable
                The IDs to map. E.g. "399046".

        Returns
        -------
        value(s)

        """
        if 'data' not in self.meta or not self.meta['data']:
            return ids
        if 'options' not in self.meta['data']:
            return ids

        id_map = {r['id']: r['name'] for r in self.meta['data']['options']}

        if not is_iterable(ids):
            return id_map[ids]

        return [id_map[i] for i in ids]

    def astype(self, dtype, errors='raise'):
        """Download and cast data to specified dtype ``dtype``.

        Parameters
        ----------
        dtype :     data type, or dict of column name -> data type
                    Use a numpy.dtype or Python type to cast entire column to.
        errors :    'raise' | 'ignore'
                    Control raising of exceptions on invalid data for provided
                    dtype.

                        - ``raise`` : allow exceptions to be raised
                        - ``ignore`` : suppress exceptions.

        Returns
        -------
        np.ndarray

        """
        return self.to_series().astype(dtype, errors=errors)

    def to_series(self):
        """Return this column as pandas.Series."""
        if self.name != '_id':
            query = f'SELECT {self.name}, _id'
        else:
            query = f'SELECT {self.name}'
        rows = self.table.query(query, no_limit=True)
        return process_records(rows,
                               row_id_index=self.name != '_id',
                               dtypes={self.name: self.dtype} if self.table.sanitize else None
                               ).iloc[:, 0]

    @write_access
    @check_token
    def clear(self):
        """Clear this column."""
        if not self.key:
            raise ValueError(f'Unable to clear column {self.name}')

        row_ids = self.table.row_ids

        records = [{'row_id': r,
                    'row': {self.name: None}} for r in row_ids]

        if not self.table._hold:
            r = batch_upload(partial(self.table.base.batch_update_rows, self.table.name),
                             records, desc='Clearing',
                             batch_size=self.table.max_operations,
                             progress=self.table.progress)

            if 'success' in r:
                logger.info('Clear successful!')
        else:
            self.table._queue += records

    @write_access
    @check_token
    def delete(self):
        """Delete this column."""
        if not self.key:
            raise ValueError(f'Unable to delete column {self.name}')

        self.table._stale = True
        resp = self.table.base.delete_column(self.table.name, self.key)

        if not resp.get('success'):
            raise ValueError(f'Error writing to table: {resp}')

        # Update table meta data
        _ = self.table.fetch_meta()

        logger.info(f'Column "{self.name}" deleted.')

    def contains(self, pat):
        """Filter to strings containing given substring."""
        if not isinstance(pat, str):
            raise TypeError(f'`pat` must be str, not "{type(pat)}"')

        if self.dtype == 'text':
            return Filter(f"{self.name} LIKE '%{pat}%'")
        elif self.dtype in ('single-select', 'multiple-select'):
            return self.isin([o for o in self.options if pat in o])
        raise ValueError('Can only Filter by substring if Column is of '
                         f'type "text" or single-/multiple-select, not {self.dtype}')

    def startswith(self, pat):
        """Filter to strings starting with given substring."""
        if self.dtype != 'text':
            raise ValueError('Can only Filter by substring if Column is of '
                             f'type "text", not {self.dtype}')
        if not isinstance(pat, str):
            raise TypeError(f'`pat` must be str, not "{type(pat)}"')
        return Filter(f"{self.name} LIKE '{pat}%'")

    def endswith(self, pat):
        """Filter to strings ending with given substring."""
        if self.dtype != 'text':
            raise ValueError('Can only Filter by substring if Column is of '
                             f'type "text", not {self.dtype}')
        if not isinstance(pat, str):
            raise TypeError(f'`pat` must be str, not "{type(pat)}"')
        return Filter(f"{self.name} LIKE '%{pat}'")

    def isin(self, other, online=True):
        """Filter to values in `other`.

        Parameters
        ----------
        other :     iterable
        online :    bool | "auto"
                    Whether to use an "online" Filter query or to download the
                    column and run the isin query offline. The latter
                    makes sense if the `other` contains many thousand of
                    values.

        Returns
        ------
        Filter
                    A Filter query (if online=True).
        pandas.Series
                    A boolean series.

        """
        if isinstance(other, pd.Series):
            other = other.values

        if not is_iterable(other):
            return self == other

        if len(other) == 0:
            raise ValueError('Unable to compare empty container')
        elif len(other) == 1:
            return self == other[0]

        _ = validate_comparison(self, other, allow_iterable=True)

        if online:
            other = tuple(other)

            return Filter(f"{self.name} IN {str(other)}")
        else:
            return self.to_series().isin(other)

    def isnull(self, empty_str=True):
        """Filter for NULL.

        Parameters
        ----------
        empty_str :     bool
                        Whether to treat empty strings as null. Only relevant
                        for text columns.

        Returns
        -------
        Filter

        """
        if empty_str and self.dtype == 'text':
            return Filter(f"{self.name} IS NULL or {self.name} = ''")
        else:
            return Filter(f"{self.name} IS NULL")

    def notnull(self, empty_str=True):
        """Filter for not NULL.

        Parameters
        ----------
        empty_str :     bool
                        Whether to treat empty strings as null. Only relevant
                        for text columns.

        Returns
        -------
        Filter

        """
        if empty_str and self.dtype == 'text':
            return Filter(f"{self.name} IS NOT NULL and {self.name} != ''")
        else:
            return Filter(f"{self.name} IS NOT NULL")

    def map(self, arg, na_action=None):
        """Map values of columns according to input correspondence.

        Parameters
        ----------
        arg :       function, collections.abc.Mapping subclass or Series
                    Mapping correspondence.
        na_action : None | 'ignore'
                    If 'ignore', propagate NaN values, without passing them to
                    the mapping correspondence.

        Returns
        -------
        Series
                    Same index as caller.

        """
        return self.to_series().map(arg, na_action=na_action)

    @write_access
    @check_token
    def rename(self, new_name):
        """Rename column.

        Parameters
        ----------
        new_name :  str
                    The new name of the column.

        """
        if not isinstance(new_name, str):
            raise TypeError(f'New name must be str, not "{type(new_name)}"')

        if new_name in self.table.columns:
            raise ValueError(f'Column name "{new_name}" already exists.')

        self.table._stale = True

        resp = self.table.base.rename_column(self.table.name,
                                             self.key,
                                             new_name)

        self.table._stale = True

        if 'name' not in resp:
            raise ValueError(f'Error writing to table: {resp}')

        logger.info(f'Column renamed: {self.name} -> {self.new_name}')

        # Update table meta data
        _ = self.table.fetch_meta()

        self.name = new_name

    @write_access
    @check_token
    def resize(self, width):
        """Resize column.

        Parameters
        ----------
        width :     int
                    The new width of the column.

        """
        try:
            width = int(width)
        except ValueError:
            raise TypeError(f'New width must be integer, not "{type(width)}"')
        except BaseException:
            raise

        resp = self.table.base.resize_column(self.table.name,
                                             self.key,
                                             width)
        self.table._stale = True

        if 'name' not in resp:
            raise ValueError(f'Error writing to table: {resp}')

        logger.debug(f'Column {self.name} resized.')

    @write_access
    @check_token
    def freeze(self):
        """Freeze column.

        Use unfreeze() method to unfreeze column.

        """
        resp = self.table.base.freeze_column(self.table.name,
                                             self.key,
                                             frozen=True)
        if 'name' not in resp:
            raise ValueError(f'Error writing to table: {resp}')

        logger.debug(f'Column {self.name} frozen.')

    @write_access
    @check_token
    def unfreeze(self):
        """Unfreeze column.

        Use freeze() method to freeze column.

        """
        resp = self.table.base.freeze_column(self.table.name,
                                             self.key,
                                             frozen=False)
        if 'name' not in resp:
            raise ValueError(f'Error writing to table: {resp}')

        logger.debug(f'Column {self.name} unfrozen.')

    def unique(self):
        """Return unique values in this column."""
        rows = self.table.query(f'SELECT DISTINCT {self.name}', no_limit=True)

        return process_records(rows,
                               dtypes=self.dtype if self.table.sanitize else None
                               ).iloc[:, 0].values

    def update(self, values):
        """Update this column with given values.

        The main difference between this and simply setting them is that here
        we make sure to only write values that have changed back to the table.

        Parameters
        ----------
        values :    iterable
                    Must be of same length and data type as the column.

        """
        values = np.asarray(values)

        l = len(self)
        if len(values) != l:
            raise ValueError(f'Length of values ({len(values)}) does not '
                             f'match length of column ({l})')

        # Convert np.nan to None
        # This is required because we want to treat np.nan like None
        # (i.e. no/empty value) but np.nan is != None
        values[pd.isnull(values)] = None

        # Find values that need updating. Note we're using a dedicated function
        # to avoid issues with pandas StringType (see `is_equal_array` for
        # further explanation)
        needs_update = ~is_equal_array(self.values, values)

        if any(needs_update):
            self.table.loc[needs_update, self.name] = values[needs_update]

    def value_counts(self, **kwargs):
        """Return a Series contain counts of unique values.

        Parameters
        ----------
        **kwargs
                    Keyword arguments are passed through to
                    `pandas.Series.value_counts()`.

        Returns
        -------
        counts :    pandas.Series

        """
        return self.to_series().value_counts(**kwargs)

    @write_access
    def add_options(self, options):
        """Add options for this column.

        Only works for single- and multi-select columns.

        Parameters
        ----------
        options :   iterable
                    A list/array of strings.

        """
        if self.dtype not in ('single-select', 'multi-select'):
            raise TypeError('Can only set options for single- or multi-select '
                            f'columns. This column is of type "{self.dtype}".')

        if not isinstance(options, (list, set, np.ndarray)):
            raise ValueError('`options` must be list, set or array - got '
                             f'type({self.options})')

        if not all([isinstance(e, dict) for e in options]):
            payload = [{
                        'name': str(o),
                        'color': '#aaa',
                        'textColor': '#000000'
                        } for o in options if o]
        else:
            payload = options

        self.table.base.add_column_options(self.table.name, self.name, payload)

        # Update table meta data
        _ = self.table.fetch_meta()

        logger.info(f'Added column options: {options}')


class Filter:
    """Class representing an SQL WHERE query."""

    def __init__(self, query):
        self.query = query

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'SQL filter query <"{self.query}">'

    def __invert__(self):
        if ' AND ' in self.query or ' or ' in self.query:
            raise ValueError('Unable to invert Filter combinations')
        elif ' = ' in self.query:
            return Filter(self.query.replace(' = ', ' != '))
        elif ' != ' in self.query:
            return Filter(self.query.replace(' != ', ' = '))
        elif ' NOT IN ' in self.query:
            return Filter(self.query.replace(' NOT IN ', ' IN '))
        elif ' IN ' in self.query:
            return Filter(self.query.replace(' IN ', ' NOT IN '))
        elif ' IS NOT ' in self.query:
            return Filter(self.query.replace(' IS NOT ', ' IS '))
        elif ' IS ' in self.query:
            return Filter(self.query.replace(' IS ', ' IS NOT '))
        elif ' NOT LIKE ' in self.query:
            return Filter(self.query.replace(' NOT LIKE ', ' LIKE '))
        elif ' LIKE ' in self.query:
            return Filter(self.query.replace(' LIKE ', ' NOT LIKE '))
        else:
            raise ValueError(f'Unable to invert Filter "{self.query}"')

    def __and__(self, other):
        if isinstance(other, Filter):
            return Filter(f'({self.query}) AND ({other.query})')
        elif isinstance(other, Column):
            if other.dtype == 'checkbox':
                return Filter(f'({self.query}) AND {other.name}')
            raise TypeError('Unable to combine Filter and column of type '
                            f'"{other}.dtype"')

        raise TypeError(f'Unable to combine Filter and "{type(other)}"')

    def __or__(self, other):
        if isinstance(other, Filter):
            return Filter(f'({self.query}) OR ({other.query})')
        elif isinstance(other, Column):
            if other.dtype == 'checkbox':
                return Filter(f'({self.query}) OR ({other.name})')

        raise TypeError(f'Unable to combine Filter and "{type(other)}"')


class LocIndexer:
    def __init__(self, table):
        self.table = table

    @property
    def read_only(self):
        """Pass through read-only property from table."""
        return self.table.read_only

    def __getitem__(self, key):
        limit = None
        if type(key) is tuple:
            if len(key) == 1:
                where, cols = key, '*'
            elif len(key) == 2:
                where, cols = key
            elif len(key) == 3:
                where, cols, limit = key
            elif len(key) > 3:
                raise IndexError(f'Too many indexers ({len(key)})')
            else:
                raise IndexError(f'Unable to use indexer "{key}"')
        else:
            where, cols = key, '*'

        if isinstance(where, pd.Series):
            where = where.values

        # If where is boolean mask fetch the entire frame
        if is_array_like(where) and (where.dtype in (bool, pd.BooleanDtype())):
            query = create_query(self.table, columns=cols, where=None, limit=limit)
        else:
            query = create_query(self.table, columns=cols, where=where, limit=limit)

        records = self.table.query(query, no_limit=True)
        data = process_records(records,
                               row_id_index=isinstance(cols, str) and cols != '_id',
                               dtypes=self.table.dtypes.to_dict() if self.table.sanitize else None)

        # Reindex columns so that we have columns even if data is empty
        if cols != '*':
            data = data.reindex(make_iterable(cols), axis=1)
        else:
            # This sorts columns like in the UI
            data = data.reindex(self.table.columns, axis=1)

        # If index was boolean mask subset to requested rows
        if is_array_like(where) and (where.dtype in (bool, pd.BooleanDtype())):
            data = data.loc[where].copy()

        # If a single row was requested
        if isinstance(key, int):
            data = data.iloc[0]

        # If a single column was requested
        if isinstance(cols, str) and cols != '*':
            data = data[cols]

        return data

    @write_access
    @check_token
    def __setitem__(self, key, values):
        if not isinstance(key, tuple):
            if isinstance(values, pd.DataFrame):
                for col in values.columns:
                    if col not in self.table.columns:
                        logger.warning(f'Skipping column "{col}": '
                                       'not present in table.')
                    self[key, col] = values[col]
                return
            else:
                raise KeyError('Must provide DataFrame when writing to table '
                               'using the .loc Indexer without specifying the '
                               'column.')
        elif len(key) != 2:
            raise IndexError(f'Wrong number of indexers ({len(key)})')

        where, col = key

        if col not in self.table.columns:
            raise KeyError('Column {col} must exists, use `add_column()` method to '
                           f'create "{col}" before setting its values')

        if isinstance(where, pd.Series):
            where = where.values

        if is_iterable(where):
            where = np.asarray(where)

            # If boolean series
            if where.dtype == bool:
                row_ids = self.table.row_ids[where]
            else:
                # Check if these are row IDs
                is_id = np.isin(where, self.table.row_ids)

                if all(is_id):
                    row_ids = where
                else:
                    raise KeyError('Can only index by boolean iterable or '
                                   'iterable of row IDs.')
        else:
            row_ids = self[where, '_id'].values

        if isinstance(values, (pd.Series, Column)):
            values = values.values

        if not is_iterable(values):
            values = [values] * len(self.table)
        elif len(values) != len(row_ids):
            raise ValueError(f'Length of values ({len(values)}) does not '
                             f'match length of rows ({len(row_ids)})')

        # Validate datatype
        validate_dtype(self.table, col, values)

        # This checks for potential int64 -> int32 issues
        values = validate_values(values, col=self.table[col])

        records = [{'row_id': r,
                    'row': {col: v if not pd.isnull(v) else None}} for r, v in zip(row_ids, values)]

        if not self.table._hold:
            r = batch_upload(partial(self.table.base.batch_update_rows,
                                     self.table.name),
                             records,
                             batch_size=self.table.max_operations,
                             progress=self.table.progress)

            if 'success' in r:
                logger.info(f'Successfully wrote to "{col}"!')
        else:
            self.table._queue += records


class iLocIndexer:
    def __init__(self, table):
        self.table = table

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = self.parse_slice(key)

            if step:
                warnings.warn(f'Step {step} is applied only after the data has '
                              'been downloaded.')

            # Construct the query
            if start and stop:
                q = f'SELECT * FROM {self.table.name} LIMIT {start}, {stop}'
            elif start:
                q = f'SELECT * FROM {self.table.name} LIMIT {start}, {self.table.shape[0] - start}'
            else:
                q = f'SELECT * FROM {self.table.name} LIMIT {stop}'

            data = self.table.query(q)

            # Must apply step afterwards
            if step:
                data = data[::step]

            return process_records(data, columns=self.table.columns)

    def parse_slice(self, s):
        if s.start and s.start < 0:
            start = self.table.shape[0] + s.start
        else:
            start = s.start

        if s.stop and s.stop < 0:
            if not start:
                stop = self.table.shape[0] + s.stop
            else:
                stop = self.table.shape[0] + s.stop - start
        else:
            stop = s.stop

        return start, stop, s.step


def create_query(table, columns=None, where=None, limit=None):
    """Create SQL query."""
    if isinstance(where, slice) and not isinstance(limit, type(None)):
        raise TypeError('Unable to construct WHERE query with limit and slice '
                        'as index')

    if not isinstance(columns, type(None)) and columns != '*':
        columns = make_iterable(columns).astype(str)
        if len(columns) == 1:
            q = f'SELECT {columns[0]}'
        else:
            q = f'SELECT {", ".join(columns)}'
    else:
        q = ' SELECT *'

    q += f' FROM {table.name}'

    if not isinstance(where, type(None)):
        if isinstance(where, Filter):
            q += f' WHERE {where.query}'
        elif isinstance(where, Column):
            if not where.dtype == 'checkbox':
                raise TypeError('Can only query by columns with dtype '
                                f'"checkbox", got "{where.dtype}"')
            q += f' WHERE {where.name} = True'
        elif isinstance(where, slice):
            if slice.start == slice.stop:
                raise KeyError('Slice start and stop must not be the same')
            elif where.step:
                raise KeyError('Unable to use slice with step for indexing')

            if where.start and where.start < 0:
                start = table.shape[0] + where.start
            else:
                start = where.start

            if where.stop and where.stop < 0:
                stop = table.shape[0] + where.stop
            else:
                stop = where.stop

            if start and stop:
                q += f' LIMIT {start}, {stop-start}'
            elif start:
                q += f' LIMIT {start}, {table.shape[0] - start}'
            elif stop:
                q += f' LIMIT {stop}'
        elif isinstance(where, int):
            q += f' LIMIT {where}, 1'
        elif isinstance(where, str):
            if where not in table.row_ids:
                ValueError(f'"{where}" does not appear to be a valid row id')
            q += f" WHERE _id = '{where}'"
        elif is_iterable(where):
            if not all(np.isin(where, table.row_ids)):
                raise ValueError('Expected iterable to be a list of valid row IDs.')
            q_str = '("' + '", "'.join(where) + '")'
            q += f" WHERE _id IN {q_str}"
        else:
            raise TypeError(f'Unable to construct WHERE query from "{type(where)}"')

    if isinstance(limit, numbers.Number):
        q += f' LIMIT {limit}'

    return q


def batch_upload(func, records, batch_size=1000, desc='Writing',
                 omit_errors=False, batch_param='rows_data', progress=True):
    """Upload/update rows in batches of defined size."""
    no_errors = True
    with tqdm(desc=desc,
              total=len(records),
              disable=len(records) < batch_size or not progress) as pbar:

        for i in range(0, len(records), batch_size):
            batch = records[i: i + batch_size]

            r = func(**{batch_param: batch})

            # Catching error messages for the different functions is a bit hit and
            # miss without a documented schema
            if r.get('success', False):
                continue
            if 'inserted_row_count' in r:
                continue
            if 'deleted_rows' in r:
                continue

            msg = f'Error editing table (batch {int(i / batch_size)}/{len(records) // batch_size + 1}): {r}'
            if not omit_errors:
                raise ValueError(msg)
            else:
                logger.error(msg)
                no_errors = False

            pbar.update(len(batch))
            pbar.refresh()  # for some reason this is necessary

    return {'success'} if no_errors else {'errors'}


class BundleEdits:
    """Context manager delays uploading edits until after exit.

    Can be useful if you are changing many values at a time and want to wait
    till you are done for pushing the changes to Seatable.

    Currently this only works for updating rows.
    """
    def __init__(self, table):
        if not isinstance(table, Table):
            raise TypeError(f'Expected `seaserpent.Table`, got {type(table)}')
        self.table = table
        self.nested = False

    def __enter__(self):
        # Deal with cases of nested bundling
        if self.table._hold:
            self.nested = True
        self.table._hold = True

    def __exit__(self, type, value, traceback):
        if not self.nested:
            if self.table._queue:
                # We need to combine records with the same row ID because if a
                # single batch contains records that point tot he same row it will
                # apparently only use the last edit
                rows = {}
                for r in self.table._queue:
                    rid = r['row_id']
                    rows[rid] = rows.get(rid, {'row_id': rid, 'row': {}})
                    rows[rid]['row'].update(r['row'])

                r = batch_upload(partial(self.table.base.batch_update_rows, self.table.name),
                                 list(rows.values()),
                                 batch_size=self.table.max_operations,
                                 progress=self.table.progress)

                if 'success' in r:
                    logger.info(f'Successfully wrote queue ({len(rows)} rows) to table!')
                    self.table._queue = []
                else:
                    logger.warning('Something went wrong while writing queue to '
                                   'table.')

            self.table._hold = False
