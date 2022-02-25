import logging
import json
import numbers
import requests
import warnings

import datetime as dt
import numpy as np
import pandas as pd

from functools import partial
from seatable_api.main import SeaTableAPI
from seatable_api.constants import ColumnTypes
from tqdm.auto import trange, tqdm

from .utils import (process_records, make_records,
                    is_iterable, make_iterable, is_hashable,
                    map_columntype, find_base, write_access,
                    validate_dtype, validate_comparison, validate_table,
                    validate_values, suppress_print)

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


class Table:
    """A remote data table.

    Parameters
    ----------
    table :         str | int
                    Name or index of the table. If index, you must provide a
                    base.
    base :          str | int | SeaTableAPI, optional
                    Name, ID or UUID of the base containing ``table``. If not
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
            (self.base,
             self.auth_token,
             self.server) = find_base(base=base,
                                      required_table=table if not isinstance(table, int) else None)
        else:
            self.base = base
            self.server = base.server_url
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

        self.base.query = suppress_print(self.base.query)

    def __array__(self, dtype=None):
         return np.array(self.values, dtype=dtype)

    def __dir__(self):
        """Custom __dir__ to make columns searchable."""
        return list(set(super().__dir__() + list(self.columns)))

    def __len__(self):
        """Length of table."""
        return self.shape[0]

    def __getattr__(self, name):
        if name not in self.columns:
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
    def __setitem__(self, key, values):
        if not is_hashable(key):
            raise KeyError('Key must be hashable (i.e. a single column). Use '
                           '.loc indexer to set values for a specific slice.')

        # Update meta data for self
        _ = self.fetch_meta()

        if key not in self.columns:
            raise KeyError('Column must exists, use `add_column()` method to '
                           f'create {"key"} before setting its values')

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
        values = validate_values(values, dtype=self.dtypes[key])

        # Fetch the IDs
        row_ids = self.query('SELECT _id', no_limit=True)

        records = [{'row_id': r['_id'],
                    'row': {key: v if not pd.isnull(v) else None}} for r, v in zip(row_ids, values)]

        r = batch_upload(partial(self.base.batch_update_rows, self.name),
                         records, batch_size=self.max_operations,
                         progress=self.progress)

        if 'success' in r:
            logger.info('Write successful!')

    def __repr__(self):
        shape = self.shape
        return f'SeaTable <"{self.name}", {shape[0]} rows, {shape[1]} columns>'

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
        return self['_id'].values

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
        """Create a new table from pandas DataFrame.

        Parameters
        ----------
        df :            pandas.DataFrame
                        DataFrame to export to SeaTable. Datatypes are inferred:
                          - object -> text
                          - int, float -> number
                          - bool -> check box
                          - categorical -> single select
        table_name :    str
                        Name of the new table.
        base :          str | int
                        Name or ID of base.
        id_col :        str | int
                        Name or index of the ID column to use.

        Returns
        -------
        Table
                        Will be initialized with `read_only=False`.

        """
        # Some sanity checks
        if len(df.columns) < len(np.unique(df.columns)):
            raise ValueError('Table must not contain duplicate column names')

        if isinstance(id_col, int):
            id_col = df.columns[id_col]

        if id_col not in df.columns:
            raise ValueError(f'ID column "{id_col}" not among columns.')

        # Validate table
        df = validate_table(df)

        if df[id_col].dtype == object:
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

            if col.dtype == object:
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
            else:
                options = None

            table.add_column(col_name=c, col_type=dtype, col_options=options)

        logger.info('New columns added.')

        # Add the actual data
        table.append(df)
        logger.info('Data uploaded.')

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
        base, token, server = find_base(base, auth_token=auth_token, server=server)

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
        miss = columns[~np.isin(columns, self.columns)]
        if any(miss):
            raise KeyError(f'"{miss}" not among columns')

    @write_access
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
    def append(self, other):
        """Append rows of `other` to the end of this table.

        Columns in `other` that are not also in the table are ignored.

        Parameters
        ----------
        other :     pandas.DataFrame

        """
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
                                         dtype=self.dtypes[col])

        records = make_records(other)

        r = batch_upload(partial(self.base.batch_append_rows, self.name),
                         records, desc='Appending',
                         batch_size=self.max_operations,
                         progress=self.progress)

        if 'success' in r:
            logger.info('Rows successfully added!')

    @write_access
    def delete_rows(self, rows):
        """Delete given rows.

        Parameters
        ----------
        rows :      int | iterable
                    Either a single row ID, a list of row IDs or a boolean
                    array of the same length as the table. IMPORTANT: this
                    expects Python indices, i.e. the first row will have index
                    0 (by contrast, the first row in in SeaTable has index 1)!

        """
        if isinstance(rows, int):
            rows = [rows]

        rows = np.asarray(rows)

        if rows.dtype != bool:
            rows = rows.astype(int)
        elif len(rows) != len(self):
            raise ValueError(f'Length of boolean array ({len(rows)}) does not '
                             f'match that of table ({len(self)})')

        row_ids = self.row_ids[rows]

        r = batch_upload(partial(self.base.batch_delete_rows, self.name),
                         row_ids.tolist(),
                         desc='Deleting',
                         batch_param='row_ids',
                         batch_size=self.max_operations,
                         progress=self.progress)

        if 'success' in r:
            logger.info('Rows successfully deleted!')

    def fetch_logs(self, max_entries=25):
        """Fetch activity logs for this table.

        Parameters
        ----------
        max_entries :   int
                        Maximum number of logs to return. We can fetch 25
                        entries in one query. Set to ``None`` to fetch all
                        logs (may take a while). Note that we might fall
                        short of this if there are multiple tables in the
                        same base.

        Returns
        -------
        pandas.DataFrame

        """
        page = 1
        logs = []
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

            if max_entries and (page * 25) >= max_entries:
                break

            page += 1

        # Combine
        logs = pd.concat(logs, axis=0)

        # Some clean-up:
        logs['operation'] = logs.operation.map(json.loads)

        # Drop irrelevant entries
        table = logs.operation.map(lambda x: x.get('table_id', None))
        logs = logs[table == self.id].copy()

        # Extract/parse relevant values
        logs['op_time'] = (logs.op_time / 1e3).map(dt.datetime.fromtimestamp)
        logs['op_type'] = logs.operation.map(lambda x: x['op_type'])

        users = self.collaborators.set_index('email').name.to_dict()
        logs.insert(0, 'user', logs.author.map(users))
        logs.drop('author', inplace=True, axis=1)

        logs['rows_modified'] = logs.operation.map(lambda x: len(x.get('row_ids', [])))
        logs.loc[logs.op_type == 'modify_row', 'rows_modified'] = 1

        col = logs.pop('operation')
        logs['details'] = col

        return logs.reset_index(drop=True)

    def fetch_meta(self):
        """Fetch/update meta data for table and base."""
        self.base_meta = self.base.get_metadata()

        meta = [t for t in self.base_meta['tables'] if t['name'] == self.name or t['_id'] == self.name]

        if len(meta) == 0:
            raise ValueError(f'No table with name "{self.name}" in base')
        elif len(meta) > 1:
            raise ValueError(f'Multiple tables with name "{self.name}" in base')

        self._meta = meta[0]

        return meta[0]

    def head(self, n=5):
        """Return top N rows as pandas DataFrame."""
        data = self.base.query(f'SELECT * FROM {self.name} LIMIT {n}')
        return process_records(data, columns=self.columns,
                               dtypes=self.dtypes.to_dict() if self.sanitize else None)

    @write_access
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
        if name == '_id':
            self.meta = {'type': str, 'key': None}
        else:
            self.meta = [c for c in table.meta['columns'] if c['name'] == name][0]

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
        return Filter((self > other).query.replace('<', '<='))

    def __lt__(self, other):
        _ = validate_comparison(self, other)
        if isinstance(other, numbers.Number):
            return Filter(f"{self.name} < {other}")
        raise TypeError(f"'>' not supported between column and {type(other)}")

    def __and__(self, other):
        if isinstance(other, Column):
            if other.dtype == 'checkbox':
                return Filter(f'{self.name} AND {other.name}')

        raise TypeError(f'Unable to combine Column and "{type(other)}"')

    def __or__(self, other):
        if isinstance(other, Column):
            if other.dtype == 'checkbox':
                return Filter(f'{self.name} OR {other.name}')

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
    def values(self):
        return self.to_series().values

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
        rows = self.table.query(f'SELECT {self.name}', no_limit=True)
        return process_records(rows,
                               row_id_index=self.name != '_id',
                               dtypes={self.name: self.dtype} if self.table.sanitize else None
                               ).iloc[:, 0]

    @write_access
    def clear(self):
        """Clear this column."""
        if not self.key:
            raise ValueError(f'Unable to clear column {self.name}')

        row_ids = self.table.row_ids

        records = [{'row_id': r,
                    'row': {self.name: None}} for r in row_ids]

        r = batch_upload(partial(self.table.base.batch_update_rows, self.table.name),
                         records, desc='Clearing',
                         batch_size=self.table.max_operations,
                         progress=self.table.progress)

        if 'success' in r:
            logger.info('Clear successful!')

    @write_access
    def delete(self):
        """Delete this column."""
        if not self.key:
            raise ValueError(f'Unable to delete column {self.name}')

        self.table._stale = True
        resp = self.table.base.delete_column(self.table.name, self.key)

        if not resp.get('success'):
            raise ValueError(f'Error writing to table: {resp}')
        logger.info(f'Column "{self.name}" deleted.')

    def contains(self, pat):
        """Filter to strings containing given substring."""
        if self.dtype != 'text':
            raise ValueError('Can only Filter by substring if Column is of '
                             f'type "text", not {self.dtype}')
        if not isinstance(pat, str):
            raise TypeError(f'`pat` must be str, not "{type(pat)}"')
        return Filter(f"{self.name} LIKE '%{pat}%'")

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

    def isin(self, other):
        """Filter to values in `other`."""
        if not is_iterable(other):
            return self == other

        if len(other) == 0:
            raise ValueError('Unable to compare empty container')
        elif len(other) == 1:
            return self == other[0]

        _ = validate_comparison(self, other, allow_iterable=True)
        other = tuple(other)

        return Filter(f"{self.name} IN {str(other)}")

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

        self.name = new_name

    def unique(self):
        """Return unique values in this column."""
        rows = self.table.query(f'SELECT DISTINCT {self.name}', no_limit=True)

        # Rows is a list of dicts where ['{colum: value1}, {column: value2}']
        # Empty strings and Nones appear to show up as e.g. [{column_id: None}]
        # Here we will force the column_id entries to column names
        rows = [{self.name: list(r.values())[0]} for r in rows]

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

        # Find values that need updating
        needs_update = self.values != values

        if any(needs_update):
            self.table.loc[needs_update, self.name] = values[needs_update]

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
                where, cols = key, self.table.columns
            elif len(key) == 2:
                where, cols = key
            elif len(key) == 3:
                where, cols, limit = key
            elif len(key) > 3:
                raise IndexError(f'Too many indexers ({len(key)})')
            else:
                raise IndexError(f'Unable to use indexer "{key}"')
        else:
            where, cols = key, self.table.columns

        if isinstance(where, pd.Series):
            where = where.values

        # If where is boolean mask fetch the entire frame
        if isinstance(where, np.ndarray):
            if where.dtype != bool:
                raise KeyError('Unable to index by non-boolean iterable')
            query = create_query(self.table, columns=cols, where=None, limit=limit)
        else:
            query = create_query(self.table, columns=cols, where=where, limit=limit)

        records = self.table.query(query, no_limit=True)
        data = process_records(records,
                               row_id_index=isinstance(cols, str) and cols != '_id',
                               dtypes=self.table.dtypes.to_dict() if self.table.sanitize else None)

        # Reindex columns so that we have columns even if data is empty
        data = data.reindex(make_iterable(cols), axis=1)

        # If index was boolean mask subset to requested rows
        if isinstance(where, np.ndarray):
            data = data.loc[where].copy()

        # If a single row was requested
        if isinstance(key, int):
            data = data.iloc[0]

        # If a single column was requested
        if isinstance(cols, str):
            data = data[cols]

        return data

    @write_access
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
            raise KeyError('Column must exists, use `add_column()` method to '
                           f'create "{col}" before setting its values')

        if isinstance(where, pd.Series):
            where = where.values

        if is_iterable(where):
            where = np.asarray(where)
            if where.dtype != bool:
                raise KeyError('Unable to index by non-boolean iterable')
            row_ids = self.table.row_ids[where]
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
        values = validate_values(values, dtype=self.table.dtypes[col])

        records = [{'row_id': r,
                    'row': {col: v if not pd.isnull(v) else None}} for r, v in zip(row_ids, values)]

        r = batch_upload(partial(self.table.base.batch_update_rows,
                                 self.table.name),
                         records,
                         batch_size=self.table.max_operations,
                         progress=self.table.progress)

        if 'success' in r:
            logger.info(f'Successfully wrote to "{col}"!')


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

    if not isinstance(columns, type(None)):
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
            q += f' WHERE {where.name}'
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
