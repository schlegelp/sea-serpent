import logging
import numbers
import requests
import warnings

import numpy as np
import pandas as pd

from seatable_api.main import SeaTableAPI
from tqdm.auto import trange

from .utils import (process_records, validate_comparison, is_iterable,
                    make_iterable, map_columntype, find_base, write_access,
                    validate_dtype, is_hashable)

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


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
    progress :      bool
                    Whether to show progress bars.

    """

    def __init__(self, table, base=None, auth_token=None, server=None,
                 read_only=True, max_operations=1000, progress=True):
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

        self.loc = LocIndexer(self)
        # Haven't decided if the below would actually be useful
        #self.iloc = iLocIndexer(self)

        self.read_only = read_only

        # Maximum number of operations (e.g. edits) per batch
        self.max_operations = max_operations
        self.progress = progress

    def __dir__(self):
        """Custom __dir__ to make columns searchable."""
        return list(set(super().__dir__() + list(self.columns)))

    def __len__(self):
        """Length of table."""
        return self.shape[0]

    def __getattr__(self, name):
        if name not in self.columns:
            raise AttributeError(f'Table has no "{name}" column')
        return Column(name=name, table=self)

    def __getitem__(self, key):
        # If single string assume this is a column and return the promise
        if is_hashable(key):
            if key not in self.columns and key not in ('_id', ):
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
        return process_records(records, columns=columns)

    @write_access
    def __setitem__(self, key, values):
        if not is_hashable(key):
            raise KeyError('Key must be hashable (i.e. a single column). Use '
                           '.loc indexer to set values for a specific slice.')

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

        # Fetch the IDs
        row_ids = self.query('SELECT _id', no_limit=True)

        records = [{'row_id': r['_id'],
                    'row': {key: v}} for r, v in zip(row_ids, values)]

        r = batch_update_rows(self, records,
                              batch_size=self.max_operations)

        if 'success' in r:
            logger.info('Write successful!')

    def __repr__(self):
        shape = self.shape
        return f'SeaTable <"{self.name}", {shape[0]} rows, {shape[1]} columns>'

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
    def dtypes(self):
        """Column data types."""
        return pd.Series([c['type'] for c in self.meta['columns']],
                         index=self.columns)

    @property
    def shape(self):
        """Shape of table."""
        n_rows = self.query('SELECT COUNT(*)',
                            no_limit=False)[0].get('COUNT(*)', 'NA')
        return (n_rows, len(self.columns))

    @classmethod
    def new(cls, table_name, base, auth_token=None, server=None):
        """Create a new table.

        Parameters
        ----------
        name_name : str
                    Name of the new table.
        base :      str | int
                    Name or ID of base.

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

        base.add_table(table_name, lang='en')

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
    def add_column(self, col_name, col_type, data=None):
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

        """
        # Make sure meta data is up-to-date
        self.fetch_meta()

        if col_name in self.columns:
            raise ValueError(f'Column "{col_name}" already exists.')

        col_type = map_columntype(col_type)

        resp = self.base.insert_column(table_name=self.name,
                                       column_name=col_name,
                                       column_type=col_type)

        # Make sure meta is updated before next use
        self._stale = True

        if not resp.get('name'):
            raise ValueError(f'Error writing to table: {resp}')
        logger.info(f'Column "{col_name}" added.')

    def fetch_logs(self, page=1, per_page=25):
        """Fetch activity logs for this table."""
        url = f"{self.server}/dtable-server/api/v1/dtables/{self.base.dtable_uuid}/operations/?page={page}&per_page={per_page}"
        r = requests.get(url, headers=self.base.headers)
        r.raise_for_status()

        return pd.DataFrame.from_records(r.json()['operations'])

    def fetch_meta(self):
        """Fetch/update meta data for table and base."""
        self.base_meta = self.base.get_metadata()

        meta = [t for t in self.base_meta['tables'] if t['name'] == self.name]

        if len(meta) == 0:
            raise ValueError(f'No table with name "{self.name}" in base')
        elif len(meta) > 1:
            raise ValueError(f'Multiple tables with name "{self.name}" in base')

        self._meta = meta[0]

        return meta[0]

    def head(self, n=5):
        """Return top N rows as pandas DataFrame."""
        data = self.base.query(f'SELECT * FROM {self.name} LIMIT {n}')
        return process_records(data, columns=self.columns)

    def to_frame(self, row_id_index=True):
        """Download entire table as pandas DataFrame."""
        data = self.base.query(f'SELECT * FROM {self.name} LIMIT {self.shape[0]}')
        return process_records(data,
                               columns=self.columns,
                               row_id_index=row_id_index)

    def query(self, query, no_limit=False):
        """Run SQL query against this table.

        Parameters
        ----------
        query :     str
                    The SQL query. The "FROM {TABLENAME}" will be automatically
                    added to the end of the query if not already present.
        no_limit :  bool
                    By default, the SeaTable API limits queries to 100 rows. If
                    True, we will override this by adding `LIMIT {table.shape[0]}`
                    to the query.

        Returns
        -------
        list
                    Records (list of dicts) contain the results of the query.

        """
        if 'from' not in query.lower():
            query = f'{query} FROM {self.name}'
        if no_limit and 'LIMIT' not in query:
            query += f' LIMIT {self.shape[0]}'
        logger.debug(f'Running SQL query: {query}')
        return self.base.query(query)


class Column:
    """Class representing a table column."""

    def __init__(self, name, table):
        self.name = name
        self.table = table
        if name == '_id':
            self.meta = {'type': int, 'key': None}
        else:
            self.meta = [c for c in table.meta['columns'] if c['name'] == name][0]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Column <column={self.name}, table="{self.table.name}", datatype={self.dtype}>'

    def __eq__(self, other):
        _ = validate_comparison(self, other)
        if isinstance(other, str):
            # Escape the value
            return Filter(f"{self.name} = '{other}'")
        else:
            # This is assuming other is a number
            return Filter(f"{self.name} = {other}")

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

    def to_series(self):
        """Return this column as pandas.Series."""
        rows = self.table.query(f'SELECT {self.name}', no_limit=True)
        return process_records(rows, row_id_index=self.name != '_id').iloc[:, 0]

    def clear(self):
        """Clear this column."""
        if not self.key:
            raise ValueError(f'Unable to clear column {self.name}')

        row_ids = self.to_series().index

        records = [{'row_id': r,
                    'row': {self.name: None}} for r in row_ids]

        r = batch_update_rows(self.table, records, desc='Clearing',
                              batch_size=self.table.max_operations)

        if 'success' in r:
            logger.info('Clear successful!')

    def delete(self):
        """Delete this column."""
        if not self.key:
            raise ValueError(f'Unable to delete column {self.name}')

        self.table._stale = True
        resp = self.table.base.delete_column(self.table.name, self.key)

        if not resp.get('success'):
            raise ValueError(f'Error writing to table: {resp}')
        logger.info(f'Column "{self.name}" deleted.')

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

        if not resp.get('success'):
            raise ValueError(f'Error writing to table: {resp}')
        logger.info('Column renamed.')

    def unique(self):
        """Return unique values in this column."""
        rows = self.table.query(f'SELECT DISTINCT {self.name}', no_limit=True)
        return process_records(rows).iloc[:, 0].values


class Filter:
    """Class representing an SQL WHERE query."""
    def __init__(self, query):
        self.query = query

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'SQL query <"{self.query}">'

    def __neg__(self):
        if self.query.startswith('NOT'):
            return Filter(self.query[4:])
        else:
            return Filter(f'NOT {self.query}')

    def __and__(self, other):
        if isinstance(other, Filter):
            return Filter(f'{self.query} AND {other.query}')
        elif isinstance(other, Column):
            if other.dtype == 'checkbox':
                return Filter(f'{self.query} AND {other.name}')

        raise TypeError(f'Unable to combine Filter and "{type(other)}"')

    def __or__(self, other):
        if isinstance(other, Filter):
            return Filter(f'{self.query} OR {other.query}')
        elif isinstance(other, Column):
            if other.dtype == 'checkbox':
                return Filter(f'{self.query} OR {other.name}')

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

        query = create_query(self.table, columns=cols, where=where, limit=limit)
        records = self.table.query(query, no_limit=True)
        return process_records(records)

    @write_access
    def __setitem__(self, key, values):
        if not isinstance(key, tuple):
            raise KeyError('Must provide [index, column] key when writing to '
                           'table using the .loc Indexer.')
        elif len(key) != 2:
            raise IndexError(f'Wrong number of indexers ({len(key)})')

        where, col = key

        if col not in self.table.columns:
            raise KeyError('Column must exists, use `add_column()` method to '
                           f'create "{col}" before setting its values')

        row_ids = self[where, '_id'].index

        if isinstance(values, (pd.Series, Column)):
            values = values.values

        if not is_iterable(values):
            values = [values] * len(self.table)
        elif len(values) != len(row_ids):
            raise ValueError(f'Length of values ({len(values)}) does not '
                             f'match length of keys ({len(row_ids)})')

        # Validate datatype
        validate_dtype(self.table, col, values)

        records = [{'row_id': r,
                    'row': {col: v}} for r, v in zip(row_ids, values)]

        r = batch_update_rows(self.table, records,
                              batch_size=self.table.max_operations)

        if 'success' in r:
            logger.info('Write successful!')


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
        else:
            raise TypeError(f'Unable to construct WHERE query from "{type(where)}"')

    if isinstance(limit, numbers.Number):
        q += f' LIMIT {limit}'

    return q


def batch_update_rows(table, records, batch_size=1000, desc='Writing',
                      omit_errors=False):
    """Update rows in batches of defined size."""
    no_errors = True
    for i in trange(0, len(records), batch_size,
                    disable=len(records) < batch_size,
                    desc=desc):
        batch = records[i: i + batch_size]

        r = table.base.batch_update_rows(table.name, rows_data=batch)

        if not r.get('success'):
            if not omit_errors:
                raise ValueError(f'Error writing to table: {r}')
            else:
                logger.error(f'Error writing to table: {r}')
                no_errors = False

    return {'success'} if no_errors else {'errors'}
