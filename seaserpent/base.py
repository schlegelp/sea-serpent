import numbers
import os
import warnings

import numpy as np
import pandas as pd

from seatable_api import Base, Account

from .utils import process_records, validate_comparison, is_iterable


class Table:
    """A remote data table.

    Parameters
    ----------
    table :         str | int
                    Name or index of the table.
    base :          str | int, optional
                    Name, ID or UUID of the base containing ``table``. If not
                    provided will try to find a base containing ``table``.
    auth_token :    str, optional
                    Your user's auth token (not the base token). Can either
                    provided explicitly or be set as ``SEATABLE_TOKEN``
                    environment variable.
    server :        str, optional
                    Must be provided explicitly or set as ``SEATABLE_SERVER``
                    environment variable.

    """

    def __init__(self, table, base=None, auth_token=None, server=None):
        if isinstance(table, int) and isinstance(base, type(None)):
            raise ValueError('Must provide a `base` when giving `table` index '
                             'instead of name.')

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

        self.server = server
        self.auth_token = auth_token

        # Initialize Account
        account = Account(None, None, server)
        account.token = auth_token

        # Now find the base
        workspaces = account.list_workspaces()['workspace_list']

        if isinstance(base, type(None)):
            bases = []
            for w in workspaces:
                for b in w.get('table_list', []):
                    base = account.get_base(w['id'], b['name'])
                    for t in base.get_metadata().get('tables', []):
                        if t.get('name', None) == table:
                            bases.append(base)

            if len(bases) == 0:
                raise ValueError(f'Did not find any table "{table}"')
            elif len(bases) > 1:
                raise ValueError(f'Found multiple tables matching "{table}"')

            self.base = bases[0]
        else:
            bases = []
            for w in workspaces:
                for b in w.get('table_list', []):
                    if b['id'] == base or b['name'] == base or b['uuid'] == base:
                        bases.append((w['workspace_id'], w['name']))

            if len(bases) == 0:
                raise ValueError(f'Did not find a base "{base}"')
            elif len(bases) > 1:
                raise ValueError(f'Found multiple bases matching "{base}"')

            self.base = account.get_base(*bases[0])

        self.base_meta = self.base.get_metadata()

        if isinstance(table, int):
            meta = self.base_meta['tables'][table]
            self.name = meta['name']
        else:
            self.name = table
            meta = [t for t in self.base_meta['tables'] if t['name'] == table]

            if len(meta) == 0:
                raise ValueError(f'No table with name "{table}" in base')
            elif len(meta) > 1:
                raise ValueError(f'Multiple tables with name "{table}" in base')

            self.meta = meta[0]

        self.iloc = iLocIndexer(self)
        self.loc = LocIndexer(self)

    def __dir__(self):
        """Custom __dir__ to make columns searchable."""
        return list(set(super().__dir__() + list(self.columns)))

    def __getattr__(self, name):
        if name not in self.columns:
            raise AttributeError(f'Table has no "{name}" column')
        return Column(name=name, table=self)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item not in self.columns:
                raise AttributeError(f'Table has no "{item}" column')
            return Column(name=item, table=self)
        elif isinstance(item, Query):
            query = f'SELECT * FROM {self.name} where {item.query}'
            records = self.query(query, no_limit=True)
            return process_records(records, columns=self.columns)
        elif isinstance(item, Column):
            if not item.dtype == 'checkbox':
                raise TypeError('Can only query by columns with dtype '
                                f'"checkbox", got "{item.dtype}"')
            query = f'SELECT * FROM {self.name} where {item.name}'
            records = self.query(query, no_limit=True)
            return process_records(records, columns=self.columns)

        raise AttributeError(f'Unable to slice table by {type(item)}')

    def __repr__(self):
        shape = self.shape
        return f'SeaTable <"{self.name}", {shape[0]} rows, {shape[1]} columns>'

    @property
    def columns(self):
        """Table columns."""
        return np.array([c['name'] for c in self.meta['columns']])

    @property
    def dtypes(self):
        """Column data types."""
        return np.array([c['type'] for c in self.meta['columns']])

    @property
    def shape(self):
        """Shape of table."""
        n_rows = self.query('SELECT COUNT(*)')[0].get('COUNT(*)', 'NA')
        return (n_rows, len(self.columns))

    def head(self, n=5):
        """Return top N rows as pandas DataFrame."""
        data = self.base.query(f'SELECT * FROM {self.name} LIMIT {n}')
        return process_records(data, columns=self.columns)

    def to_frame(self):
        """Download entire table as pandas DataFrame."""
        data = self.base.query(f'SELECT * FROM {self.name} LIMIT {self.shape[0]}')
        return process_records(data, columns=self.columns)

    def query(self, query, no_limit=False):
        """Run SQL query against this table.

        Parameters
        ----------
        query :     str
                    The SQL query. The "FROM {TABLENAME}" will be automatically
                    added to the end of the query if not already present.

        """
        if 'from' not in query.lower():
            query = f'{query} FROM {self.name}'
        if no_limit:
            query += f' LIMIT {self.shape[0]}'
        return self.base.query(query)


class Column:
    """Class representing a table column."""
    def __init__(self, name, table):
        self.name = name
        self.table = table
        self.meta = [c for c in table.meta['columns'] if c['name'] == name][0]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Column <column={self.name}, table="{self.table.name}", datatype={self.dtype}>'

    def __eq__(self, other):
        _ = validate_comparison(self, other)
        if isinstance(other, str):
            # Escape the value
            return Query(f"{self.name} = '{other}'")
        else:
            # This is assuming other is a number
            return Query(f"{self.name} = {other}")

    def __ge__(self, other):
        return Query((self > other).query.replace('>', '>='))

    def __gt__(self, other):
        _ = validate_comparison(self, other)
        if isinstance(other, numbers.Number):
            return Query(f"{self.name} > {other}")
        raise TypeError(f"'>' not supported between column and {type(other)}")

    def __le__(self, other):
        return Query((self > other).query.replace('<', '<='))

    def __lt__(self, other):
        _ = validate_comparison(self, other)
        if isinstance(other, numbers.Number):
            return Query(f"{self.name} < {other}")
        raise TypeError(f"'>' not supported between column and {type(other)}")

    def __and__(self, other):
        if isinstance(other, Column):
            if other.dtype == 'checkbox':
                return Query(f'{self.name} AND {other.name}')

        raise TypeError(f'Unable to combine Column and "{type(other)}"')

    def __or__(self, other):
        if isinstance(other, Column):
            if other.dtype == 'checkbox':
                return Query(f'{self.name} OR {other.name}')

        raise TypeError(f'Unable to combine Column and "{type(other)}"')

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

        return Query(f"{self.name} IN {str(other)}")

    @property
    def dtype(self):
        return self.meta['type']

    @property
    def values(self):
        rows = self.table.query(f'SELECT {self.name}', no_limit=True)
        return process_records(rows).iloc[:, 0].values

    def unique(self):
        """Return unique values in this column."""
        rows = self.table.query(f'SELECT DISTINCT {self.name}', no_limit=True)
        return process_records(rows).iloc[:, 0].values


class Query:
    """Class representing an SQL query."""
    def __init__(self, query):
        self.query = query

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'SQL query <"{self.query}">'

    def __neg__(self):
        if self.query.startswith('NOT'):
            return Query(self.query[4:])
        else:
            return Query(f'NOT {self.query}')

    def __and__(self, other):
        if isinstance(other, Query):
            return Query(f'{self.query} AND {other.query}')
        elif isinstance(other, Column):
            if other.dtype == 'checkbox':
                return Query(f'{self.query} AND {other.name}')

        raise TypeError(f'Unable to combine Query and "{type(other)}"')

    def __or__(self, other):
        if isinstance(other, Query):
            return Query(f'{self.query} OR {other.query}')
        elif isinstance(other, Column):
            if other.dtype == 'checkbox':
                return Query(f'{self.query} OR {other.name}')

        raise TypeError(f'Unable to combine Query and "{type(other)}"')


class LocIndexer:
    def __init__(self, table):
        self.table = table

    def __getitem__(self, item):
        print(item)

class iLocIndexer:
    def __init__(self, table):
        self.table = table

    def __getitem__(self, k):
        if isinstance(k, slice):
            start, stop, step = self.parse_slice(k)

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
