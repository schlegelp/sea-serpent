# sea-serpent
A dataframe-like wrapper around the [SeaTable](https://seatable.io/en/) API.

This library tries to make interacting with SeaTables as if you were working
with a local pandas DataFrame.

Fair warning: `sea-serpent` is at an early stage and the interface might still
change substantially.

## Examples

### Getting your API (auth) token

```python
>>> import seaserpent as ss
>>> ss.get_auth_token(username='USER',
...                   password='PASSWORD',
...                   server='https://cloud.seatable.io')
{'token': 'somelongassstring1234567@£$^@£$^£'}
```

For future use, it's easiest if you were to set your default server and auth
token as `SEATABLE_SERVER` and `SEATABLE_TOKEN` environment variable.

### Initializing a table

`Table` works as connection to a single SeaTable table. If its name is unique,
you can initialize the connection with just the name:

```python
>>> import seaserpent as ss
>>> # Initialize the table
>>> # (if there are multiple tables with this name you need to provide more details)
>>> table = ss.Table(table='MyTable')
>>> table
SeaTable <"MyTable", 10 rows, 2 columns>
>>> # Inspect the first couple rows
>>> table.head()
    column1     labels
0         1          A
1         2          B
2         3          C
```

### Fetching data

The `Table` itself doesn't download any of the data. Reading the data works
via an interface similar to `pandas.DataFrames`:

```python
>>> # Fetching a column returns a promise
>>> c = table['column1']  # this works too: c = table.column1
>>> c
Column <column="column1", table="LH_bodies", datatype=number>
>>> # To get the values
>>> c.values
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
>>> # Filters are automatically translated into SQL query
>>> table.loc[table.column1 >= 7]
    column1     labels
0         7          H
1         8          I
2         9          J
>>> table.loc[table.labels.isin(['D', 'E']) ]
    column1     labels
0         4          D
1         5          E
>>> # Download the whole table as pandas DataFrame
>>> df = table.to_frame()
```

### Adding a column

```python
>>> # First we need to re-initialize the table with write access
>>> table = ss.Table(table='MyTable', read_only=False)
>>> table.add_column(col_name='checked', col_type=bool)
>>> # The column will be empty
>>> table.head()
    column1     labels   checked
0         1          A      None
1         2          B      None
2         3          C      None
```

### Pushing data to table

```python
>>> # Overwrite the whole column
>>> table['checked'] = False
>>> table.head()
    column1     labels   checked
0         1          A     False
1         2          B     False
2         3          C     False
>>> # Alternatively pass a list of values
>>> table['checked'] = [False, True, False]
>>> table.head()
    column1     labels   checked
0         1          A     False
1         2          B      True
2         3          C     False
>>> # Write to a subset of the column
>>> table.loc[:2, 'checked'] = False
>>> table.loc[table.labels == 'C', 'checked'] = True
>>> table.head()
    column1     labels   checked
0         1          A     False
1         2          B     False
2         3          C      True
```

### Deleting a column

```python
>>> table['checked'].delete()
>>> table.head()
    column1     labels
0         1          A
1         2          B
2         3          C
```

### Creating a new table

```python
>>> table = ss.Table.new(base='MyBase', table_name='MyNewTable')
```

## TODOs

- export pandas DataFrame to SeaTable
- create new table from DataFrame
