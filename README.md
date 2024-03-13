# sea-serpent
A dataframe-like wrapper around the [SeaTable](https://seatable.io/en/) API.

This library tries to make interacting with SeaTables as easy as if you were
working with a local pandas DataFrame.

## Install

From PyPI:

```bash
pip3 install sea-serpent
```

Bleeding edge from Github:

```bash
pip3 install git+https://github.com/schlegelp/sea-serpent@main
```

## Examples

### Getting your API (auth) token

```python
>>> import seaserpent as ss
>>> ss.get_auth_token(username='USER',
...                   password='PASSWORD',
...                   server='https://cloud.seatable.io')
{'token': 'somelongassstring1234567@£$^@£$^£'}
```

For future use, set your default server and auth token as `SEATABLE_SERVER` and
`SEATABLE_TOKEN` environment variable, respectively.

### Initializing a table

`Table` works as connection to a single SeaTable table. If its name is unique,
you can initialize the connection with just the name:

```python
>>> import seaserpent as ss
>>> # Initialize the table
>>> # (if there are multiple tables with this name you need to provide the base too)
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
>>> # To write only changed values to the table
>>> # (faster & better for logs)
>>> values = table.checked.values
>>> values[0:2] = True  # Change only two values
>>> table.checked.update(values)
```

### Deleting a column

```python
>>> table['checked'].delete()
>>> table.head()
    column1     labels
0         1          A
1         2          B
2         3          C
>>> # Alternatively you can also clear an entire column
>>> table.checked.clear()
>>> table.head()
    column1     labels   checked
0         1          A      None
1         2          B      None
2         3          C      None
```

### Creating a new table

Empty table:

```python
>>> table = ss.Table.new(table_name='MyNewTable', base='MyBase')
```

From pandas DataFrame:

```python
>>> table = ss.Table.from_frame(df, table_name='MyNewTable', base='MyBase')
```

### Linking tables

Create links:

```python
>>> table.link(other_table='OtherTable',    # name of the other table (must be same base)
...            link_on='Column1',           # column in this table to link on
...            link_on_other='ColumnA',     # column in other table to link on
...            link_col='OtherTableLinks')  # name of column to store links in
```

Create column that pulls data from linked table:

```python
>>> table.add_linked_column(col_name='LinkedData',      # name of new column
...                         link_col='OtherTableLinks', # column with link(s) to other table
...                         link_on='some_value',       # which column in other table to link to
...                         formula='lookup')           # how to aggregate data (lookup, mean, max, etc)
```

## Random notes, limitations & oddities

1. For convenience and ease of access we're using names to identify tables,
   columns and bases. Hence you should avoid duplicate names if at all possible.
2. 64 bit integers/floats are truncated when writing to a table. I suspect this
   happens on the server side when decoding the JSON payload because manually
   entering large numbers through the web interface works perfectly well
   (copy-pasting still fails though). Hence, `seaserpent` quietly downcasts 64
   bit to 32 bit if possible and failing that converts to strings before uploading.
3. The web interface appears to only show floats up to the 8th decimal. In the
   database the precision must be higher though because I have successfully
   written 1e-128 floats.
4. Infinite values (i.e. `np.inf`) raise an error when trying to write.
5. Cells manually cleared through the UI return empty strings (``''``). By
   default, ``sea-serpent`` will convert these to ``None`` where possible.
