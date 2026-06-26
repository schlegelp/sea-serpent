# sea-serpent
A dataframe-like wrapper around the [SeaTable](https://seatable.io/en/) API.

## Highlights
- **DataFrame-like interface**: Interact with your tables as if they were local pandas DataFrames. No more wrestling with the API!
- **Automatic data type handling**: No more manual conversions. Just work with your data and let `sea-serpent` handle the rest.
- **Read and write access**: Fetch data, add new columns, update existing values, and even create new tables - all with the same intuitive interface.
- **Easy authentication management**: Store your API tokens securely and switch between multiple SeaTable instances with ease.

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

In newer versions of SeaTable, you can get your API token directly from the website:
Go to Personal Settings -> Account Token and copy the token from there. Then you can store it for later use:

```python
>>> import seaserpent as ss
>>> ss.set_auth_token(token='YOUR_ACCOUNT_TOKEN', server='https://cloud.seatable.io')
Saved SeaTable auth token to ~/.seaserpent/secrets/instance.cloud.seatable.io.json
'YOUR_ACCOUNT_TOKEN'
```

If you work with an older instance of SeaTable, you can get your API token by providing your username and password:

```python
>>> import seaserpent as ss
>>> ss.get_auth_token(username='USER',
...                   password='PASSWORD',
...                   server='https://cloud.seatable.io')
Saved SeaTable auth token to ~/.seaserpent/secrets/instance.cloud.seatable.io.json
{'token': 'YOUR_ACCOUNT_TOKEN'}
```

 Note how the token is tied to a specific server. This enables you to work with multiple SeaTable instances at
 the same time by just switching the server URL and let `sea-serpent` handle the rest. When working with tables,
 you can either provide the server explicitly (via the `server` argument) or set a default server URL via the `SEATABLE_SERVER` environment variable.

For legacy reasons, we also support a `SEATABLE_TOKEN` environment variable for the auth token. The resolution
order is:

1. Explicit `auth_token` via `server` argument
2. Host-specific secret file: `instance.<host>.json`
3. General secret file: `seatable_secret.json`
4. `SEATABLE_TOKEN` environment variable (legacy)

#### Fine-grained (base-level) access tokens

Besides account tokens, SeaTable can issue **base-level "API tokens"** that are restricted to a
single base and can be read-only or read+write (Base -> ... -> "Advanced" -> "API Tokens"). These
work everywhere an account token does — pass one as `auth_token` (or store it via `set_auth_token` /
`SEATABLE_TOKEN`) and `sea-serpent` auto-detects the token type:

```python
>>> import seaserpent as ss
>>> t = ss.Table('MyTable',
...              auth_token='YOUR_BASE_TOKEN',
...              server='https://cloud.seatable.io')
```

A base-level token is tied to exactly one base, so there is no base/workspace look-up — you don't
even need to pass `base`. You still need to specify the `server` (explicitly or via
`SEATABLE_SERVER`). Writes (including creating tables via `Table.new` / `Table.from_frame`) require a
read+write token; a read-only token will be rejected by the server on write attempts.

### Initializing a table

`Table` works as connection to a single SeaTable table. If its name is unique,
you can initialize the connection with just the name:

```python
>>> import seaserpent as ss
>>> # Initialize the table
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

If there are multiple tables with a given name, you need to also specify a `base`!

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

## Additional Notes & Limitations

1. For convenience and ease of access we're using names to identify tables,
   columns and bases. Hence you should avoid duplicate names if at all possible.
2. 64-bit integers/floats are truncated when writing to a table. I suspect this
   happens on the server side when decoding the JSON payload because manually
   entering large numbers through the web interface works perfectly well
   (copy-pasting still fails though). Hence, `sea-serpent` quietly downcasts 64
   bit to 32-bit if possible and failing that converts to strings before uploading.
3. The web interface appears to only show floats up to the 8th decimal. In the
   database the precision must be higher though because I have successfully
   written 1e-128 floats.
4. Infinite values (i.e. `np.inf`) raise an error when trying to write.
5. Cells manually cleared through the UI return empty strings (``''``). By
   default, ``sea-serpent`` will silenelty convert these to ``None``.
