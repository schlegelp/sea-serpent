# sea-serpent
A dataframe-like wrapper around the [SeaTable](https://seatable.io/en/) API.

This library tries to make interacting with SeaTables as if you were working
with a local pandas DataFrame.

At this early sstage, `sea-serpent` provides a read-only interface.

## Examples

Getting your API (auth) token:

```python
>>> import seaserpent as ss
>>> ss.get_auth_token(username='USER',
...                   password='PASSWORD',
...                   server='https://cloud.seatable.io')
{'token': 'somelongassstring1234567@£$^@£$^£'}
```

For future use, it's easiest if you were to set your default server and auth
token as `SEATABLE_SERVER` and `SEATABLE_TOKEN` environment variable.

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
>>> # Fetching a column returns a promise
>>> c = table['column1']  # this works too: c = table.column1
>>> c
Column <column="column1", table="LH_bodies", datatype=number>
>>> # To get the values
>>> c.values
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
>>> # Filters are automatically translated into SQL query
>>> table[table.column1 >= 7]
    column1     labels
0         7          H
1         8          I
2         9          J
>>> table[table.labels.isin(['D', 'E']) ]
    column1     labels
0         4          D
1         5          E
>>> # Download the whole table as pandas DataFrame
>>> df = table.to_frame()
```
