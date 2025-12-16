# Troubleshooting BigQuery Connection Issues

## Error: "Unrecognized name: symbol at [12:15]"

This error means the BigQuery query is trying to access a column `symbol` that doesn't exist in the table being queried.

### Possible Causes

1. **Wrong Dataset Name**: The dataset in `.env` doesn't match your actual BigQuery dataset
2. **Wrong Table Name**: The table name doesn't match
3. **Table Structure Different**: The table exists but has different column names

### Solution Steps

#### Step 1: Find Your Actual Dataset and Table

Run the diagnostic script:

```bash
python3.12 find_bigquery_tables.py
```

This will:
- List all datasets in your project
- List all tables in each dataset
- Show the schema of `sample_ticker_data` if found
- Tell you the correct dataset name to use

#### Step 2: Update .env File

Based on the output, update your `.env` file:

```bash
# If the script found the table in dataset "market_data"
BIGQUERY_DATASET=market_data

# Or if it's in a different dataset
BIGQUERY_DATASET=your-actual-dataset-name
```

#### Step 3: Verify Table Schema

The table `sample_ticker_data` should have these columns:
- `date` (DATE)
- `symbol` (STRING)
- `open`, `high`, `low`, `close` (FLOAT)
- `volume` (INTEGER)
- `adjusted_close` (FLOAT)

If your table has different columns, the code will need to be adjusted.

#### Step 4: Test the Connection

```bash
# Test BigQuery connection
python3.12 -c "
from src.data.bigquery_client import BigQueryClient
bq = BigQueryClient()
print('Connection OK')
print('Dataset:', bq.dataset)
print('Available datasets:', bq.list_datasets())
"
```

### Common Issues

**Issue 1: Dataset name mismatch**
- Check your BigQuery console for the actual dataset name
- Update `BIGQUERY_DATASET` in `.env`

**Issue 2: Table in different dataset**
- The table might be in a different dataset than `market_data`
- Update `.env`: `BIGQUERY_DATASET=market_data` (or your actual dataset name)

**Issue 3: Table doesn't exist**
- Verify the table name in BigQuery console
- The table might be named differently (e.g., `stock_data`, `market_data`)

### Quick Fix

If you know the correct dataset name, update `.env`:

```bash
BIGQUERY_DATASET=market_data  # or whatever your dataset is called
```

Then restart the API server.

