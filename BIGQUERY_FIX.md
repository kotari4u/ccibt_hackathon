# BigQuery Schema Fix

## Issue
The error "Unrecognized name: symbol at [13:15]" occurred because:
1. The actual table is `sample_ticker_data` (not `market_data`)
2. The date column is `date` (not `timestamp`)
3. Missing columns: `bid_ask_spread` and `trade_count`

## Solution Applied

1. **Updated `get_market_data()` method** to:
   - Use `date` column instead of `timestamp`
   - Query `sample_ticker_data` table by default
   - Handle missing columns gracefully
   - Fall back to `timestamp` if `date` doesn't work

2. **Added configuration option**:
   - `BIGQUERY_TABLE_NAME` environment variable (defaults to `sample_ticker_data`)

## Configuration

Add to your `.env` file:

```bash
BIGQUERY_PROJECT_ID=ccibt-hack25ww7-736
BIGQUERY_DATASET=market_data
BIGQUERY_TABLE_NAME=sample_ticker_data
```

## Table Schema

Your actual table `sample_ticker_data` has:
- `date` (DATE) - Date of the market data
- `symbol` (STRING) - Trading symbol
- `open` (FLOAT) - Opening price
- `high` (FLOAT) - Highest price
- `low` (FLOAT) - Lowest price
- `close` (FLOAT) - Closing price
- `volume` (INTEGER) - Volume traded
- `adjusted_close` (FLOAT) - Adjusted closing price

## Testing

The query now:
1. Uses `date` column and aliases it as `timestamp` for compatibility
2. Selects only existing columns
3. Adds missing columns (`bid_ask_spread`, `trade_count`) as None values
4. Falls back to `timestamp` column if `date` fails

Try your prediction endpoint again - it should work now!

