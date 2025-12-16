# Schema Mapping - Code to BigQuery Tables

This document maps the code's expected schema to the actual BigQuery tables in `data_set_config.json`.

## Market Data

**Code expects:** `market_data` table with `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `bid_ask_spread`, `trade_count`

**Actual table:** `sample_ticker_data` with:
- `date` (DATE) → mapped to `timestamp` in code
- `symbol` (STRING)
- `open`, `high`, `low`, `close`, `volume` (FLOAT/INTEGER)
- `adjusted_close` (FLOAT)
- **Missing:** `bid_ask_spread`, `trade_count` (added as None in code)

## Financial Events

**Code expects:** `financial_events` table with `event_id`, `event_type`, `event_date`, `symbol`, `description`, `actual_value`, `expected_value`, `surprise_factor`, `impact_score`

**Actual tables:** Multiple sources combined:
1. **US_Economic_Indicators** - Economic releases
   - `indicator_date` → `event_date`
   - `indicator_name` → `event_type`
   - `value` → `actual_value`
   - No `symbol` (economic indicators are market-wide)

2. **acquisitions_update_2021** - Acquisition events
   - `acquisition_date` → `event_date`
   - `acquiring_company` → `symbol`
   - `deal_value_usd` → `actual_value`

3. **analyst_ratings_processed** - Analyst rating changes
   - `report_date` → `event_date`
   - `symbol` (exists)
   - `rating`, `price_target` → `description`, `actual_value`

## News Sentiment

**Code expects:** `news_sentiment` table with `timestamp`, `symbol`, `headline`, `sentiment_score`, `source`

**Actual table:** `stock_news` with:
- `publish_time` (TIMESTAMP) → `timestamp`
- `symbol` (STRING)
- `title` → `headline`
- `sentiment_score` (FLOAT)
- `source` (STRING)

## Additional Tables Available

- **indexData** - Market index data (S&P 500, NASDAQ, etc.)
- **communications** - Corporate communications/news releases
- **sp500_companies** - S&P 500 company metadata
- **symbols_valid_meta** - Symbol metadata
- **analyst_ratings_processed** - Analyst ratings
- **acquisitions_update_2021** - Acquisition events
- **US_Economic_Indicators** - Economic indicators

## Column Mappings

| Code Column | Actual Column(s) | Table(s) |
|------------|------------------|----------|
| `timestamp` | `date` | `sample_ticker_data` |
| `timestamp` | `publish_time` | `stock_news` |
| `event_date` | `indicator_date`, `acquisition_date`, `report_date` | Multiple |
| `headline` | `title` | `stock_news` |
| `bid_ask_spread` | None (added as None) | N/A |
| `trade_count` | None (added as None) | N/A |

## Query Updates

All queries have been updated to:
1. Use correct table names from `data_set_config.json`
2. Map column names correctly (e.g., `date` → `timestamp`)
3. Handle missing columns gracefully
4. Combine multiple event sources into unified format

