"""
BigQuery client for data ingestion and querying.
Handles connection, authentication, and data extraction from Google BigQuery.
Updated to match actual schema in data_set_config.json
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from google.auth.exceptions import DefaultCredentialsError
import structlog
import json
from pathlib import Path

from src.utils.config import settings

logger = structlog.get_logger(__name__)


# Symbol to column name mapping for wide format table
SYMBOL_TO_COLUMN_MAP = {
    'DJI': 'Dow Jones __DJI_',
    'IXIC': 'Nasdaq __IXIC_',
    'GSPC': 'S&P500 __GSPC_',
    'SPY': 'S&P500 __GSPC_',  # SPY tracks S&P 500
    'NYA': 'NYSE Composite __NYA_',
    'RUT': 'Russell 2000 __RUT_',
    'VIX': 'CBOE Volitility __VIX_',
    'GDAXI': 'DAX Index __GDAXI_',
    'FTSE': 'FTSE 100 __FTSE_',
    'HSI': 'Hang Seng Index __HSI_',
    'CC=F': 'Cocoa _CC=F_',
    'KC=F': 'Coffee _KC=F_',
    'ZC=F': 'Corn _ZC=F_',
    'CT=F': 'Cotton _CT=F_',
    'LE=F': 'Live Cattle _LE=F_',
    'OJ=F': 'Orange Juice _OJ=F_',
    'ZS=F': 'Soybeans _ZS=F_',
    'SB=F': 'Sugar _SB=F_',
    'ZW=F': 'Wheat _ZW=F_',
    'EH=F': 'Ethanol _EH=F_',
    'HO=F': 'Heating Oil _HO=F_',
    'NG=F': 'Natural Gas _NG=F_',
    'BZ=F': 'Crude Oil-Brent _BZ=F_',
    'CL=F': 'Crude Oil-WTI _CL=F_',
    'HG=F': 'Copper _HG=F_',
    'GC=F': 'Gold _GC=F_',
    'PA=F': 'Palladium _PA=F_',
    'PL=F': 'Platinum _PL=F_',
    'SI=F': 'Silver _SI=F_',
    'FVX': 'Treasury Yield 5 Years __FVX_',
    'IRX': 'Treasury Bill 13 Week __IRX_',
    'TNX': 'Treasury Yield 10 Years __TNX_',
    'TYX': 'Treasury Yield 30 Years __TYX_',
}


def find_column_for_symbol(symbol: str, available_columns: List[str]) -> Optional[str]:
    """
    Find the column name for a given symbol in the wide format table.
    
    Args:
        symbol: Trading symbol (e.g., 'DJI', 'GSPC')
        available_columns: List of available column names in the table
        
    Returns:
        Column name if found, None otherwise
    """
    # First try direct mapping
    if symbol in SYMBOL_TO_COLUMN_MAP:
        column_name = SYMBOL_TO_COLUMN_MAP[symbol]
        if column_name in available_columns:
            return column_name
    
    # Try to find by pattern matching
    symbol_upper = symbol.upper()
    for col in available_columns:
        # Check if symbol appears in column name
        if symbol_upper in col.upper() or col.upper().replace(' ', '').replace('_', '').replace('__', '_').endswith(f'_{symbol_upper}_'):
            return col
    
    return None


class BigQueryClient:
    """
    Client for interacting with Google BigQuery.
    
    Handles authentication, query execution, and data retrieval
    for market data, events, and news sentiment based on actual schema.
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset: Optional[str] = None,
        credentials_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize BigQuery client.
        
        Args:
            project_id: GCP project ID (defaults to config)
            dataset: BigQuery dataset name (defaults to config)
            credentials_path: Path to service account JSON (optional)
            config_path: Path to data_set_config.json (optional)
        """
        self.project_id = project_id or settings.bigquery_project_id
        
        # Try to get dataset from BQ_DATASET_ID if available (format: project.dataset.table)
        # Otherwise use BIGQUERY_DATASET
        dataset_from_env = None
        if hasattr(settings, 'bq_dataset_id') and settings.bq_dataset_id:
            # Parse project.dataset.table format
            parts = settings.bq_dataset_id.split('.')
            if len(parts) >= 2:
                dataset_from_env = parts[1]  # Get dataset from middle part
        
        self.dataset = dataset or dataset_from_env or settings.bigquery_dataset
        
        if not self.project_id:
            raise ValueError("BigQuery project_id must be provided")
        
        # Load dataset configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "data_set_config.json"
        
        self.table_config = self._load_table_config(config_path)
        
        try:
            if credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
                self.client = bigquery.Client(
                    project=self.project_id, credentials=credentials
                )
            elif settings.google_application_credentials:
                credentials = service_account.Credentials.from_service_account_file(
                    settings.google_application_credentials
                )
                self.client = bigquery.Client(
                    project=self.project_id, credentials=credentials
                )
            else:
                # Use default credentials
                self.client = bigquery.Client(project=self.project_id)
            
            logger.info(
                "BigQuery client initialized",
                project_id=self.project_id,
                dataset=self.dataset,
            )
        except DefaultCredentialsError as e:
            logger.error("Failed to initialize BigQuery client", error=str(e))
            raise
    
    def _load_table_config(self, config_path: Path) -> Dict[str, Any]:
        """Load table configuration from JSON file."""
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get("dataset_config", {}).get("tables", {})
            else:
                logger.warning("Dataset config file not found", path=str(config_path))
                return {}
        except Exception as e:
            logger.warning("Failed to load dataset config", error=str(e))
            return {}
    
    def execute_query(self, query: str, use_legacy_sql: bool = False) -> pd.DataFrame:
        """
        Execute a BigQuery SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            use_legacy_sql: Whether to use legacy SQL syntax
            
        Returns:
            DataFrame with query results
            
        Raises:
            Exception: If query execution fails
        """
        try:
            logger.debug("Executing BigQuery query", query_length=len(query))
            query_job = self.client.query(query, job_config=bigquery.QueryJobConfig(
                use_legacy_sql=use_legacy_sql
            ))
            results = query_job.result()
            df = results.to_dataframe()
            logger.info(
                "Query executed successfully",
                rows_returned=len(df),
                bytes_processed=query_job.total_bytes_processed,
            )
            return df
        except Exception as e:
            logger.error("Query execution failed", error=str(e), query=query[:200])
            raise
    
    def get_market_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV market data for a symbol from sample_ticker_data table.
        
        Args:
            symbol: Trading symbol (e.g., 'SPY', 'AAPL')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with columns: timestamp (from date), symbol, open, high, low, close, volume, adjusted_close
        """
        table_name = "sample_ticker_data"
        
        # Build possible table paths to try
        possible_paths = []
        
        # First, try to get dataset from data_set_config.json
        if self.table_config and "sample_ticker_data" in self.table_config:
            config_dataset = self.table_config.get("sample_ticker_data", {}).get("dataset_name")
            if config_dataset:
                possible_paths.append(f"{self.project_id}.{config_dataset}.{table_name}")
        
        # Add configured dataset
        possible_paths.append(f"{self.project_id}.{self.dataset}.{table_name}")
        
        # Add alternative paths based on common dataset names
        alternative_datasets = ["market_data"]
        for alt_dataset in alternative_datasets:
            if alt_dataset != self.dataset:
                possible_paths.append(f"{self.project_id}.{alt_dataset}.{table_name}")
        
        # Try to get dataset from BQ_DATASET_ID if available (format: project.dataset.table)
        if hasattr(settings, 'bq_dataset_id') and settings.bq_dataset_id:
            parts = settings.bq_dataset_id.split('.')
            if len(parts) >= 2:
                dataset_from_id = parts[1]
                # If it's project.dataset.table format, use the table from the ID
                if len(parts) >= 3:
                    table_from_id = parts[2]
                    possible_paths.insert(0, f"{self.project_id}.{dataset_from_id}.{table_from_id}")
                possible_paths.insert(0, f"{self.project_id}.{dataset_from_id}.{table_name}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in possible_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        possible_paths = unique_paths
        
        # Try each possible path
        last_error = None
        for table_path in possible_paths:
            is_wide_format = False  # Initialize
            try:
                # First, try to detect the table schema
                dataset_name = table_path.split('.')[1] if '.' in table_path else self.dataset
                schema = self.get_table_schema(table_name, dataset_name)
                if not schema:
                    # Try to get schema from the full path
                    try:
                        table_ref = self.client.dataset(dataset_name).table(table_name)
                        table = self.client.get_table(table_ref)
                        schema = [{"name": field.name, "type": field.field_type} for field in table.schema]
                    except:
                        pass
                
                if schema:
                    available_columns = [f['name'] for f in schema]
                    logger.debug("Table columns detected", columns=available_columns[:20], symbol=symbol)
                    
                    # Check if table has a 'symbol' column (long format)
                    has_symbol_column = 'symbol' in [col.lower() for col in available_columns]
                    
                    # Check if this is a wide format table (has Date and symbol columns like "Dow Jones __DJI_")
                    # Only treat as wide format if it has Date AND doesn't have a symbol column
                    is_wide_format = 'Date' in available_columns and not has_symbol_column and any('__' in col or '_' in col for col in available_columns if col != 'Date')
                    
                    logger.debug("Table format detection", has_symbol_column=has_symbol_column, is_wide_format=is_wide_format, date_column='Date' in available_columns)
                    
                    if is_wide_format:
                        # Wide format: Date column + one column per symbol
                        column_name = find_column_for_symbol(symbol, available_columns)
                        if not column_name:
                            raise ValueError(
                                f"Symbol '{symbol}' not found in table. Available columns: {available_columns[:10]}..."
                            )
                        
                        # Query wide format table
                        query = f"""
                        SELECT 
                            date as timestamp,
                            '{symbol}' as symbol,
                            CAST({column_name} AS FLOAT64) as close,
                            CAST({column_name} AS FLOAT64) as open,
                            CAST({column_name} AS FLOAT64) as high,
                            CAST({column_name} AS FLOAT64) as low,
                            CAST(NULL AS INT64) as volume
                        FROM `{table_path}`
                        WHERE {column_name} IS NOT NULL
                        """
                        
                        if start_date:
                            query += f" AND Date >= '{start_date.date().isoformat()}'"
                        if end_date:
                            query += f" AND Date <= '{end_date.date().isoformat()}'"
                        
                        query += " ORDER BY Date ASC"
                        
                        if limit:
                            query += f" LIMIT {limit}"
                        
                        logger.debug("Querying wide format table", column=column_name, symbol=symbol)
                        df = self.execute_query(query)
                    else:
                        # Long format: traditional OHLCV with symbol column
                        # Find the actual column name (might be 'symbol', 'Symbol', 'SYMBOL', etc.)
                        symbol_col = None
                        for col in available_columns:
                            if col.lower() == 'symbol':
                                symbol_col = col
                                break
                        
                        if not symbol_col:
                            # Try to find date column too
                            date_col = None
                            for col in available_columns:
                                if col.lower() in ['date', 'timestamp']:
                                    date_col = col
                                    break
                            
                            raise ValueError(
                                f"Table has neither 'symbol' column nor wide format. "
                                f"Available columns: {available_columns[:20]}"
                            )
                        
                        # Find date column
                        date_col = None
                        for col in available_columns:
                            if col.lower() in ['date', 'timestamp']:
                                date_col = col
                                break
                        
                        if not date_col:
                            date_col = 'date'  # fallback
                        
                        # Build query with proper column names
                        # Use case-insensitive and whitespace-tolerant matching
                        query = f"""
                        SELECT 
                            date as timestamp,
                            symbol,
                            open,
                            high,
                            low,
                            close,
                            volume
                        FROM `{table_path}`
                        WHERE UPPER(TRIM(symbol)) = UPPER(TRIM('{symbol}'))
                        """
                        
                        logger.debug("Long format query", symbol_col=symbol_col, date_col=date_col, symbol=symbol, query_preview=query[:200])
                        
                        if start_date:
                            query += f" AND {date_col} >= '{start_date.date().isoformat()}'"
                        if end_date:
                            query += f" AND {date_col} <= '{end_date.date().isoformat()}'"
                        
                        query += f" ORDER BY {date_col} ASC"
                        
                        if limit:
                            query += f" LIMIT {limit}"
                        
                        logger.debug("Querying long format table", symbol=symbol)
                        df = self.execute_query(query)
                else:
                    # Fallback: try long format query first
                    query = f"""
                    SELECT 
                        date as timestamp,
                        symbol,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM `{table_path}`
                    WHERE symbol = '{symbol}'
                    """
                    
                    if start_date:
                        query += f" AND date >= '{start_date.date().isoformat()}'"
                    if end_date:
                        query += f" AND date <= '{end_date.date().isoformat()}'"
                    
                    query += " ORDER BY date ASC"
                    
                    if limit:
                        query += f" LIMIT {limit}"
                    
                    logger.info("Trying long format query (fallback)", path=table_path, symbol=symbol)
                    logger.info("Query VAMSI ###### %s", query)
                    df = self.execute_query(query)
                
                # Add missing columns for compatibility with downstream code
                if 'bid_ask_spread' not in df.columns:
                    df['bid_ask_spread'] = None
                if 'trade_count' not in df.columns:
                    df['trade_count'] = None
                
                logger.info("Successfully queried market data %s", query, table_path=table_path, rows=len(df), format="wide" if is_wide_format else "long")
                return df
            except Exception as e:
                last_error = e
                error_msg = str(e)
                logger.debug("Query failed for path", path=table_path, error=error_msg[:100])
                # Continue to next path
                continue
        
        # If all paths failed, try to help user by listing available datasets/tables
        error_details = f"Failed to query market data for symbol '{symbol}'. "
        error_details += f"Tried paths: {possible_paths}. "
        error_details += f"Last error: {str(last_error)}.\n\n"
        
        # Try to list available datasets and tables to help debug
        try:
            available_datasets = self.list_datasets()
            if available_datasets:
                error_details += f"Available datasets: {available_datasets}\n"
                # Try to find the table in any dataset
                for ds in available_datasets[:3]:  # Check first 3 datasets
                    tables = self.list_tables(ds)
                    if table_name in tables:
                        error_details += f"Found table '{table_name}' in dataset '{ds}'\n"
                        # Try to get schema
                        schema = self.get_table_schema(table_name, ds)
                        if schema:
                            columns = [f['name'] for f in schema]
                            error_details += f"Table columns: {columns}\n"
        except Exception as e:
            logger.debug("Could not list datasets/tables for debugging", error=str(e))
        
        raise ValueError(error_details)
    
    def get_index_data(
        self,
        index_symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 1000,
    ) -> pd.DataFrame:
        """
        Retrieve index data from indexData table.
        
        Args:
            index_symbol: Index symbol (e.g., '^GSPC' for S&P 500)
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with index data
        """
        table_name = "indexData"
        
        query = f"""
        SELECT 
            date as timestamp,
            index_symbol as symbol,
            open,
            high,
            low,
            close,
            volume
        FROM `{self.project_id}.{self.dataset}.{table_name}`
        WHERE index_symbol = '{index_symbol}'
        """
        
        if start_date:
            query += f" AND date >= '{start_date.date().isoformat()}'"
        if end_date:
            query += f" AND date <= '{end_date.date().isoformat()}'"
        
        query += " ORDER BY date ASC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query)
    
    def get_financial_events(
        self,
        symbol: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Retrieve financial events from multiple sources:
        - US_Economic_Indicators (economic releases)
        - acquisitions_update_2021 (acquisition events)
        - analyst_ratings_processed (analyst rating changes)
        
        Args:
            symbol: Filter by symbol (optional)
            event_type: Filter by event type (optional)
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with event data
        """
        events = []
        
        # Get economic indicators as events
        try:
            econ_query = f"""
            SELECT 
                CONCAT('econ_', indicator_name) as event_id,
                indicator_name as event_type,
                indicator_date as event_date,
                CAST(NULL AS STRING) as symbol,
                indicator_name as description,
                value as actual_value,
                CAST(NULL AS FLOAT64) as expected_value,
                CAST(NULL AS FLOAT64) as surprise_factor,
                CAST(NULL AS FLOAT64) as impact_score
            FROM `{self.project_id}.{self.dataset}.US_Economic_Indicators`
            WHERE 1=1
            """
            
            if start_date:
                econ_query += f" AND indicator_date >= '{start_date.date().isoformat()}'"
            if end_date:
                econ_query += f" AND indicator_date <= '{end_date.date().isoformat()}'"
            if event_type:
                econ_query += f" AND indicator_name = '{event_type}'"
            
            econ_df = self.execute_query(econ_query)
            if len(econ_df) > 0:
                events.append(econ_df)
        except Exception as e:
            logger.warning("Failed to fetch economic indicators", error=str(e))
        
        # Get acquisition events
        try:
            acq_query = f"""
            SELECT 
                CONCAT('acq_', acquiring_company, '_', acquired_company) as event_id,
                'acquisition' as event_type,
                acquisition_date as event_date,
                acquiring_company as symbol,
                CONCAT('Acquisition: ', acquiring_company, ' acquired ', acquired_company) as description,
                CAST(deal_value_usd AS FLOAT64) as actual_value,
                CAST(NULL AS FLOAT64) as expected_value,
                CAST(NULL AS FLOAT64) as surprise_factor,
                CAST(NULL AS FLOAT64) as impact_score
            FROM `{self.project_id}.{self.dataset}.acquisitions_update_2021`
            WHERE 1=1
            """
            
            if start_date:
                acq_query += f" AND acquisition_date >= '{start_date.date().isoformat()}'"
            if end_date:
                acq_query += f" AND acquisition_date <= '{end_date.date().isoformat()}'"
            if symbol:
                acq_query += f" AND (acquiring_company = '{symbol}' OR acquired_company = '{symbol}')"
            
            acq_df = self.execute_query(acq_query)
            if len(acq_df) > 0:
                events.append(acq_df)
        except Exception as e:
            logger.warning("Failed to fetch acquisition events", error=str(e))
        
        # Get analyst rating changes as events
        try:
            rating_query = f"""
            SELECT 
                CONCAT('rating_', symbol, '_', CAST(report_date AS STRING)) as event_id,
                'analyst_rating' as event_type,
                report_date as event_date,
                symbol,
                CONCAT('Analyst rating: ', rating, ' by ', firm) as description,
                CAST(price_target AS FLOAT64) as actual_value,
                CAST(NULL AS FLOAT64) as expected_value,
                CAST(NULL AS FLOAT64) as surprise_factor,
                CAST(NULL AS FLOAT64) as impact_score
            FROM `{self.project_id}.{self.dataset}.analyst_ratings_processed`
            WHERE 1=1
            """
            
            if symbol:
                rating_query += f" AND symbol = '{symbol}'"
            if start_date:
                rating_query += f" AND report_date >= '{start_date.date().isoformat()}'"
            if end_date:
                rating_query += f" AND report_date <= '{end_date.date().isoformat()}'"
            
            rating_df = self.execute_query(rating_query)
            if len(rating_df) > 0:
                events.append(rating_df)
        except Exception as e:
            logger.warning("Failed to fetch analyst ratings", error=str(e))
        
        # Combine all events
        if events:
            combined_df = pd.concat(events, ignore_index=True)
            combined_df = combined_df.sort_values('event_date', ascending=False)
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_upcoming_events(
        self, days_ahead: int = 30, min_impact_score: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get upcoming financial events within specified days.
        Uses economic indicators and other event sources.
        
        Args:
            days_ahead: Number of days to look ahead
            min_impact_score: Minimum impact score filter (optional, not used for now)
            
        Returns:
            DataFrame of upcoming events
        """
        end_date = datetime.now() + timedelta(days=days_ahead)
        start_date = datetime.now()
        
        # Get upcoming economic indicators
        events = []
        
        try:
            query = f"""
            SELECT 
                CONCAT('econ_', indicator_name) as event_id,
                indicator_name as event_type,
                indicator_date as event_date,
                CAST(NULL AS STRING) as symbol,
                indicator_name as description,
                CAST(NULL AS FLOAT64) as expected_value,
                CAST(0.5 AS FLOAT64) as impact_score
            FROM `{self.project_id}.{self.dataset}.US_Economic_Indicators`
            WHERE indicator_date >= CURRENT_DATE()
              AND indicator_date <= '{end_date.date().isoformat()}'
            ORDER BY indicator_date ASC
            """
            
            df = self.execute_query(query)
            if len(df) > 0:
                events.append(df)
        except Exception as e:
            logger.warning("Failed to fetch upcoming economic indicators", error=str(e))
        
        if events:
            combined_df = pd.concat(events, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_news_sentiment(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Retrieve news sentiment data from stock_news table.
        
        Args:
            symbol: Filter by symbol (optional)
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum rows to return
            
        Returns:
            DataFrame with columns: timestamp, symbol, headline, sentiment_score, source
        """
        table_name = "stock_news"
        
        query = f"""
        SELECT 
            publish_time as timestamp,
            symbol,
            title as headline,
            sentiment_score,
            source
        FROM `{self.project_id}.{self.dataset}.{table_name}`
        WHERE 1=1
        """
        
        if symbol:
            query += f" AND symbol = '{symbol}'"
        if start_date:
            query += f" AND publish_time >= '{start_date.isoformat()}'"
        if end_date:
            query += f" AND publish_time <= '{end_date.isoformat()}'"
        
        query += " ORDER BY publish_time DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query)
    
    def get_communications(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Retrieve corporate communications from communications table.
        
        Args:
            symbol: Filter by company symbol (optional)
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum rows to return
            
        Returns:
            DataFrame with communications data
        """
        table_name = "communications"
        
        query = f"""
        SELECT 
            communication_id,
            publish_date as timestamp,
            company_symbol as symbol,
            headline,
            body_text as description
        FROM `{self.project_id}.{self.dataset}.{table_name}`
        WHERE 1=1
        """
        
        if symbol:
            query += f" AND company_symbol = '{symbol}'"
        if start_date:
            query += f" AND publish_date >= '{start_date.isoformat()}'"
        if end_date:
            query += f" AND publish_date <= '{end_date.isoformat()}'"
        
        query += " ORDER BY publish_date DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query)
    
    def get_symbol_metadata(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Get symbol metadata from symbols_valid_meta or sp500_companies.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            DataFrame with symbol metadata
        """
        queries = []
        
        # Try symbols_valid_meta first
        try:
            query = f"""
            SELECT 
                symbol,
                company_name,
                exchange,
                asset_type,
                currency
            FROM `{self.project_id}.{self.dataset}.symbols_valid_meta`
            """
            if symbol:
                query += f" WHERE symbol = '{symbol}'"
            
            df = self.execute_query(query)
            if len(df) > 0:
                return df
        except Exception as e:
            logger.debug("symbols_valid_meta not available, trying sp500_companies", error=str(e))
        
        # Fall back to sp500_companies
        try:
            query = f"""
            SELECT 
                symbol,
                company_name,
                CAST(NULL AS STRING) as exchange,
                sector as asset_type,
                CAST(NULL AS STRING) as currency
            FROM `{self.project_id}.{self.dataset}.sp500_companies`
            """
            if symbol:
                query += f" WHERE symbol = '{symbol}'"
            
            return self.execute_query(query)
        except Exception as e:
            logger.warning("Failed to fetch symbol metadata", error=str(e))
            return pd.DataFrame()
    
    def save_predictions(
        self,
        predictions: List[Dict[str, Any]],
        table_name: str = "predictions",
    ) -> None:
        """
        Save prediction results to BigQuery.
        
        Args:
            predictions: List of prediction dictionaries
            table_name: Target table name
        """
        if not predictions:
            logger.warning("No predictions to save")
            return
        
        df = pd.DataFrame(predictions)
        table_id = f"{self.project_id}.{self.dataset}.{table_name}"
        
        try:
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                autodetect=True,
            )
            
            job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
            job.result()  # Wait for job to complete
            
            logger.info(
                "Predictions saved to BigQuery",
                table=table_id,
                rows=len(predictions),
            )
        except Exception as e:
            logger.error("Failed to save predictions", error=str(e), table=table_id)
            raise
    
    def list_datasets(self) -> List[str]:
        """List all datasets in the project."""
        try:
            datasets = list(self.client.list_datasets())
            dataset_names = [dataset.dataset_id for dataset in datasets]
            logger.info("Available datasets", datasets=dataset_names)
            return dataset_names
        except Exception as e:
            logger.error("Failed to list datasets", error=str(e))
            return []
    
    def list_tables(self, dataset_name: Optional[str] = None) -> List[str]:
        """List all tables in a dataset."""
        dataset = dataset_name or self.dataset
        try:
            dataset_ref = self.client.dataset(dataset)
            tables = list(self.client.list_tables(dataset_ref))
            table_names = [table.table_id for table in tables]
            logger.info("Available tables", dataset=dataset, tables=table_names)
            return table_names
        except Exception as e:
            logger.error("Failed to list tables", dataset=dataset, error=str(e))
            return []
    
    def get_table_schema(self, table_name: str, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get schema for a specific table."""
        dataset = dataset_name or self.dataset
        try:
            table_ref = self.client.dataset(dataset).table(table_name)
            table = self.client.get_table(table_ref)
            schema = [{"name": field.name, "type": field.field_type, "mode": field.mode} 
                     for field in table.schema]
            logger.info("Table schema", dataset=dataset, table=table_name, fields=len(schema))
            return schema
        except Exception as e:
            logger.error("Failed to get table schema", dataset=dataset, table=table_name, error=str(e))
            return []
    
    def test_connection(self) -> bool:
        """
        Test BigQuery connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            query = "SELECT 1 as test"
            self.execute_query(query)
            logger.info("BigQuery connection test successful")
            return True
        except Exception as e:
            logger.error("BigQuery connection test failed", error=str(e))
            return False
