"""
Data processing and cleaning pipeline.
Handles data normalization, validation, and preparation for ML models.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
import structlog

from src.utils.config import settings

logger = structlog.get_logger(__name__)


class DataProcessor:
    """
    Processes and cleans market data for analysis and modeling.
    
    Handles missing values, outliers, data validation, and normalization.
    """
    
    def __init__(self):
        """Initialize data processor."""
        self.logger = logger
    
    def clean_market_data(
        self, df: pd.DataFrame, remove_outliers: bool = True
    ) -> pd.DataFrame:
        """
        Clean and validate market data.
        
        Args:
            df: Raw market data DataFrame
            remove_outliers: Whether to remove statistical outliers
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        df = df.copy()
        
        # Ensure timestamp/date is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        elif 'date' in df.columns:
            # Handle date column from sample_ticker_data
            df['timestamp'] = pd.to_datetime(df['date'])
            df = df.sort_values('timestamp')
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        if len(df) < initial_rows:
            logger.info("Removed duplicate rows", removed=initial_rows - len(df))
        
        # Validate OHLC data
        invalid_mask = (
            (df['high'] < df['low'])
            | (df['high'] < df['open'])
            | (df['high'] < df['close'])
            | (df['low'] > df['open'])
            | (df['low'] > df['close'])
        )
        
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            logger.warning(
                "Removed invalid OHLC rows",
                count=invalid_count,
                total_rows=len(df),
            )
            df = df[~invalid_mask]
        
        # Handle missing values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    # Forward fill, then backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    logger.info(
                        "Filled missing values",
                        column=col,
                        count=missing_count,
                    )
        
        # Remove outliers using IQR method
        if remove_outliers:
            df = self._remove_outliers(df)
        
        # Ensure volume is non-negative
        if 'volume' in df.columns:
            df['volume'] = df['volume'].clip(lower=0)
        
        logger.info(
            "Data cleaning completed",
            final_rows=len(df),
            columns=list(df.columns),
        )
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove outliers using IQR method.
        
        Args:
            df: DataFrame to clean
            columns: Columns to check for outliers (defaults to OHLC)
            
        Returns:
            DataFrame with outliers removed
        """
        if columns is None:
            columns = ['open', 'high', 'low', 'close']
        
        df = df.copy()
        initial_rows = len(df)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR  # Using 3x IQR for more conservative filtering
            upper_bound = Q3 + 3 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            if outlier_mask.any():
                df = df[~outlier_mask]
        
        removed = initial_rows - len(df)
        if removed > 0:
            logger.info("Removed outliers", count=removed, columns=columns)
        
        return df
    
    def normalize_data(
        self, df: pd.DataFrame, method: str = "min_max", columns: Optional[List[str]] = None
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize data using specified method.
        
        Args:
            df: DataFrame to normalize
            method: Normalization method ('min_max', 'z_score', 'robust')
            columns: Columns to normalize (defaults to numeric columns)
            
        Returns:
            Tuple of (normalized DataFrame, normalization parameters for inverse transform)
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df = df.copy()
        norm_params = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == "min_max":
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    norm_params[col] = {"method": "min_max", "min": min_val, "max": max_val}
            
            elif method == "z_score":
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
                    norm_params[col] = {"method": "z_score", "mean": mean_val, "std": std_val}
            
            elif method == "robust":
                median_val = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                if iqr > 0:
                    df[col] = (df[col] - median_val) / iqr
                    norm_params[col] = {"method": "robust", "median": median_val, "iqr": iqr}
        
        return df, norm_params
    
    def calculate_returns(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """
        Calculate returns and log returns.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            
        Returns:
            DataFrame with added return columns
        """
        df = df.copy()
        
        # Simple returns
        df['returns'] = df[price_col].pct_change()
        
        # Log returns
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Remove first row (NaN)
        df = df.iloc[1:].reset_index(drop=True)
        
        return df
    
    def resample_data(
        self, df: pd.DataFrame, frequency: str = "1H", agg_method: str = "ohlc"
    ) -> pd.DataFrame:
        """
        Resample time series data to different frequency.
        
        Args:
            df: DataFrame with timestamp index or column
            frequency: Pandas frequency string (e.g., '1H', '1D', '1W')
            agg_method: Aggregation method ('ohlc', 'last', 'mean')
            
        Returns:
            Resampled DataFrame
        """
        if 'timestamp' not in df.columns and df.index.name != 'timestamp':
            raise ValueError("DataFrame must have 'timestamp' column or index")
        
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        if agg_method == "ohlc":
            resampled = df.resample(frequency).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            })
        elif agg_method == "last":
            resampled = df.resample(frequency).last()
        elif agg_method == "mean":
            resampled = df.resample(frequency).mean()
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")
        
        resampled = resampled.dropna()
        resampled = resampled.reset_index()
        
        logger.info(
            "Data resampled",
            original_rows=len(df),
            resampled_rows=len(resampled),
            frequency=frequency,
        )
        
        return resampled
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return metrics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            "total_rows": len(df),
            "missing_values": {},
            "duplicates": 0,
            "data_range": {},
            "completeness_score": 0.0,
        }
        
        # Check missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            quality_metrics["missing_values"][col] = {
                "count": int(missing_count),
                "percentage": round(missing_pct, 2),
            }
        
        # Check duplicates
        if 'timestamp' in df.columns and 'symbol' in df.columns:
            quality_metrics["duplicates"] = int(
                df.duplicated(subset=['timestamp', 'symbol']).sum()
            )
        
        # Data range for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            quality_metrics["data_range"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
            }
        
        # Completeness score (percentage of non-null values)
        total_cells = len(df) * len(df.columns)
        null_cells = df.isnull().sum().sum()
        quality_metrics["completeness_score"] = round(
            ((total_cells - null_cells) / total_cells) * 100, 2
        )
        
        return quality_metrics

