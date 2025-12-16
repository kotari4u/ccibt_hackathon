"""
Feature engineering for market data.
Creates technical indicators, statistical features, and derived metrics.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


class FeatureEngineer:
    """
    Creates technical indicators and features for ML models.
    
    Generates features like RSI, MACD, Bollinger Bands, ATR, and more.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.logger = logger
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            period: RSI period (default 14)
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' series
        """
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        
        return {"macd": macd, "signal": signal, "histogram": histogram}
    
    def calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, num_std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            period: Moving average period
            num_std: Number of standard deviations
            
        Returns:
            Dictionary with 'upper', 'middle', 'lower' bands
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return {"upper": upper, "middle": middle, "lower": lower}
    
    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period
            
        Returns:
            ATR series
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_moving_averages(
        self, prices: pd.Series, periods: List[int] = [5, 10, 20, 50, 200]
    ) -> Dict[str, pd.Series]:
        """
        Calculate multiple moving averages.
        
        Args:
            prices: Price series
            periods: List of periods for MA calculation
            
        Returns:
            Dictionary with MA series for each period
        """
        mas = {}
        for period in periods:
            mas[f"ma_{period}"] = prices.rolling(window=period).mean()
        return mas
    
    def calculate_volatility(
        self,
        returns: pd.Series,
        window: int = 20,
        annualized: bool = True,
    ) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            returns: Returns series
            window: Rolling window size
            annualized: Whether to annualize volatility
            
        Returns:
            Volatility series
        """
        volatility = returns.rolling(window=window).std()
        
        if annualized:
            # Annualize assuming 252 trading days
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    def calculate_volume_indicators(
        self,
        volume: pd.Series,
        prices: pd.Series,
        period: int = 20,
    ) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators.
        
        Args:
            volume: Volume series
            prices: Price series
            period: Period for calculations
            
        Returns:
            Dictionary with volume indicators
        """
        indicators = {}
        
        # Volume moving average
        indicators["volume_ma"] = volume.rolling(window=period).mean()
        
        # Volume ratio (current volume / average volume)
        indicators["volume_ratio"] = volume / indicators["volume_ma"]
        
        # On-Balance Volume (OBV)
        price_change = prices.diff()
        obv = (volume * np.sign(price_change)).fillna(0).cumsum()
        indicators["obv"] = obv
        
        # Volume-weighted average price (VWAP approximation)
        indicators["vwap"] = (prices * volume).rolling(window=period).sum() / volume.rolling(
            window=period
        ).sum()
        
        return indicators
    
    def calculate_momentum_indicators(
        self, prices: pd.Series, periods: List[int] = [10, 20, 30]
    ) -> Dict[str, pd.Series]:
        """
        Calculate momentum indicators.
        
        Args:
            prices: Price series
            periods: Periods for momentum calculation
            
        Returns:
            Dictionary with momentum indicators
        """
        indicators = {}
        
        for period in periods:
            indicators[f"momentum_{period}"] = prices / prices.shift(period) - 1
            indicators[f"roc_{period}"] = (
                (prices - prices.shift(period)) / prices.shift(period) * 100
            )
        
        return indicators
    
    def create_all_features(
        self, df: pd.DataFrame, include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Create all technical indicators for a market data DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            include_volume: Whether to include volume-based features
            
        Returns:
            DataFrame with added feature columns
        """
        df = df.copy()
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        # Ensure data is sorted by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # MACD
        macd_data = self.calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        df['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        df['bb_position'] = (df['close'] - bb_data['lower']) / (
            bb_data['upper'] - bb_data['lower']
        )
        
        # ATR
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
            df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Moving Averages
        mas = self.calculate_moving_averages(df['close'])
        for key, value in mas.items():
            df[key] = value
        
        # Moving average crossovers
        if 'ma_5' in df.columns and 'ma_20' in df.columns:
            df['ma_cross_5_20'] = (df['ma_5'] - df['ma_20']) / df['ma_20']
        
        # Volatility
        df['volatility_20'] = self.calculate_volatility(df['returns'], window=20)
        df['volatility_60'] = self.calculate_volatility(df['returns'], window=60)
        
        # Volume indicators
        if include_volume and 'volume' in df.columns:
            volume_indicators = self.calculate_volume_indicators(
                df['volume'], df['close']
            )
            for key, value in volume_indicators.items():
                df[key] = value
        
        # Momentum indicators
        momentum_indicators = self.calculate_momentum_indicators(df['close'])
        for key, value in momentum_indicators.items():
            df[key] = value
        
        # Price position (within high-low range)
        if all(col in df.columns for col in ['high', 'low']):
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Log features
        logger.info(
            "Feature engineering completed",
            original_columns=len(df.columns) - len([c for c in df.columns if c.startswith('ma_') or c.startswith('momentum_') or c.startswith('roc_')]),
            total_features=len(df.columns),
        )
        
        return df
    
    def create_lag_features(
        self, df: pd.DataFrame, columns: List[str], lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Create lagged features for time series modeling.
        
        Args:
            df: DataFrame
            columns: Column names to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        return df
    
    def create_rolling_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [5, 10, 20],
        functions: List[str] = ["mean", "std", "min", "max"],
    ) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: DataFrame
            columns: Columns to calculate statistics for
            windows: Rolling window sizes
            functions: Statistical functions to apply
            
        Returns:
            DataFrame with rolling statistics
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                for func in functions:
                    if func == "mean":
                        df[f"{col}_rolling_mean_{window}"] = df[col].rolling(
                            window=window
                        ).mean()
                    elif func == "std":
                        df[f"{col}_rolling_std_{window}"] = df[col].rolling(
                            window=window
                        ).std()
                    elif func == "min":
                        df[f"{col}_rolling_min_{window}"] = df[col].rolling(
                            window=window
                        ).min()
                    elif func == "max":
                        df[f"{col}_rolling_max_{window}"] = df[col].rolling(
                            window=window
                        ).max()
        
        return df

