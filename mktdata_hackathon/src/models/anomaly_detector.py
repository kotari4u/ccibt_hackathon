"""
Anomaly detection module for market data.
Implements statistical and ML-based anomaly detection methods.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


class AnomalyDetector:
    """
    Detects anomalies in market data using multiple methods.
    
    Supports Z-score, IQR, clustering-based, and statistical methods.
    """
    
    def __init__(self):
        """Initialize anomaly detector."""
        self.scaler = StandardScaler()
        self.logger = logger
    
    def detect_zscore_anomalies(
        self,
        data: pd.Series,
        threshold: float = 3.0,
        window: Optional[int] = None,
    ) -> pd.Series:
        """
        Detect anomalies using Z-score method.
        
        Args:
            data: Data series to analyze
            threshold: Z-score threshold (default 3.0)
            window: Rolling window for Z-score calculation (None = global)
            
        Returns:
            Boolean series indicating anomalies
        """
        if window is None:
            # Global Z-score
            z_scores = np.abs(stats.zscore(data.dropna()))
            anomalies = pd.Series(False, index=data.index)
            valid_idx = data.dropna().index
            anomalies.loc[valid_idx] = z_scores > threshold
        else:
            # Rolling Z-score
            rolling_mean = data.rolling(window=window).mean()
            rolling_std = data.rolling(window=window).std()
            z_scores = np.abs((data - rolling_mean) / rolling_std)
            anomalies = z_scores > threshold
        
        return anomalies.fillna(False)
    
    def detect_iqr_anomalies(
        self,
        data: pd.Series,
        multiplier: float = 1.5,
        window: Optional[int] = None,
    ) -> pd.Series:
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        Args:
            data: Data series to analyze
            multiplier: IQR multiplier (default 1.5)
            window: Rolling window (None = global)
            
        Returns:
            Boolean series indicating anomalies
        """
        if window is None:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            anomalies = (data < lower_bound) | (data > upper_bound)
        else:
            rolling_q1 = data.rolling(window=window).quantile(0.25)
            rolling_q3 = data.rolling(window=window).quantile(0.75)
            rolling_iqr = rolling_q3 - rolling_q1
            lower_bound = rolling_q1 - multiplier * rolling_iqr
            upper_bound = rolling_q3 + multiplier * rolling_iqr
            anomalies = (data < lower_bound) | (data > upper_bound)
        
        return anomalies.fillna(False)
    
    def detect_clustering_anomalies(
        self,
        df: pd.DataFrame,
        features: List[str],
        method: str = "dbscan",
        **kwargs,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Detect anomalies using clustering methods.
        
        Args:
            df: DataFrame with features
            features: List of feature column names
            method: Clustering method ('dbscan' or 'kmeans')
            **kwargs: Additional parameters for clustering algorithms
            
        Returns:
            Tuple of (anomaly series, metadata dictionary)
        """
        # Prepare data
        feature_data = df[features].dropna()
        if len(feature_data) == 0:
            logger.warning("No valid data for clustering")
            return pd.Series(False, index=df.index), {}
        
        # Scale features
        scaled_data = self.scaler.fit_transform(feature_data)
        
        if method == "dbscan":
            eps = kwargs.get("eps", 0.5)
            min_samples = kwargs.get("min_samples", 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        elif method == "kmeans":
            n_clusters = kwargs.get("n_clusters", 5)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        labels = clusterer.fit_predict(scaled_data)
        
        # Anomalies are points labeled as -1 (DBSCAN) or outliers in smallest clusters (KMeans)
        if method == "dbscan":
            anomalies_mask = labels == -1
        else:
            # For KMeans, identify outliers as points far from cluster centers
            cluster_centers = clusterer.cluster_centers_
            distances = []
            for i, point in enumerate(scaled_data):
                cluster_id = labels[i]
                center = cluster_centers[cluster_id]
                distance = np.linalg.norm(point - center)
                distances.append(distance)
            
            threshold = np.percentile(distances, 95)  # Top 5% as anomalies
            anomalies_mask = np.array(distances) > threshold
        
        # Create anomaly series aligned with original DataFrame
        anomaly_series = pd.Series(False, index=df.index)
        anomaly_series.loc[feature_data.index] = anomalies_mask
        
        metadata = {
            "method": method,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "n_anomalies": int(anomalies_mask.sum()),
            "anomaly_rate": float(anomalies_mask.sum() / len(anomalies_mask)),
        }
        
        logger.info("Clustering anomaly detection completed", **metadata)
        
        return anomaly_series, metadata
    
    def detect_volume_anomalies(
        self,
        volume: pd.Series,
        prices: pd.Series,
        threshold_multiplier: float = 2.0,
    ) -> pd.Series:
        """
        Detect unusual volume patterns.
        
        Args:
            volume: Volume series
            prices: Price series
            threshold_multiplier: Multiplier for volume threshold
            
        Returns:
            Boolean series indicating volume anomalies
        """
        # Calculate volume moving average
        volume_ma = volume.rolling(window=20).mean()
        volume_std = volume.rolling(window=20).std()
        
        # Volume spike detection
        volume_threshold = volume_ma + (threshold_multiplier * volume_std)
        volume_anomalies = volume > volume_threshold
        
        # Price-volume divergence (high volume with low price movement)
        price_change = prices.pct_change().abs()
        volume_ratio = volume / volume_ma
        
        # High volume but small price change
        divergence_anomalies = (volume_ratio > threshold_multiplier) & (
            price_change < price_change.rolling(window=20).quantile(0.25)
        )
        
        anomalies = volume_anomalies | divergence_anomalies
        
        return anomalies.fillna(False)
    
    def detect_price_anomalies(
        self,
        prices: pd.Series,
        method: str = "combined",
        **kwargs,
    ) -> Dict[str, pd.Series]:
        """
        Detect price anomalies using multiple methods.
        
        Args:
            prices: Price series
            method: Detection method ('zscore', 'iqr', 'combined')
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with anomaly detection results
        """
        results = {}
        
        if method in ["zscore", "combined"]:
            results["zscore"] = self.detect_zscore_anomalies(
                prices, threshold=kwargs.get("zscore_threshold", 3.0)
            )
        
        if method in ["iqr", "combined"]:
            results["iqr"] = self.detect_iqr_anomalies(
                prices, multiplier=kwargs.get("iqr_multiplier", 1.5)
            )
        
        if method == "combined":
            # Combine methods (union)
            combined = pd.Series(False, index=prices.index)
            for method_result in results.values():
                combined = combined | method_result
            results["combined"] = combined
        
        return results
    
    def calculate_anomaly_score(
        self,
        df: pd.DataFrame,
        features: List[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Calculate composite anomaly score from multiple features.
        
        Args:
            df: DataFrame with features
            features: List of feature columns
            weights: Optional weights for each feature
            
        Returns:
            Anomaly score series (0-1 scale)
        """
        if weights is None:
            weights = {feat: 1.0 / len(features) for feat in features}
        
        scores = pd.Series(0.0, index=df.index)
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            # Normalize feature to 0-1 scale
            feature_data = df[feature].dropna()
            if len(feature_data) == 0:
                continue
            
            feature_min = feature_data.min()
            feature_max = feature_data.max()
            
            if feature_max != feature_min:
                normalized = (df[feature] - feature_min) / (feature_max - feature_min)
                # Use absolute deviation from median as anomaly indicator
                median = normalized.median()
                anomaly_indicator = np.abs(normalized - median)
                scores += weights.get(feature, 1.0 / len(features)) * anomaly_indicator
        
        # Normalize final score to 0-1
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores.fillna(0.0)
    
    def detect_regime_changes(
        self,
        returns: pd.Series,
        window: int = 60,
        threshold: float = 2.0,
    ) -> pd.Series:
        """
        Detect regime changes (e.g., volatility regime shifts).
        
        Args:
            returns: Returns series
            window: Rolling window size
            threshold: Threshold for regime change detection
            
        Returns:
            Boolean series indicating regime changes
        """
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=window).std()
        
        # Calculate volatility of volatility
        vol_of_vol = rolling_vol.rolling(window=window).std()
        
        # Detect significant changes
        vol_change = rolling_vol.pct_change()
        regime_changes = np.abs(vol_change) > threshold
        
        return regime_changes.fillna(False)

