"""
Event impact modeling for financial events.
Correlates event types with historical market reactions and calculates impact scores.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import structlog

logger = structlog.get_logger(__name__)


class EventImpactModel:
    """
    Models the impact of financial events on market volatility and prices.
    
    Analyzes historical event reactions and predicts future impacts.
    """
    
    def __init__(self):
        """Initialize event impact model."""
        self.impact_model = None
        self.scaler = StandardScaler()
        self.event_taxonomy = {}
        self.logger = logger
    
    def create_event_taxonomy(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create taxonomy of event types and their characteristics.
        
        Args:
            events_df: DataFrame with event data
            
        Returns:
            Dictionary with event taxonomy
        """
        taxonomy = {}
        
        if 'event_type' not in events_df.columns:
            logger.warning("No event_type column found")
            return taxonomy
        
        for event_type in events_df['event_type'].unique():
            type_events = events_df[events_df['event_type'] == event_type]
            
            taxonomy[event_type] = {
                "count": len(type_events),
                "avg_surprise": float(type_events['surprise_factor'].mean())
                if 'surprise_factor' in type_events.columns
                else None,
                "date_range": {
                    "first": str(type_events['event_date'].min()),
                    "last": str(type_events['event_date'].max()),
                },
            }
        
        self.event_taxonomy = taxonomy
        
        logger.info("Event taxonomy created", n_event_types=len(taxonomy))
        
        return taxonomy
    
    def calculate_event_impact(
        self,
        market_data: pd.DataFrame,
        event_date: datetime,
        symbol: str,
        window_before: int = 5,
        window_after: int = 10,
    ) -> Dict[str, Any]:
        """
        Calculate market impact around a specific event.
        
        Args:
            market_data: Market data DataFrame
            event_date: Event date
            symbol: Trading symbol
            window_before: Days before event to analyze
            window_after: Days after event to analyze
            
        Returns:
            Dictionary with impact metrics
        """
        # Filter data for symbol
        symbol_data = market_data[market_data['symbol'] == symbol].copy()
        
        if len(symbol_data) == 0:
            return {"error": "No data found for symbol"}
        
        # Ensure timestamp is datetime
        if 'timestamp' in symbol_data.columns:
            symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'])
            symbol_data = symbol_data.set_index('timestamp').sort_index()
        elif 'date' in symbol_data.columns:
            symbol_data['timestamp'] = pd.to_datetime(symbol_data['date'])
            symbol_data = symbol_data.set_index('timestamp').sort_index()
        
        # Convert event_date to datetime if needed
        if isinstance(event_date, str):
            event_date = pd.to_datetime(event_date)
        elif hasattr(event_date, 'date'):
            event_date = pd.to_datetime(event_date)
        
        # Find event date in data
        event_idx = symbol_data.index.get_indexer([event_date], method='nearest')[0]
        if event_idx == -1:
            return {"error": "Event date not found in data"}
        
        event_timestamp = symbol_data.index[event_idx]
        
        # Get windows
        start_date = event_timestamp - timedelta(days=window_before)
        end_date = event_timestamp + timedelta(days=window_after)
        
        window_data = symbol_data.loc[start_date:end_date].copy()
        
        if len(window_data) == 0:
            return {"error": "Insufficient data in window"}
        
        # Calculate returns if not present
        if 'returns' not in window_data.columns:
            window_data['returns'] = window_data['close'].pct_change()
        
        # Split into before and after
        event_idx_in_window = window_data.index.get_indexer([event_timestamp], method='nearest')[0]
        
        before_data = window_data.iloc[:event_idx_in_window]
        after_data = window_data.iloc[event_idx_in_window + 1:]
        
        # Calculate metrics
        impact_metrics = {
            "event_date": str(event_timestamp),
            "symbol": symbol,
            "before": {
                "avg_return": float(before_data['returns'].mean()) if len(before_data) > 0 else 0,
                "volatility": float(before_data['returns'].std()) if len(before_data) > 0 else 0,
                "avg_volume": float(before_data['volume'].mean()) if 'volume' in before_data.columns and len(before_data) > 0 else 0,
            },
            "after": {
                "avg_return": float(after_data['returns'].mean()) if len(after_data) > 0 else 0,
                "volatility": float(after_data['returns'].std()) if len(after_data) > 0 else 0,
                "avg_volume": float(after_data['volume'].mean()) if 'volume' in after_data.columns and len(after_data) > 0 else 0,
            },
        }
        
        # Calculate changes
        impact_metrics["volatility_change"] = (
            impact_metrics["after"]["volatility"] - impact_metrics["before"]["volatility"]
        )
        impact_metrics["volatility_change_pct"] = (
            impact_metrics["volatility_change"] / impact_metrics["before"]["volatility"] * 100
            if impact_metrics["before"]["volatility"] > 0
            else 0
        )
        
        impact_metrics["return_change"] = (
            impact_metrics["after"]["avg_return"] - impact_metrics["before"]["avg_return"]
        )
        
        return impact_metrics
    
    def analyze_event_type_impact(
        self,
        events_df: pd.DataFrame,
        market_data: pd.DataFrame,
        event_type: str,
        window_after: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze historical impact of a specific event type.
        
        Args:
            events_df: Events DataFrame
            market_data: Market data DataFrame
            event_type: Event type to analyze
            window_after: Days after event to analyze
            
        Returns:
            Dictionary with aggregated impact statistics
        """
        type_events = events_df[events_df['event_type'] == event_type].copy()
        
        if len(type_events) == 0:
            return {"error": f"No events found for type: {event_type}"}
        
        impacts = []
        
        for _, event in type_events.iterrows():
            impact = self.calculate_event_impact(
                market_data,
                pd.to_datetime(event['event_date']),
                event.get('symbol', 'SPY'),
                window_after=window_after,
            )
            
            if "error" not in impact:
                impacts.append(impact)
        
        if len(impacts) == 0:
            return {"error": "No valid impacts calculated"}
        
        # Aggregate statistics
        volatility_changes = [imp["volatility_change_pct"] for imp in impacts]
        return_changes = [imp["return_change"] for imp in impacts]
        
        aggregated = {
            "event_type": event_type,
            "n_events": len(impacts),
            "avg_volatility_change_pct": float(np.mean(volatility_changes)),
            "std_volatility_change_pct": float(np.std(volatility_changes)),
            "avg_return_change": float(np.mean(return_changes)),
            "std_return_change": float(np.std(return_changes)),
            "positive_vol_impact_rate": float(
                sum(1 for vc in volatility_changes if vc > 0) / len(volatility_changes)
            ),
            "impact_distribution": {
                "min": float(np.min(volatility_changes)),
                "q25": float(np.percentile(volatility_changes, 25)),
                "median": float(np.median(volatility_changes)),
                "q75": float(np.percentile(volatility_changes, 75)),
                "max": float(np.max(volatility_changes)),
            },
        }
        
        logger.info(
            "Event type impact analyzed",
            event_type=event_type,
            n_events=len(impacts),
            avg_vol_change=aggregated["avg_volatility_change_pct"],
        )
        
        return aggregated
    
    def calculate_volatility_multiplier(
        self,
        event_type: str,
        surprise_factor: Optional[float] = None,
        base_multiplier: Optional[float] = None,
    ) -> float:
        """
        Calculate volatility multiplier for an event type.
        
        Args:
            event_type: Type of event
            surprise_factor: Surprise factor (if available)
            base_multiplier: Base multiplier from historical analysis
            
        Returns:
            Volatility multiplier
        """
        # Default multipliers by event type (can be learned from data)
        default_multipliers = {
            "earnings": 1.3,
            "fed_announcement": 1.5,
            "cpi_release": 1.4,
            "jobs_report": 1.3,
            "gdp_release": 1.2,
            "default": 1.1,
        }
        
        multiplier = base_multiplier or default_multipliers.get(
            event_type, default_multipliers["default"]
        )
        
        # Adjust based on surprise factor
        if surprise_factor is not None:
            # Higher surprise = higher multiplier
            surprise_adjustment = 1 + (abs(surprise_factor) * 0.2)
            multiplier *= surprise_adjustment
        
        return multiplier
    
    def build_impact_model(
        self,
        events_df: pd.DataFrame,
        market_data: pd.DataFrame,
        features: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Build predictive model for event impacts.
        
        Args:
            events_df: Historical events DataFrame
            market_data: Market data DataFrame
            features: List of feature columns to use
            
        Returns:
            Dictionary with model and metadata
        """
        if features is None:
            features = ["surprise_factor", "event_type_encoded"]
        
        # Prepare training data
        training_data = []
        
        for _, event in events_df.iterrows():
            impact = self.calculate_event_impact(
                market_data,
                pd.to_datetime(event['event_date']),
                event.get('symbol', 'SPY'),
            )
            
            if "error" not in impact:
                row = {
                    "volatility_change_pct": impact["volatility_change_pct"],
                    "surprise_factor": event.get("surprise_factor", 0),
                    "event_type": event.get("event_type", "unknown"),
                }
                training_data.append(row)
        
        if len(training_data) < 10:
            raise ValueError("Insufficient training data")
        
        train_df = pd.DataFrame(training_data)
        
        # Encode event types
        event_type_encoded = pd.get_dummies(train_df["event_type"], prefix="event")
        X = pd.concat([train_df[["surprise_factor"]], event_type_encoded], axis=1)
        y = train_df["volatility_change_pct"]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        self.impact_model = model
        
        # Calculate R-squared
        r_squared = model.score(X_scaled, y)
        
        metadata = {
            "model_type": "LinearRegression",
            "n_samples": len(training_data),
            "r_squared": float(r_squared),
            "features": list(X.columns),
        }
        
        logger.info("Impact model built successfully", **metadata)
        
        return {
            "model": model,
            "scaler": self.scaler,
            "metadata": metadata,
        }
    
    def predict_event_impact(
        self,
        event_type: str,
        surprise_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Predict impact of an upcoming event.
        
        Args:
            event_type: Type of event
            surprise_factor: Expected surprise factor
            
        Returns:
            Dictionary with predicted impact
        """
        if self.impact_model is None:
            # Fallback to multiplier method
            multiplier = self.calculate_volatility_multiplier(
                event_type, surprise_factor
            )
            return {
                "predicted_volatility_multiplier": multiplier,
                "method": "multiplier",
            }
        
        # Use trained model
        # This is simplified - in production, would need proper feature encoding
        multiplier = self.calculate_volatility_multiplier(event_type, surprise_factor)
        
        return {
            "predicted_volatility_multiplier": multiplier,
            "method": "model",
            "confidence": 0.7,  # Would calculate from model uncertainty
        }
    
    def create_event_reaction_matrix(
        self,
        events_df: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create matrix of event types vs market reactions.
        
        Args:
            events_df: Events DataFrame
            market_data: Market data DataFrame
            
        Returns:
            DataFrame with reaction matrix
        """
        event_types = events_df['event_type'].unique()
        reaction_matrix = []
        
        for event_type in event_types:
            impact_stats = self.analyze_event_type_impact(
                events_df, market_data, event_type
            )
            
            if "error" not in impact_stats:
                reaction_matrix.append({
                    "event_type": event_type,
                    "avg_volatility_change_pct": impact_stats["avg_volatility_change_pct"],
                    "std_volatility_change_pct": impact_stats["std_volatility_change_pct"],
                    "avg_return_change": impact_stats["avg_return_change"],
                    "positive_impact_rate": impact_stats["positive_vol_impact_rate"],
                    "n_events": impact_stats["n_events"],
                })
        
        return pd.DataFrame(reaction_matrix)

