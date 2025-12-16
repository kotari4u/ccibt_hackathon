"""
Basic usage examples for Market Activity Prediction Agent.
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.bigquery_client import BigQueryClient
from src.data.data_processor import DataProcessor
from src.data.feature_engineering import FeatureEngineer
from src.models.ensemble_predictor import EnsemblePredictor
from src.models.anomaly_detector import AnomalyDetector


def example_bigquery_connection():
    """Example: Connect to BigQuery and fetch data."""
    print("Example 1: BigQuery Connection")
    print("-" * 50)
    
    try:
        # Initialize client
        bq_client = BigQueryClient()
        
        # Test connection
        if bq_client.test_connection():
            print("✓ BigQuery connection successful")
        else:
            print("✗ BigQuery connection failed")
            return
        
        # Fetch market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        market_data = bq_client.get_market_data(
            symbol="SPY",
            start_date=start_date,
            end_date=end_date,
        )
        
        print(f"✓ Fetched {len(market_data)} rows of market data")
        print(f"  Date range: {market_data['timestamp'].min()} to {market_data['timestamp'].max()}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_data_processing():
    """Example: Process and clean market data."""
    print("\nExample 2: Data Processing")
    print("-" * 50)
    
    try:
        # Create sample data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'SPY',
            'open': np.random.uniform(400, 450, 100),
            'high': np.random.uniform(410, 460, 100),
            'low': np.random.uniform(390, 440, 100),
            'close': np.random.uniform(400, 450, 100),
            'volume': np.random.randint(1000000, 10000000, 100),
        })
        
        # Process data
        processor = DataProcessor()
        cleaned_data = processor.clean_market_data(sample_data)
        returns_data = processor.calculate_returns(cleaned_data)
        
        print(f"✓ Processed {len(cleaned_data)} rows")
        print(f"✓ Calculated returns: {len(returns_data)} rows")
        
        # Validate quality
        quality = processor.validate_data_quality(cleaned_data)
        print(f"✓ Data completeness: {quality['completeness_score']:.1f}%")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_feature_engineering():
    """Example: Create technical indicators."""
    print("\nExample 3: Feature Engineering")
    print("-" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'SPY',
            'open': np.random.uniform(400, 450, 100),
            'high': np.random.uniform(410, 460, 100),
            'low': np.random.uniform(390, 440, 100),
            'close': np.random.uniform(400, 450, 100),
            'volume': np.random.randint(1000000, 10000000, 100),
        })
        
        # Create features
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_all_features(sample_data)
        
        print(f"✓ Created features: {len(features.columns)} columns")
        print(f"  Sample features: {list(features.columns[:10])}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_anomaly_detection():
    """Example: Detect anomalies."""
    print("\nExample 4: Anomaly Detection")
    print("-" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data with anomalies
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = np.random.uniform(400, 450, 100)
        prices[50] = 500  # Anomaly
        prices[75] = 350  # Anomaly
        
        price_series = pd.Series(prices, index=dates)
        
        # Detect anomalies
        detector = AnomalyDetector()
        anomalies = detector.detect_zscore_anomalies(price_series, threshold=2.0)
        
        n_anomalies = anomalies.sum()
        print(f"✓ Detected {n_anomalies} anomalies")
        
        if n_anomalies > 0:
            anomaly_dates = dates[anomalies]
            print(f"  Anomaly dates: {anomaly_dates.tolist()[:5]}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_prediction():
    """Example: Make volatility prediction."""
    print("\nExample 5: Volatility Prediction")
    print("-" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample market data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        returns = np.random.normal(0, 0.01, 252)
        prices = 400 * np.exp(np.cumsum(returns))
        
        market_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'SPY',
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 252),
        })
        
        # Make prediction
        predictor = EnsemblePredictor()
        prediction = predictor.predict_volatility(
            market_data=market_data,
            symbol='SPY',
            horizon_days=5,
        )
        
        print(f"✓ Prediction completed")
        print(f"  Forecast: {prediction['forecast'][:3]}...")
        print(f"  Confidence: {prediction['avg_confidence']:.2f}")
        print(f"  Rationale: {prediction['rationale'][:100]}...")
        
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("Market Activity Prediction Agent - Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_data_processing()
    example_feature_engineering()
    example_anomaly_detection()
    example_prediction()
    
    # BigQuery example (requires credentials)
    # Uncomment if you have BigQuery configured
    # example_bigquery_connection()
    
    print("\n" + "=" * 50)
    print("Examples completed!")

