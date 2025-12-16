"""
Tests for data processing module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.data_processor import DataProcessor


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'symbol': 'SPY',
        'open': np.random.uniform(400, 450, 100),
        'high': np.random.uniform(410, 460, 100),
        'low': np.random.uniform(390, 440, 100),
        'close': np.random.uniform(400, 450, 100),
        'volume': np.random.randint(1000000, 10000000, 100),
    })


def test_clean_market_data(sample_market_data):
    """Test market data cleaning."""
    processor = DataProcessor()
    cleaned = processor.clean_market_data(sample_market_data)
    
    assert len(cleaned) > 0
    assert 'timestamp' in cleaned.columns
    assert cleaned['high'].ge(cleaned['low']).all()


def test_calculate_returns(sample_market_data):
    """Test returns calculation."""
    processor = DataProcessor()
    returns_data = processor.calculate_returns(sample_market_data, price_col='close')
    
    assert 'returns' in returns_data.columns
    assert 'log_returns' in returns_data.columns


def test_validate_data_quality(sample_market_data):
    """Test data quality validation."""
    processor = DataProcessor()
    quality = processor.validate_data_quality(sample_market_data)
    
    assert 'total_rows' in quality
    assert 'completeness_score' in quality
    assert quality['completeness_score'] >= 0

