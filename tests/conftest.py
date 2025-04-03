"""Configuration fixtures for pytest."""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    index = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    np.random.seed(42)  # For reproducibility
    
    data = {}
    for ticker in tickers:
        # Generate random price movements
        prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, len(index)))
        data[ticker] = prices
    
    df = pd.DataFrame(data, index=index)
    return df

@pytest.fixture
def sample_fundamental_data():
    """Generate sample fundamental data for testing."""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    data = {
        'Ticker': tickers,
        'PE_Ratio': [20.5, 30.2, 25.1, 60.3, 15.8],
        'PB_Ratio': [15.3, 12.1, 5.2, 14.3, 4.8],
        'ROE': [0.35, 0.28, 0.22, 0.18, 0.25],
        'ROA': [0.21, 0.18, 0.15, 0.08, 0.12],
        'ROIC': [0.25, 0.22, 0.18, 0.12, 0.20],
        'Profit_Margin': [0.25, 0.35, 0.28, 0.05, 0.32],
        'Revenue_Growth': [0.15, 0.12, 0.20, 0.30, 0.08],
        'Earnings_Growth': [0.18, 0.14, 0.22, 0.25, 0.10],
        'Debt_to_Equity': [0.3, 0.5, 0.2, 0.8, 0.15],
        'Current_Ratio': [2.5, 1.8, 3.2, 1.5, 2.8],
        'Interest_Coverage': [25.0, 18.0, 30.0, 10.0, 35.0]
    }
    
    return pd.DataFrame(data).set_index('Ticker')

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        'GENERAL': {
            'log_level': 'INFO',
            'data_dir': 'test_data',
            'results_dir': 'test_data/results'
        },
        'SCORING': {
            'fundamental_weight': 0.7,
            'technical_weight': 0.2,
            'quality_weight': 0.1
        },
        'FUNDAMENTAL_WEIGHTS': {
            'PE_Ratio': 0.10,
            'PB_Ratio': 0.05,
            'ROE': 0.15,
            'ROA': 0.10,
            'ROIC': 0.15,
            'Profit_Margin': 0.10,
            'Revenue_Growth': 0.10,
            'Earnings_Growth': 0.10,
            'Debt_to_Equity': 0.05,
            'Current_Ratio': 0.05,
            'Interest_Coverage': 0.05
        },
        'TECHNICAL_WEIGHTS': {
            'trend_ma': 0.15,
            'rsi': 0.15,
            'macd': 0.10,
            'relative_strength': 0.10
        },
        'PORTFOLIO_OPTIMIZATION': {
            'risk_free_rate': 0.02,
            'max_weight_per_asset': 0.25,
            'min_weight_per_asset': 0.01
        }
    }

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, 'results'), exist_ok=True)
        yield temp_dir
