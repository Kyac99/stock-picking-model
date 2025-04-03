"""Tests for data collectors."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.data.collectors import YahooFinanceCollector


def test_collector_initialization():
    """Test the initialization of data collectors."""
    collector = YahooFinanceCollector()
    assert collector is not None
    assert hasattr(collector, 'fetch_stock_data')


@patch('src.data.collectors.yf.download')
def test_fetch_stock_data(mock_download, sample_price_data):
    """Test fetching stock data from Yahoo Finance."""
    # Setup the mock
    mock_download.return_value = sample_price_data
    
    # Initialize collector
    collector = YahooFinanceCollector()
    
    # Call the method to test
    result = collector.fetch_stock_data(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    # Verify the mock was called correctly
    mock_download.assert_called_once()
    
    # Verify the result is as expected
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


@patch('src.data.collectors.yf.Ticker')
def test_fetch_fundamental_data(mock_ticker_class):
    """Test fetching fundamental data for stocks."""
    # Setup the mock
    mock_ticker = MagicMock()
    mock_ticker.info = {
        'trailingPE': 25.5,
        'priceToBook': 12.3,
        'returnOnEquity': 0.35,
        'returnOnAssets': 0.22,
        'profitMargins': 0.28,
        'revenueGrowth': 0.15,
        'earningsGrowth': 0.18,
        'debtToEquity': 0.5,
        'currentRatio': 2.1,
        'totalCash': 100000000000,
        'totalDebt': 50000000000,
        'totalRevenue': 300000000000,
        'grossProfits': 150000000000,
        'ebitda': 80000000000,
        'operatingCashflow': 75000000000,
        'freeCashflow': 60000000000
    }
    mock_ticker_class.return_value = mock_ticker
    
    # Initialize collector
    collector = YahooFinanceCollector()
    
    # Call the method to test
    result = collector.fetch_fundamental_data('AAPL')
    
    # Verify the results
    assert isinstance(result, dict)
    assert 'PE_Ratio' in result
    assert 'PB_Ratio' in result
    assert 'ROE' in result
    assert 'ROA' in result
    assert result['PE_Ratio'] == 25.5
    assert result['PB_Ratio'] == 12.3
    assert result['ROE'] == 0.35
    assert result['ROA'] == 0.22
