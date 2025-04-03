"""Tests for stock scoring models."""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.models.scoring import StockScorer


def test_scorer_initialization(mock_config):
    """Test initialization of the StockScorer class."""
    scorer = StockScorer(config=mock_config)
    assert scorer is not None
    assert hasattr(scorer, 'calculate_fundamental_score')
    assert hasattr(scorer, 'calculate_technical_score')
    assert hasattr(scorer, 'calculate_combined_score')


def test_calculate_fundamental_score(sample_fundamental_data, mock_config):
    """Test calculation of fundamental scores."""
    # Initialize scorer
    scorer = StockScorer(config=mock_config)
    
    # Call the method to test
    result = scorer.calculate_fundamental_score(sample_fundamental_data)
    
    # Verify result structure
    assert isinstance(result, pd.Series)
    assert not result.empty
    assert result.index.equals(sample_fundamental_data.index)
    
    # Verify score range
    assert result.min() >= 0
    assert result.max() <= 1
    
    # Verify relative ranking
    # Higher ROE and lower P/E should result in better scores
    assert result['AAPL'] > result['AMZN']  # AAPL has better fundamentals than AMZN in our test data


def test_calculate_technical_score(sample_price_data, mock_config):
    """Test calculation of technical scores."""
    with patch('src.models.scoring.calculate_rsi'):
        with patch('src.models.scoring.calculate_macd'):
            # Initialize scorer
            scorer = StockScorer(config=mock_config)
            
            # Call the method to test
            result = scorer.calculate_technical_score(
                price_data=sample_price_data,
                window=20
            )
            
            # Verify result structure
            assert isinstance(result, pd.Series)
            assert not result.empty
            
            # Verify score range
            assert result.min() >= 0
            assert result.max() <= 1


def test_calculate_combined_score(sample_fundamental_data, sample_price_data, mock_config):
    """Test calculation of combined scores."""
    with patch.object(StockScorer, 'calculate_fundamental_score') as mock_fund_score:
        with patch.object(StockScorer, 'calculate_technical_score') as mock_tech_score:
            # Setup mocks
            fund_scores = pd.Series({
                'AAPL': 0.85,
                'MSFT': 0.82,
                'GOOGL': 0.75,
                'AMZN': 0.68,
                'META': 0.79
            })
            tech_scores = pd.Series({
                'AAPL': 0.75,
                'MSFT': 0.80,
                'GOOGL': 0.72,
                'AMZN': 0.85,
                'META': 0.65
            })
            mock_fund_score.return_value = fund_scores
            mock_tech_score.return_value = tech_scores
            
            # Initialize scorer
            scorer = StockScorer(config=mock_config)
            
            # Call the method to test
            result = scorer.calculate_combined_score(
                fundamental_data=sample_fundamental_data,
                price_data=sample_price_data
            )
            
            # Verify result structure
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert 'Fundamental_Score' in result.columns
            assert 'Technical_Score' in result.columns
            assert 'Combined_Score' in result.columns
            
            # Verify combined score calculation
            # Combined score should be weighted average of fund and tech scores
            expected_combined = fund_scores * 0.7 + tech_scores * 0.2  # weights from mock_config
            pd.testing.assert_series_equal(
                result['Combined_Score'],
                expected_combined,
                check_exact=False,
                rtol=1e-2
            )
