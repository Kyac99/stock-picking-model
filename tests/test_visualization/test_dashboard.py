"""Tests for visualization dashboard."""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from src.visualization.dashboard import PerformanceDashboard


@pytest.fixture
def sample_performance_data():
    """Generate sample performance data for testing."""
    index = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    
    np.random.seed(42)  # For reproducibility
    
    # Portfolio values over time
    portfolio_value = 1000 * (1 + np.cumsum(np.random.normal(0.001, 0.015, len(index))))
    
    # Benchmark values over time (e.g., S&P 500)
    benchmark_value = 1000 * (1 + np.cumsum(np.random.normal(0.0008, 0.012, len(index))))
    
    data = {
        'Portfolio_Value': portfolio_value,
        'Benchmark_Value': benchmark_value
    }
    
    return pd.DataFrame(data, index=index)


def test_dashboard_initialization(sample_performance_data):
    """Test initialization of the PerformanceDashboard class."""
    dashboard = PerformanceDashboard(
        performance_data=sample_performance_data,
        portfolio_name='Test Portfolio',
        benchmark_name='S&P 500'
    )
    
    assert dashboard is not None
    assert hasattr(dashboard, 'plot_performance')
    assert hasattr(dashboard, 'plot_drawdown')
    assert hasattr(dashboard, 'plot_monthly_returns')
    assert hasattr(dashboard, 'plot_risk_return')
    assert hasattr(dashboard, 'generate_performance_report')


@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.savefig')
def test_plot_performance(mock_savefig, mock_figure, sample_performance_data):
    """Test plotting portfolio performance."""
    # Setup mock
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_fig.add_subplot.return_value = mock_ax
    mock_figure.return_value = mock_fig
    
    # Initialize dashboard
    dashboard = PerformanceDashboard(
        performance_data=sample_performance_data,
        portfolio_name='Test Portfolio',
        benchmark_name='S&P 500'
    )
    
    # Call the method to test
    dashboard.plot_performance(figsize=(10, 6), save_path='test.png')
    
    # Verify plot creation
    mock_figure.assert_called_once()
    assert mock_ax.plot.call_count >= 2  # Should plot at least portfolio and benchmark
    assert mock_ax.legend.call_count >= 1
    assert mock_ax.set_title.call_count >= 1
    
    # Verify save was called if save_path was provided
    mock_savefig.assert_called_once_with('test.png', dpi=300, bbox_inches='tight')


@patch('matplotlib.pyplot.figure')
def test_plot_drawdown(mock_figure, sample_performance_data):
    """Test plotting drawdown chart."""
    # Setup mock
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_fig.add_subplot.return_value = mock_ax
    mock_figure.return_value = mock_fig
    
    # Initialize dashboard
    dashboard = PerformanceDashboard(
        performance_data=sample_performance_data,
        portfolio_name='Test Portfolio',
        benchmark_name='S&P 500'
    )
    
    # Call the method to test
    dashboard.plot_drawdown(figsize=(10, 6))
    
    # Verify plot creation
    mock_figure.assert_called_once()
    assert mock_ax.fill_between.call_count >= 1  # Should create area plot for drawdown
    assert mock_ax.set_title.call_count >= 1
    assert mock_ax.set_ylabel.call_count >= 1


def test_calculate_performance_metrics(sample_performance_data):
    """Test calculation of performance metrics."""
    # Initialize dashboard
    dashboard = PerformanceDashboard(
        performance_data=sample_performance_data,
        portfolio_name='Test Portfolio',
        benchmark_name='S&P 500'
    )
    
    # Call the method to test
    metrics = dashboard.calculate_performance_metrics()
    
    # Verify metrics structure
    assert isinstance(metrics, dict)
    assert 'Portfolio' in metrics
    assert 'Benchmark' in metrics
    
    portfolio_metrics = metrics['Portfolio']
    benchmark_metrics = metrics['Benchmark']
    
    # Verify portfolio metrics
    assert 'Total Return' in portfolio_metrics
    assert 'Annualized Return' in portfolio_metrics
    assert 'Annualized Volatility' in portfolio_metrics
    assert 'Sharpe Ratio' in portfolio_metrics
    assert 'Max Drawdown' in portfolio_metrics
    
    # Verify benchmark metrics
    assert 'Total Return' in benchmark_metrics
    assert 'Annualized Return' in benchmark_metrics
    assert 'Annualized Volatility' in benchmark_metrics
    assert 'Sharpe Ratio' in benchmark_metrics
    assert 'Max Drawdown' in benchmark_metrics


@patch('matplotlib.pyplot.figure')
def test_generate_performance_report(mock_figure, sample_performance_data):
    """Test generation of performance report with multiple plots."""
    # Setup mock
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_fig.add_subplot.return_value = mock_ax
    mock_figure.return_value = mock_fig
    
    # Initialize dashboard
    dashboard = PerformanceDashboard(
        performance_data=sample_performance_data,
        portfolio_name='Test Portfolio',
        benchmark_name='S&P 500'
    )
    
    # Patch individual plot methods
    with patch.object(dashboard, 'plot_performance') as mock_performance:
        with patch.object(dashboard, 'plot_drawdown') as mock_drawdown:
            with patch.object(dashboard, 'plot_monthly_returns') as mock_monthly:
                with patch.object(dashboard, 'plot_risk_return') as mock_risk:
                    # Call the method to test
                    dashboard.generate_performance_report(save_path='test_report.pdf')
                    
                    # Verify all plot methods were called
                    mock_performance.assert_called_once()
                    mock_drawdown.assert_called_once()
                    mock_monthly.assert_called_once()
                    mock_risk.assert_called_once()
