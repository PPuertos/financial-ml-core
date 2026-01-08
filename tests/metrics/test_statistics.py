# External Libraries
import pytest
import pandas as pd
import numpy as np

# Importing from our library
from ...finml_core.metrics.statistics import (
    log_returns, simple_returns, volatility, rolling_mean, rolling_std
)

# --- FIXTURES (Test Data) ---
@pytest.fixture
def sample_prices():
    return pd.Series([100.0, 102.0, 101.0, 105.0])

@pytest.fixture
def flat_returns():
    return pd.Series([1.0, 2.0, 3.0])

@pytest.fixture
def constant_returns():
    return pd.Series([0.05, 0.05, 0.05])

# --- TESTS ---
def test_log_returns_basic(sample_prices):
    """Verifies the manual calculation of a log return."""
    result = log_returns(sample_prices)
    expected = np.log(102) - np.log(100)
    assert result.iloc[1] == pytest.approx(expected)

def test_simple_returns_basic(sample_prices):
    """Verifies the manual calculation of simple returns."""
    result = simple_returns(sample_prices)
    assert result.iloc[1] == pytest.approx(0.02)

def test_volatility_annualization(flat_returns):
    """Verifies the annualization scaling factor."""
    # Std Dev for [1,2,3] is 1.0. Annualized is 1.0 * sqrt(252)
    result = volatility(flat_returns, annualize=True, scale=252)
    assert result == pytest.approx(np.sqrt(252))

def test_volatility_constant(constant_returns):
    """Verifies that constant data yields zero volatility."""
    result = volatility(constant_returns, annualize=False)
    assert result == pytest.approx(0.0)

# --- TESTS FOR ROLLING METRICS ---
def test_rolling_mean_basic(sample_prices):
    """Verifies that the rolling mean with window 2 is accurate."""
    # Prices: [100, 102, 101, 105]
    # Window 2 at index 1: mean(100, 102) = 101.0
    # Window 2 at index 2: mean(102, 101) = 101.5
    
    result = rolling_mean(sample_prices, window=2)
    
    assert np.isnan(result.iloc[0]) # First value must be NaN
    assert result.iloc[1] == pytest.approx(101.0)
    assert result.iloc[2] == pytest.approx(101.5)

def test_rolling_std_basic(sample_prices):
    """Verifies that the rolling std dev with window 2 is accurate."""
    # Prices: [100, 102, ...]
    # Window 2 at index 1: std(100, 102) 
    # Mean=101. SumSqDiff = (100-101)^2 + (102-101)^2 = 1 + 1 = 2.
    # Variance (N-1) = 2 / (2-1) = 2.
    # StdDev = sqrt(2) ≈ 1.4142
    
    result = rolling_std(sample_prices, window=2)
    
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(np.sqrt(2))