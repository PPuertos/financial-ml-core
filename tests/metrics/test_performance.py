# External Libraries
import pytest
import pandas as pd

# Importing from our library
from ...finml_core.metrics.performance import sharpe_ratio

@pytest.fixture
def winning_returns():
    """Returns that average 1% per day (huge, but good for testing)."""
    return pd.Series([0.01, 0.01, 0.01, 0.01, 0.02])

@pytest.fixture
def losing_returns():
    """Returns that average -1% per day."""
    return pd.Series([-0.01, -0.01, -0.01, -0.01, -0.02])

def test_sharpe_basic(winning_returns):
    """Verifies a positive Sharpe ratio for a profitable series."""
    # Mean approx 0.012 per day. Annualized ~3.024 (302%)
    # Volatility is low.
    # Sharpe should be high and positive.
    sharpe = sharpe_ratio(winning_returns, risk_free_rate=0.0)
    assert sharpe > 0

def test_sharpe_losing(losing_returns):
    """Verifies a negative Sharpe ratio for a losing series."""
    sharpe = sharpe_ratio(losing_returns, risk_free_rate=0.0)
    assert sharpe < 0

def test_sharpe_risk_free_impact(winning_returns):
    """Verifies that increasing the risk-free rate lowers the Sharpe Ratio."""
    # Scenario A: Risk Free = 0%
    sharpe_no_risk = sharpe_ratio(winning_returns, risk_free_rate=0.0)
    
    # Scenario B: Risk Free = 10% (0.10)
    # The hurdle is higher, so the ratio must be lower.
    sharpe_with_risk = sharpe_ratio(winning_returns, risk_free_rate=0.10)
    
    assert sharpe_with_risk < sharpe_no_risk

def test_sharpe_math_check():
    """Verifies exact math on a controlled simple case."""
    # Series: [1%, -1%]. Mean = 0. Std Sample = sqrt( ((0.01-0)^2 + (-0.01-0)^2) / (2-1) ) 
    # Std = sqrt( 0.0001 + 0.0001 ) = sqrt(0.0002) = 0.01414
    returns = pd.Series([0.01, -0.01])
    
    # Mean = 0. Annualized Mean = 0.
    # So Sharpe numerator (0 - 0) is 0.
    # Result should be 0.
    sharpe = sharpe_ratio(returns, risk_free_rate=0.0)
    assert sharpe == pytest.approx(0.0)