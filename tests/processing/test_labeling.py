# External libraries
import pytest
import pandas as pd
import numpy as np

# Importing from our library
from ...finml_core.processing.labeling import TripleBarrierLabeling

@pytest.fixture
def organic_data():
    """
    Creates a price series with enough history to warm up volatility calculation.
    """
    # 1. Warm-up period (20 days of random movement to generate volatility)
    np.random.seed(42)
    warmup = [100.0]
    for _ in range(19):
        # Random walk +/- 1%
        warmup.append(warmup[-1] * (1 + np.random.normal(0, 0.01)))
    
    last_price = warmup[-1] # Approx 100-ish
    
    # 2. Bull Scenario (Jump 10%) -> Should hit Top
    # We make a jump big enough to exceed any reasonable volatility barrier
    bull_start = [last_price] * 2 # Stabilize
    bull_jump = [last_price * 1.10] * 5 # Jump 10% and stay there
    
    # 3. Bear Scenario (Drop 10%) -> Should hit Bottom
    bear_start = [last_price] * 2
    bear_drop = [last_price * 0.90] * 5 # Drop 10%
    
    # 4. Stagnation (Flat) -> Should hit Time Limit
    flat = [last_price] * 15 # Perfectly flat = 0 volatility updates? 
    # Actually, EWM carries over history. Volatility will decay but not be zero immediately.
    
    # Concatenate scenarios into separate DataFrames to avoid overlap confusion
    return {
        'warmup': warmup,
        'bull': bull_start + bull_jump,
        'bear': bear_start + bear_drop,
        'flat': flat
    }

def test_bull_scenario(organic_data):
    """Verifies Label 1 (Profit)."""
    # Combine warmup + bull scenario
    prices = organic_data['warmup'] + organic_data['bull']
    n = len(prices)
    
    df = pd.DataFrame({
        'Close': prices,
        'High': [p + 0.1 for p in prices], # High slightly above
        'Low': [p - 0.1 for p in prices]   # Low slightly below
    })
    
    # Use small span so we get volatility quickly
    labeler = TripleBarrierLabeling(
        stop_loss_multiplier=2.0, 
        take_profit_multiplier=2.0, 
        time_limit=5, 
        vol_span=5
    )
    
    # We don't group here
    results = labeler.apply(df)
    
    # The jump happens right after warmup.
    # Index of jump start = len(warmup).
    # The label for the day BEFORE the jump should be 1.
    idx_before_jump = len(organic_data['warmup']) - 1
    
    # We expect a 1 because looking forward 5 days, price jumps 10%.
    assert results['label'].iloc[idx_before_jump] == 1.0
    # Returns should be positive
    assert results['return'].iloc[idx_before_jump] > 0

def test_bear_scenario(organic_data):
    """Verifies Label -1 (Loss)."""
    prices = organic_data['warmup'] + organic_data['bear']
    df = pd.DataFrame({
        'Close': prices,
        'High': [p + 0.1 for p in prices],
        'Low': [p - 0.1 for p in prices]
    })
    
    labeler = TripleBarrierLabeling(time_limit=5, vol_span=5)
    results = labeler.apply(df)
    
    idx_before_drop = len(organic_data['warmup']) - 1
    
    # Expect -1 because price drops 10%
    assert results['label'].iloc[idx_before_drop] == -1.0
    # Returns should be negative
    assert results['return'].iloc[idx_before_drop] < 0

def test_stagnation_scenario(organic_data):
    """Verifies Label 0 (Time Limit) and Return Calculation."""
    # Warmup + Flat line
    prices = organic_data['warmup'] + organic_data['flat']
    df = pd.DataFrame({
        'Close': prices,
        'High': [p + 0.01 for p in prices], # Very tight range
        'Low': [p - 0.01 for p in prices]
    })
    
    labeler = TripleBarrierLabeling(time_limit=5, vol_span=5)
    results = labeler.apply(df)
    
    idx_start_flat = len(organic_data['warmup'])
    
    # In the flat region, volatility decays but barriers are still there.
    # Price moves 0.01. Barriers (from previous volatility) should be wider than 0.01.
    # So it shouldn't touch.
    assert results['label'].iloc[idx_start_flat] == 0.0
    
    # Check that return is calculated (not NaN) even for 0 label
    # This was the bug you found!
    ret = results['return'].iloc[idx_start_flat]
    assert not np.isnan(ret)
    # Return should be very close to 0 (since prices are flat)
    assert ret == pytest.approx(0.0, abs=1e-4)

def test_grouping_integration():
    """Verifies that grouping works with the internal calculation."""
    # Create 2 assets.
    # Asset A: Jumps (Label 1)
    # Asset B: Drops (Label -1)
    
    dates = pd.date_range('2023-01-01', periods=30)
    
    # Asset A
    prices_a = [100.0]*15 + [110.0]*15 # Jump at 15
    # Add noise to start to generate volatility
    prices_a[:15] = [100.0 + np.random.normal(0, 0.5) for _ in range(15)]
    
    df_a = pd.DataFrame({'Date': dates, 'Ticker': 'A', 'Close': prices_a, 'High': prices_a, 'Low': prices_a})
    
    # Asset B
    prices_b = [100.0]*15 + [90.0]*15 # Drop at 15
    prices_b[:15] = [100.0 + np.random.normal(0, 0.5) for _ in range(15)]
    
    df_b = pd.DataFrame({'Date': dates, 'Ticker': 'B', 'Close': prices_b, 'High': prices_b, 'Low': prices_b})
    
    df = pd.concat([df_a, df_b]).reset_index(drop=True)
    
    labeler = TripleBarrierLabeling(time_limit=5, vol_span=5)
    results = labeler.apply(df, group_col='Ticker')
    
    # Check Asset A (Jump) - Look at index 14 (before jump)
    # Since we reset index, we need to find the row.
    # Index 14 is Asset A. Index 44 is Asset B (30+14).
    
    # Check A
    res_a = results.iloc[14]
    assert res_a['label'] == 1.0
    
    # Check B
    res_b = results.iloc[44]
    assert res_b['label'] == -1.0