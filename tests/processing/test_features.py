# External Libraries
import pytest
import pandas as pd
import numpy as np

# Importing from our library
from ...finml_core.processing.features import FeatureGenerator

# --- FIXTURES (Test Data) ---

@pytest.fixture
def basic_df():
    """A simple DataFrame with 20 days of data for a single asset."""
    dates = pd.date_range(start='2023-01-01', periods=20)
    return pd.DataFrame({
        'Date': dates,
        'Ticker': ['AAPL'] * 20,
        'Open': 100,
        'High': 105,
        'Low': 95,
        'Close': np.linspace(100, 120, 20), # Price rising linearly
        'Volume': 1000
    })

@pytest.fixture
def multi_asset_df():
    """DataFrame with two assets (AAPL and GOOGL) stacked vertically."""
    dates = pd.date_range(start='2023-01-01', periods=20)
    
    # Asset 1: AAPL (Rising)
    df1 = pd.DataFrame({
        'Date': dates,
        'Ticker': 'AAPL',
        'Close': np.linspace(100, 120, 20),
        'Volume': 1000
    })
    
    # Asset 2: GOOGL (Falling)
    df2 = pd.DataFrame({
        'Date': dates,
        'Ticker': 'GOOGL',
        'Close': np.linspace(200, 180, 20),
        'Volume': 500
    })
    
    return pd.concat([df1, df2]).reset_index(drop=True)

# --- TESTS ---

def test_initialization_defaults():
    """Verifies that the class loads default configuration if nothing is provided."""
    generator = FeatureGenerator(config=None)
    
    # Should have default keys
    assert 'rsi' in generator.config
    assert generator.config['rsi']['period'] == 14
    
    # Should have default column mapping
    assert generator.map['close'] == 'Close'

def test_generate_metrics_structure(basic_df):
    """Verifies that generate creates the expected columns (RSI, Bollinger, etc.)."""
    # Custom config: Only RSI and Bollinger
    config = {
        'rsi': {'period': 14},
        'bollinger': {'window': 20}
    }
    
    generator = FeatureGenerator(config=config)
    df_out = generator.generate(basic_df)
    
    # 1. Check RSI
    assert 'RSI' in df_out.columns
    assert 'RSI_change' in df_out.columns
    
    # 2. Check Bollinger (must have added multiple columns)
    # Since _bb returns a DataFrame and we use pd.concat, names should align.
    # We assume indicators.py returns ['middle_band', 'upper_band', ... 'percent_b', 'bandwidth']
    assert 'upper_band' in df_out.columns
    assert 'percent_b' in df_out.columns

def test_generate_validation_error(basic_df):
    """Verifies that it raises an error if a required column (e.g., Close) is missing."""
    # Create a DataFrame without the 'Close' column
    bad_df = basic_df.drop(columns=['Close'])
    
    config = {'rsi': {'period': 14}}
    generator = FeatureGenerator(config=config)
    
    # Must raise KeyError because RSI requires 'Close'
    with pytest.raises(KeyError):
        generator.generate(bad_df)

def test_generate_grouping(multi_asset_df):
    """Verifies that calculations respect groups (Tickers) and don't leak data."""
    config = {
        'log_returns': {'period': 1}
    }
    generator = FeatureGenerator(config=config)
    
    # Execute with grouping
    df_out = generator.generate(multi_asset_df, group_col='Ticker')
    
    # ISOLATION CHECK:
    # The first value of EACH group must be NaN for log_returns (since period=1).
    # If grouping fails, the first value of the second ticker would use the last price of the previous one.
    
    # Index 0 is AAPL start -> Must be NaN
    assert np.isnan(df_out.iloc[0]['log_ret'])
    
    # Index 20 is GOOGL start -> Must be NaN
    assert np.isnan(df_out.iloc[20]['log_ret'])
    
    # Index 21 should have data (period 1 is available within the group)
    assert not np.isnan(df_out.iloc[21]['log_ret'])

def test_custom_mapping(basic_df):
    """Verifies that column mapping works correctly (using non-standard names)."""
    # Rename DataFrame to Spanish to simulate a different data source
    spanish_df = basic_df.rename(columns={'Close': 'Precio', 'Volume': 'Vol'})
    
    # Define mapping: Internal Name -> External Name
    col_map = {'close': 'Precio', 'volume': 'Vol'}
    config = {'log_returns': {'period': 1}}
    
    generator = FeatureGenerator(config=config, col_map=col_map)
    df_out = generator.generate(spanish_df)
    
    assert 'log_ret' in df_out.columns
    # Verify calculation: log(101.05/100) > 0
    assert df_out['log_ret'].iloc[1] > 0

def test_prepare_X_structure(basic_df):
    """Verifies filtering, lagging, and cleaning logic in prepare_X."""
    # 1. Configure and Generate Full Dataset
    config = {'rsi': {'period': 14}, 'log_returns': {'period': 1}}
    generator = FeatureGenerator(config)
    df_full = generator.generate(basic_df)
    
    # 2. Selecting Features
    # Choosing: RSI (without lags) and Log Returns (with lag 1)
    selection = {
        'RSI': [],        # Empty List -> Keeps RSI base column only
        'log_ret': [1]    # List [1] -> Keeps log_ret base + log_ret_lag_1
    }
    
    df_X = generator.prepare_X(df_full, selection)
    
    # 3. Verifications
    # Columns that MUST be present
    assert 'RSI' in df_X.columns
    assert 'log_ret' in df_X.columns
    assert 'log_ret_lag_1' in df_X.columns
    
    # Columns that MUST NOT be present
    assert 'RSI_lag_1' not in df_X.columns   # We didn't ask for RSI lags
    assert 'RSI_change' not in df_X.columns  # We didn't ask for RSI change
    assert 'Close' not in df_X.columns       # We didn't ask for raw data

    # Verification of Cleaning (dropna)
    # RSI(14) creates 13 NaNs. Lag(1) adds 1 more (potentially overlapping).
    # The final DF must be shorter than the original due to dropna.
    assert len(df_X) < len(basic_df)
    assert not df_X.isna().any().any() # There should be zero NaNs

def test_prepare_X_grouping():
    """Verifies that lags respect group boundaries in prepare_X."""
    # Create manual stacked data
    df = pd.DataFrame({
        'Ticker': ['A', 'A', 'B', 'B'],
        'Val': [1, 2, 3, 4]
    })
    
    gen = FeatureGenerator(config={'log_returns':{}}) # Dummy config
    # Simulate that 'Val' is a metric already calculated by generate()
    
    selection = {'Val': [1]}
    
    # Inject dummy dataframe into prepare_X
    df_X = gen.prepare_X(df, selection, group_col='Ticker')
    
    # Logic Check:
    # Group A: [1, 2] -> Lag 1 of '2' is '1'. Lag 1 of '1' is NaN.
    # Group B: [3, 4] -> Lag 1 of '4' is '3'. Lag 1 of '3' is NaN.
    # dropna() will remove the first row of each group (where lag is NaN).
    
    # Only row 2 (Val 2) and row 4 (Val 4) should remain.
    # Their lags should be 1.0 and 3.0 respectively.
    assert len(df_X) == 2
    assert df_X['Val_lag_1'].iloc[0] == 1.0
    assert df_X['Val_lag_1'].iloc[1] == 3.0