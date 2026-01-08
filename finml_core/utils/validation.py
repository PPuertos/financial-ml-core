# External libraries
from typing import Dict
import pandas as pd

def validate_col_map(col_map: Dict[str, str]):
    """Ensures the dictionary keys are part of the supported OHLCV schema."""
    # Validar que col_map sea dict
    if not isinstance(col_map, dict):
        raise ValueError(
            "col_map must be a dictionary (e.g., {'open':'Open', 'volume':'Volume'})"
        )

    # Default mapping
    col_map_keys = ['open', 'high', 'low', 'close', 'volume']
    
    # Merge user map with defaults
    # Not supported col_names
    invalid_col_names = [
        col_name
        for col_name in col_map
        if col_name not in col_map_keys
    ]

    if invalid_col_names:
        raise KeyError(
            f"Invalid col_map keys: {invalid_col_names}. "
            f"Supported `col_names` keys are: {list(col_map_keys)}"
        )
    
def validate_ticker_level(ticker_level, df):
    """Ensures the specified level exists in the MultiIndex."""
    if ticker_level not in df.index.names:
        raise KeyError(f"'ticker_level' not defined in df MultiIndex.")
    
def validate_date_level(date_level, df):
    """Ensures the specified level exists in the MultiIndex."""
    if date_level not in df.index.names:
        raise KeyError(f"'date_level' not defined in df MultiIndex.")
    
def validate_df_columns(df: pd.DataFrame, col_map: Dict[str, str]):
    """
    Validates that the physical columns in the DataFrame match 
    the names promised in the col_map.
    """
    # User's dataframe column names
    required_names = list(col_map.values())
    # Dataframe columns
    actual_columns = df.columns.tolist()

    # Missing columns
    missing = [name for name in required_names if name not in actual_columns]

    if missing:
        raise KeyError(
            f"Data Consistency Error: The following columns were defined in "
            f"the mapping but are missing in the DataFrame: {missing}. "
            f"Available columns: {actual_columns}"
        )