# External Libraries
import pandas as pd
from typing import Dict, List, Any, Optional

# Importing our Internal Tools
from ..metrics.statistics import log_returns
from ..metrics.indicators import rsi, bollinger_bands, macd, relative_volume
from ..utils.validation import validate_col_map, validate_ticker_level

class FeatureGenerator:
    """
        Orchestrates feature engineering pipeline from raw data.
        
        This class handles:
            1. Column Mapping (Standardizing input names).
            2. Feature Calculation (Calling metrics module).
            3. Lag Generation (Per-feature customization).
            4. Multi-asset grouping (Stacking support).

        !!! abstract "Supported Metrics & Configuration Keys:"
            - `log_ret`:
                - period (int): Return period (default: 1).
            - `rsi`:
                - period (int): Lookback period. (Default: 14).
            - `bollinger`:
                - window (int): Moving average window (default: 20).
                - std (float): Standard deviations (default: 2.0).
            - `macd`:
                - fast (int): Fast EMA (default: 12).
                - slow (int): Slow EMA (default: 26).
                - signal (int): Signal EMA (default: 9).
            - `rvol`:
                - window (int): SMA window (default: 20).
        """

    def __init__(
        self,
        col_map: Dict[str, str],
        ticker_level: str,
        config: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Args:
            col_map (Dict): Maps standard names to dataframe columns.
                
                Example:
                ```python
                    {
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    }
                ```
                which is yfinance standard.
                
            ticker_level (str): Name of the MultiIndex level containing
                the asset identifiers (e.g., 'Ticker').

            config (Dict, optional): Configuration dictionary.
                
                Defaults to:
                ```python

                    {
                        'log_ret': {'period': 1},
                        'rsi': {'period': 14},
                        'bollinger': {'window': 20, 'std': 2.0},
                        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                        'rvol': {'window': 20}
                    }
                ```
                which is the market standard configuration.

        Note:
            Your DataFrame must include 'Close' and 'Volume' columns
            to compute all supported metrics.
        
        ---
        """
        self.ticker_level = ticker_level

        default_config = {
            "log_ret": dict(period=1),
            "rsi": dict(period=14),
            "bollinger": dict(window=20, std=2.0),
            "macd": dict(fast=12, slow=26, signal=9),
            "rvol": dict(window=20)
        }

        if config:
            # config parameter content validation
            self._validate_config(default_config, config)

            # Update default config with user config
            new_config = {}
            for metric, params in config.items():
                # {**default_dict, **user_dict}:
                # default updated with user-provided values
                new_config[metric] = {**default_config[metric], **params}

            default_config = new_config

        # Update default config with user requests
        self.config = default_config

        # Map parameter content validation
        validate_col_map(col_map)
        
        self.map = col_map

    def compute_indicators(
            self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        **Phase 1: Feature Calculation Engine**

        This method orchestrates the vectorization of technical indicators across 
        a MultiIndex DataFrame. It ensures that calculations are performed 
        independently for each ticker to prevent look-ahead bias and cross-sectional 
        data leakage.

        The process follows a strict 3-step pipeline:
            1. **Validation**: Verifies that the MultiIndex structure and required 
            columns (mapped via `col_map`) exist before computation.
            2. **Grouped Transformation**: Uses `groupby().transform()` or `apply()` 
            to isolate time-series calculations by asset.
            3. **Feature Expansion**: Dynamically appends new analytical columns 
            to the dataset while preserving the original index.

        Args:
            df (pd.DataFrame): Input DataFrame with a MultiIndex (Date, Ticker) 
                               and raw OHLCV columns.

        Returns:
            (pd.DataFrame):
                
                A copy of the input DataFrame enriched with:
            
                * **Momentum**: Log Returns, RSI, RSI Change.
                * **Volatility**: Bollinger Bands (Middle, Upper, Lower, %B, Bandwidth).
                * **Trend**: MACD (Line, Signal, Histogram, Relative Histogram).
                * **Volume**: Relative Volume (RVol).

        Raises:
            KeyError: If a configured indicator lacks its required raw column 
                      mapping (e.g., trying to calculate RSI without a 'close' column).
            ValueError: If the `ticker_level` is not correctly identified in the Index.

        Note:
            This method is "non-destructive" regarding rows; it calculates metrics 
            for all available data. Handling of `NaN` values generated by rolling 
            windows is deferred to Phase 2 (Filtering).
        
        ---
        """
        ticker_level = self.ticker_level

        # Validate ticker level is in MultiIndex Dataset
        validate_ticker_level(self.ticker_level, df)

        df_out = df.copy()
        
        # Retrieve column names from map
        c_close = self.map.get('close')
        c_vol = self.map.get('volume')

        # --- VALIDATION ---
        required = []

        # Supported close-price calculated metrics
        close_metrics = ['log_ret', 'rsi', 'bollinger', 'macd']

        # User selected close-price calculated metrics
        user_close_metrics = [m for m in close_metrics if m in self.config]

        # Validate that user provided mappings for all columns
        # needed to calculate the specified metrics
        if user_close_metrics:
            required.append(c_close)

            if not c_close:
                msg = (
                    f"Column 'close' not specified. "
                    f"Required to calculate: {user_close_metrics}."
                )

                raise KeyError(msg)

        if 'rvol' in self.config:
            required.append(c_vol)

            if not c_vol:
                msg = (
                    f"Column 'volume' not specified. "
                    "Required to calculate: 'rvol' metric."
                )

                raise KeyError(msg)
        
        # Validate that all mapped columns exist in the dataset
        for col in required:
            if col not in df_out.columns:
                msg = (
                    f"Column '{col}' not found in DataFrame. "
                    "Check col_map."
                )

                raise KeyError(msg)

        # --- 1. LOG RETURNS ---
        if 'log_ret' in self.config:
            cfg = self.config['log_ret']

            period = cfg['period']
            
            # Function wrapper to handle parameters
            def _log_ret(x): return log_returns(x, period=period)

            # Calculate log returns
            df_out['log_ret'] = (
                df_out.groupby(ticker_level)[c_close]
                .transform(_log_ret)
            )

        # --- 2. RSI ---
        if 'rsi' in self.config:
            cfg = self.config['rsi']

            period = cfg['period']

            # Function wrapper to handle parameters
            def _rsi(x): return rsi(x, period=period)

            # Calculating RSI Metrics
            # RSI
            df_out['rsi'] = (
                df_out.groupby(ticker_level)[c_close]
                .transform(_rsi)
            )
            # RSI Diff
            df_out['rsi_diff'] = df_out.groupby(ticker_level)['rsi'].diff()

        # --- 3. BOLLINGER BANDS ---
        if 'bollinger' in self.config:
            cfg = self.config['bollinger']

            window = cfg['window']
            std = cfg['std']
            
            # Function wrapper to handle parameters
            def _bb(x): return bollinger_bands(x, window, std)

            # group_keys=False:
            # Prevents group_col from being added to the index.
            bb_df = (
                df_out.groupby(ticker_level, group_keys=False)[c_close]
                .apply(_bb)
            )

            # Adding bollinger bands calculated data
            df_out[bb_df.columns] = bb_df

        # --- 4. MACD ---
        if 'macd' in self.config:
            cfg = self.config['macd']

            fast = cfg['fast']
            slow = cfg['slow']
            signal = cfg['signal']

            # Function wrapper to handle parameters
            def _macd(x): return macd(x, fast, slow, signal)

            # group_keys=False:
            # Prevents group_col from being added to the index.
            macd_df = (
                df_out.groupby(ticker_level, group_keys=False)[c_close]
                .apply(_macd)
            )
            
            # Adding MACD calculated data
            df_out[macd_df.columns] = macd_df

        # --- 5. RELATIVE VOLUME ---
        if 'rvol' in self.config:
            cfg = self.config['rvol']

            window = cfg['window']
            
            # Function wrapper to handle parameters
            def _rvol(x): return relative_volume(x, window)

            # Calculating relative volume
            df_out['rvol'] = (
                df_out.groupby(ticker_level)[c_vol]
                .transform(_rvol)
            )

        return df_out
    
    def construct_feature_matrix(
            self, 
            df_metrics: pd.DataFrame,
            selection: Dict[str, List[int]] = None
        ) -> pd.DataFrame:
            """
            **Phase 2: Selection & Lagging**
            
            Refines the calculated metrics into a finalized feature matrix. This
            phase handles the dimensionality of the input space by selecting
            specific columns and generating lagged observations to capture
            autocorrelation and temporal dependencies.

            Process:
                1. **Feature Pruning**: Filters the dataset to keep only the metrics 
                specified in the `selection` dictionary.
                2. **MultiIndex Lagging**: Generates $t-n$ observations using 
                `groupby().shift()`. This ensures that lags are calculated per
                asset, preventing cross-contamination between different tickers.
                3. **Feature Expansion**: Creates new columns with the
                suffix `_lag_n`.
            
            !!! tip "Machine Learning Recommendations"
                For most ML models (Random Forest, XGBoost, etc.), it is highly recommended to 
                use **Stationary Features**. Using raw prices or moving averages (like `bb_middle`) 
                can lead to poor generalization due to unit roots. 
                
                **Recommended Stationary Set (Default):**

                * **Returns**: `log_ret` (captures percentage change).
                * **Momentum**: `rsi`, `rsi_diff` (bounded between 0-100).
                * **Volatility**: `bb_pct_b`, `bb_width` (normalized volatility).
                * **Trend**: `macd_rel_hist` (price-normalized momentum).
                * **Volume**: `rvol` (normalized activity).
            
            Args:
                df_metrics (pd.DataFrame): Enriched MultiIndex DataFrame
                    from Phase 1 (`generate()` function).

                selection (Dict[str, List[int]], optional):

                    Dictionary mapping feature names to a list of desired lags.

                    - Use `[]` for the current value ($t$) only.
                    - Use `[1, 2]` for current value plus two previous steps.

                    Defaults to:

                    ```python
                    {
                        'log_ret': [],
                        'rsi': [],
                        'rsi_diff': [],
                        'bb_pct_b': [],
                        'bb_width': [],
                        'macd_rel_hist': [],
                        'rvol': []
                    }
                    ```

            Returns:
                (pd.DataFrame):
                    The finalized Feature Matrix (X). Columns are ordered by feature 
                    then by lag (e.g., `rsi`, `rsi_lag_1`, `rsi_lag_2`).

            Notes:
                - **Temporal Dynamics**: Adding lags is essential for non-sequential 
                models (like Random Forest or XGBoost) to "see" the trend.
                - **Data Leakage**: This method preserves the MultiIndex to ensure 
                that `shift` operations never mix data from different tickers at 
                the boundaries.
            """
            ticker_level = self.ticker_level

            # Validate ticker level is in MultiIndex Dataset
            validate_ticker_level(self.ticker_level, df_metrics)
            
            df_metrics = df_metrics.copy()

            # Initialize empty DataFrame with user's input index
            df_out = pd.DataFrame(index=df_metrics.index)

            if not selection:
                selection = {
                    'log_ret': [],
                    'rsi': [],
                    'rsi_diff': [],
                    'bb_pct_b': [],
                    'bb_width': [],
                    'macd_rel_hist': [],
                    'rvol': []
                    }

            for feature, lags in selection.items():
                if feature not in df_metrics.columns:
                    raise KeyError(f"Feature '{feature}' not found.")
                
                # 1. Add the Base Feature
                df_out[feature] = df_metrics[feature]
                
                # 2. Add Lags
                for lag in lags:
                    new_col_name = f"{feature}_lag_{lag}"

                    df_out[new_col_name] = (
                        df_metrics.groupby(ticker_level)[feature]
                        .shift(lag)
                    )
            
            return df_out
    
    def _validate_config(
            self,
            default_config: Dict[str, Dict[str, Any]],
            config: Dict[str, Dict[str, Any]]
    ):
        # Metrics validation
        supported_metrics = list(default_config.keys())
        user_metrics = list(config.keys())

        # Invalid user metrics
        invalid_metrics = [
            metric
            for metric in user_metrics
            if metric not in supported_metrics
        ]

        # If there is any invalid metrics, we specify them and raise error.
        if invalid_metrics:
            msg = (
                f"Unsupported metric(s) detected: {invalid_metrics}."
                f" Supported metrics are: {supported_metrics}"
            )

            raise KeyError(msg)
        
        # Parametrers Validation
        invalid_params = {}

        for metric in config:
            # Find not valid parameters.
            bad_params = [
                param
                for param in config[metric]
                if param not in default_config[metric]
            ]

            if bad_params:
                invalid_params[metric] = bad_params

        # If there is any metric with invalid parameters, we raise error.
        if invalid_params:
            error_msg = "\n".join(
                [f"  - {m}: {p}" for m, p in invalid_params.items()]
            )

            raise KeyError(f"Invalid parameter(s) found:\n{error_msg}")