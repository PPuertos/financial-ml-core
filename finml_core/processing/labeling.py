# External Libraries
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

# Importing from our library
from ..utils.validation import (
    validate_col_map, validate_ticker_level, validate_date_level
)

class TripleBarrierLabeling:
    r"""
    Labels financial time-series using the Triple Barrier Method (TBM).

    Unlike standard fixed-horizon labeling, TBM creates dynamic labels based on 
    volatility-adjusted price targets. It effectively captures the path-dependent 
    nature of financial assets by monitoring three concurrent barriers:

    1. **Upper Barrier (Take Profit)**: Hit when price appreciation reaches 
       $P_t \cdot (1 + \text{tp}_{\text{mult}} \cdot \sigma_t)$.
    2. **Lower Barrier (Stop Loss)**: Hit when price depreciation reaches 
       $P_t \cdot (1 - \text{sl}_{\text{mult}} \cdot \sigma_t)$.
    3. **Vertical Barrier (Time Limit)**: Hit when neither price barrier is 
       touched within a fixed window $T$.

    Mathematical Outcomes:
        * **1 (Long)**: Upper barrier hit first.
        - **-1 (Short)**: Lower barrier hit first.
        * **0 (Hold)**: Vertical barrier (Time Limit) reached.

    Note:
        This implementation is optimized for MultiIndex DataFrames (Ticker, Date) 
        and uses vectorized NumPy operations for the internal path-scanning loop.
    """
    def __init__(
        self,
        col_map: Dict[str, str],
        ticker_level: str,
        date_level: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            col_map (Dict): Mapping for OHLCV columns.

                Required to compute the labeling:
                    `{'close': ..., 'high': ..., 'low': ...}`

                Example:
                    ```python
                    {
                        'close': 'Close',
                        'high': 'High',
                        'low': 'Low',
                    }
                    ```
                which is yfinance standard.

            ticker_level (str): MultiIndex level name for asset identifiers
                (e.g., 'Ticker').

            date_level (str): MultiIndex level name for timestamps
                (e.g., 'Date').

            config (Dict[str, Any], optional): Hyperparameters for the barriers.

                | Key | Type | Default | Description |
                | :--- | :--- | :--- | :--- |
                | `stop_loss_multiplier` | float | 2.0 | Volatility multiplier for SL. |
                | `take_profit_multiplier`| float | 2.0 | Volatility multiplier for TP. |
                | `time_limit` | int | 10 | Max periods to hold (Vertical Barrier). |
                | `vol_span` | int | 100 | Span for EWM Volatility calculation. |

        ---
        """
        self.ticker_level = ticker_level
        self.date_level = date_level

        # default parameters for triple barrier method labeling
        default_config = dict(
            stop_loss_multiplier = 2.0,
            take_profit_multiplier = 2.0,
            time_limit = 10,
            vol_span = 100
        )

        if config:
            # config parameter content validation
            self._validate_config(config)

            # {**default_dict, **user_dict}:
            # default updated with user-provided values
            default_config = {**default_config, **config}

        self.sl_mult = default_config.get('stop_loss_multiplier')
        self.tp_mult = default_config.get('take_profit_multiplier')
        self.time_limit = default_config.get('time_limit')
        self.vol_span = default_config.get('vol_span')

        validate_col_map(col_map)
        self.map = col_map

    def compute_outcomes(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        r"""
        Executes the Triple Barrier labeling process across all assets.

        This method orchestrates the full pipeline:
            1. **Volatility Normalization**: Computes dynamic $\sigma_t$ per ticker.
            2. **MultiIndex Grouping**: Isolates price paths by asset to prevent 
            cross-ticker data leakage during barrier scanning.
            3. **Numpy-Vectorized Search**: Triggers the internal search engine for 
            first-touch events on future price paths.

        Args:
            df (pd.DataFrame): MultiIndex DataFrame containing price series.

        Returns:
            (pd.DataFrame): 
                A DataFrame indexed like `df` with labeling results:

                - `target_side`: The final label {1, 0, -1}.
                - `transaction_return`: Unrealized return at the touch moment.
                - `t1`: Timestamp when the first barrier was hit.
                - `upper_barrier` & `lower_barrier`: The volatility-adjusted price levels.
                - `time_to_barrier`: Integer steps taken to reach the outcome.
                - `both_barriers_hit`: Boolean flag for high-volatility gap cases.

        Raises:
            KeyError: If required columns or MultiIndex levels are missing.
        """
        ticker_level = self.ticker_level

        # Validate ticker level is in MultiIndex Dataset
        validate_ticker_level(ticker_level, df)
        # Validate date level is in MultiIndex Dataset
        validate_date_level(self.date_level, df)

        df_out = df.copy()

        # Retrieve required column names from map
        close_col = self.map.get('close')
        high_col = self.map.get('high')
        low_col = self.map.get('low')

        # Validation
        req_cols = [close_col, high_col, low_col]
        
        for c in req_cols:
            if c not in df_out.columns:
                raise KeyError(
                    f"Column '{c}' not found in DataFrame."
                    "required to calculate labels with triple barrier method."
                )
        
        if ticker_level not in df.index.names:
            raise KeyError(f"'ticker_level' not defined in df's MultiIndex.")
        
        df_out = df_out[req_cols]
            
        # 1. Get Volatility and Scale it to 'time_limit' periods
        df_out['volatility'] = (
            df_out.groupby(ticker_level)[close_col]
            .transform(self._calculate_volatility)
        ) * np.sqrt(self.time_limit)

        # 2. Compute Events
        def barrier_outcomes(x): return (
                    self._compute_barrier_outcomes(
                        x, close_col, high_col, low_col
                    )
        )

        # Select only necessary input features to optimize memory usage during
        # barrier computation
        required_cols = [close_col, high_col, low_col, 'volatility']

        # group_keys=False: Prevents 'Ticker' from being added to the index.
        labeling_results = (
            df_out.groupby(ticker_level, group_keys=False)[required_cols]
            .apply(barrier_outcomes)
        )
        
        return labeling_results

    def _compute_barrier_outcomes(
        self,
        df,
        close_col: str, 
        high_col: str, 
        low_col: str
    ) -> pd.DataFrame:
        """
        Internal loop to determine barrier touches,
        looking at Future Highs and Lows.
        """
        # Convert to numpy arrays
        closes = df[close_col].values
        highs = df[high_col].values
        lows = df[low_col].values
        vols = df['volatility'].values
        dates = df.index.get_level_values(self.date_level).values

        # Number of periods of the given series
        n = df.shape[0]

        # Initialization of columns to be calculated

        # Label and returns arrays
        out_labels = np.full(n, np.nan)
        out_returns = np.full(n, np.nan)

        # Barriers arrays
        upper_barrier_series = np.full(n, np.nan)
        lower_barrier_series = np.full(n, np.nan)

        # Time to hit a barrier
        time_to_barrier_series = np.full(n, np.nan)

        # Date when barrier was hit
        t1_serie = np.full(n, pd.NaT)
        
        # Binary column to return wether or not both barriers hit
        both_barriers_hit = np.full(n, False)

        # Limit index to avoid out of bounds
        limit_idx = n - self.time_limit

        for i in range(limit_idx):
            current_price = closes[i]
            current_vol = vols[i]

            # Skip invalid data
            if np.isnan(current_vol):
                continue

            # Define Barriers
            # Upper = Price + (Vol * Mult)
            # Lower = Price - (Vol * Mult)
            upper_barrier = current_price * (1 + self.tp_mult * current_vol)
            lower_barrier = current_price * (1 - self.sl_mult * current_vol)

            # Add it to the initialized series
            upper_barrier_series[i] = upper_barrier
            lower_barrier_series[i] = lower_barrier
            
            # Future Prices (not seen by actual volatility)
            future_highs = highs[i+1 : i+1+self.time_limit]
            future_lows = lows[i+1 : i+1+self.time_limit]
            future_closes = closes[i+1 : i+1+self.time_limit]

            # Future Dates
            future_dates = dates[i+1 : i+1+self.time_limit]

            # Validating if any of the high values hit the ceil
            hit_upper_mask = future_highs >= upper_barrier
            any_hit_upper = np.any(hit_upper_mask)

            # Validating if any of the low values hit the floor
            hit_lower_mask = future_lows <= lower_barrier
            any_hit_lower = np.any(hit_lower_mask)

            # SCENARIO 1: Hit Take Profit
            if any_hit_upper and not any_hit_lower:
                # Take profit Signal
                out_labels[i] = 1

                # First high idx that hit the ceil (Take Profit)
                tp_idx = hit_upper_mask.argmax()
                # Take profit value
                take_profit = upper_barrier

                # Calculate Positive Returns
                out_returns[i] = (take_profit / current_price) - 1

                # Date when barrier was hit
                t1_serie[i] = future_dates[tp_idx]

                # Time it took to hit the upper barrier
                time_to_barrier = tp_idx + 1
                # Add it to our initialized series
                time_to_barrier_series[i] = time_to_barrier

            # SCENARIO 2: Hit Stop Loss
            elif any_hit_lower and not any_hit_upper:
                # Stop Loss Signal
                out_labels[i] = -1

                # First low idx that hit the floor (Stop Loss)
                sl_idx = hit_lower_mask.argmax()
                # Stop loss value
                stop_loss = lower_barrier

                # Calculate Negative Returns
                out_returns[i] = (stop_loss / current_price) - 1

                # Date when barrier was hit
                t1_serie[i] = future_dates[sl_idx]

                # Time it took to hit the lower barrier
                time_to_barrier = sl_idx + 1
                # Add it to our initialized series
                time_to_barrier_series[i] = time_to_barrier

            # SCENARIO 3: Hit Both (Extreme Volatility)
            elif any_hit_upper and any_hit_lower:
                # Take Profit and Stop Loss Idx's
                tp_idx = hit_upper_mask.argmax()
                sl_idx = hit_lower_mask.argmax()

                # CASE 1: Take profit occurred before stop loss
                if tp_idx < sl_idx:
                    # Take Profit Signal
                    out_labels[i] = 1

                    # Take profit value
                    take_profit = upper_barrier
                    # Calculate take profit returns
                    out_returns[i] = (take_profit / current_price) - 1

                    # Date when barrier was hit
                    t1_serie[i] = future_dates[tp_idx]

                    # Time it took to hit the upper barrier
                    time_to_barrier = tp_idx + 1
                    # Add it to out initialized series
                    time_to_barrier_series[i] = time_to_barrier

                # CASE 2: Stop loss occurred before or at the same period as
                #         the take profit
                else:
                    # Pessimistic assumption: Stop Loss Signal
                    out_labels[i] = -1

                    # Stop loss value
                    stop_loss = lower_barrier
                    
                    # Calculate stop loss returns
                    out_returns[i] = (stop_loss / current_price) - 1

                    # Checking if both barriers was hit the same day
                    if tp_idx == sl_idx:
                        both_barriers_hit[i] = True

                    # Date when barrier was hit (Worst Case Scenario)
                    t1_serie[i] = future_dates[sl_idx]

                    # Time it took to hit the barrier
                    time_to_barrier = sl_idx + 1
                    # Add it to out initialized series
                    time_to_barrier_series[i] = time_to_barrier

            # SCENARIO 4: Time Limit Hit
            else:
                # Hold Signal
                out_labels[i] = 0

                # Market on Close Value (MOC)
                final_price = future_closes[-1]
                # Calculating MOC returns
                out_returns[i] = (final_price / current_price) - 1

                # Vertical Barrier Hit
                t1_serie[i] = future_dates[-1]

                # Time to vertical barrier (time limit)
                time_to_barrier_series[i] = self.time_limit

        # Returning labels and returns by transaction
        return pd.DataFrame({
            "target_side": out_labels,
            "transaction_return": out_returns,
            "t1": t1_serie,
            "lower_barrier": lower_barrier_series,
            "upper_barrier": upper_barrier_series,
            "time_to_barrier": time_to_barrier_series,
            "both_barriers_hit": both_barriers_hit
        }, index=df.index)

    def _calculate_volatility(self, prices: pd.Series) -> pd.Series:
        """
        Calculates Exponentially Weighted Moving Standard Deviation (EWMSD)
        of returns.
        """
        # Returns: (Pt / Pt-1) - 1
        returns = prices.pct_change()
        # EWM Standard Deviation (Standard in Risk Management)
        return returns.ewm(
            span=self.vol_span,
            adjust=False,
            min_periods=self.vol_span
        ).std()
    
    def _validate_config(
            self,
            config: Dict[str, Any]
    ):
        supported_metrics = [
            'stop_loss_multiplier', 'take_profit_multiplier',
            'time_limit', 'vol_span'
        ]

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
        
        # Validating integer parameters
        int_params = ['time_limit', 'vol_span']
        invalid_int_params = [
            param
            for param in user_metrics
            if (param in int_params) &
            (not isinstance(config[param], int))
        ]

        if invalid_int_params:
            raise KeyError(
                f"Parameter(s) '{invalid_int_params}' must be an integer."
            )
        
        # Validating float parameters
        float_params = ['stop_loss_multiplier', 'take_profit_multiplier']
        invalid_float_params = [
            param
            for param in user_metrics
            if (param in float_params) &
            (not isinstance(config[param], (float, int)))
        ]

        if invalid_float_params:
            raise KeyError(
                f"Parameter(s) '{invalid_float_params}' must be a float."
            )