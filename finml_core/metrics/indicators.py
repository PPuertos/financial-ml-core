# External Libraries
import pandas as pd

# Imported from our library
from .statistics import rolling_mean, rolling_std
from ..config.constants import EPSILON

def bollinger_bands(
    prices: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    r"""
    Calculates Bollinger Bands and derived volatility metrics
    (%B and Bandwidth).

    Bollinger Bands consist of a middle band (SMA) and two outer bands
    calculated using the standard deviation of the price series.

    Mathematical Formulas:
        - $$ \text{MB}_t = \mu_{P,n} $$
        - $$ \text{UB}_t = \mu_{P,n} + (k \cdot \sigma_{P,n}) $$
        - $$ \text{LB}_t = \mu_{P,n} - (k \cdot \sigma_{P,n}) $$

    Where:
        - $\mu_{P,n}$: The rolling mean (SMA) of price $P$ over window $n$.
        - $\sigma_{P,n}$: The rolling standard deviation of price $P$ over
                          window $n$.
        - $k$: Standard deviation multiplier.

    Args:
        prices (pd.Series): Time series of asset prices.
        window (int, optional): Moving average window size. Defaults to 20.
        num_std (float, optional): Number of standard deviations (k).
                                   Defaults to 2.0.

    Returns:
        (pd.DataFrame): A MultiIndex-compatible DataFrame containing:

            * `bb_middle`: The Simple Moving Average (SMA).
            * `bb_upper`: Upper volatility band.
            * `bb_lower`: Lower volatility band.
            * `bb_pct_b`: Price position relative to the bands
                          (1.0 = Upper, 0.0 = Lower).
            * `bb_width`: Normalized width of the bands, measuring
                          relative volatility.
    
    ---
    """
    # Calculating Simple Moving Average and Standard Deviation
    sma = rolling_mean(prices, window=window)
    std = rolling_std(prices, window=window)

    # Calculate Bands
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)

    # Calculate Derived Indicators
    # "Height" of the band
    band_range = upper - lower
    # We handle division by zero if upper == lower (though rare in prices).
    safe_band_range = band_range.replace(0, EPSILON)
    # %B: 1.0 means price is at upper band, 0.0 at lower band.
    percent_b = (prices - lower) / safe_band_range

    # We handle division by zero if sma == 0 (though rare rolling means).
    safe_sma = sma.replace(0, EPSILON)
    # Bandwidth: Width relative to the moving average
    bandwidth = band_range / (safe_sma)

    # 4. Pack into a clean DataFrame
    return pd.DataFrame({
        'bb_middle': sma,
        'bb_upper': upper,
        'bb_lower': lower,
        'bb_pct_b': percent_b,
        'bb_width': bandwidth
    }, index=prices.index)

def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    r"""
    Calculates the Relative Strength Index (RSI) using Wilder's Smoothing.

    The RSI calculates a ratio of the recent upward price movements to the
    absolute price movements.

    $$ RSI = 100 - \frac{100}{1 + RS} $$

    Args:
        prices (pd.Series): Time series of asset prices.
        period (int, optional): The lookback period. 
            Defaults to 14 (Standard industry value proposed by Wilder).

    Returns:
        (pd.Series): The RSI values (0-100).

    Notes:
        This implementation uses Wilder's Smoothing ($\alpha = 1/N$), which is 
        standard in technical analysis. This creates a recursive dependency,
        so early values may vary slightly depending on the data start point.

        **Why `adjust=False`?** This mimics the recursive formula used in most
        trading platforms: $y_t = (1-\alpha)y_{t-1} + \alpha x_t$.
        Using `adjust=True` (pandas default) would calculate weights based on
        finite history, leading to values that diverge from standard market
        indicators.
    """
    # 1. Calculate daily changes
    delta = prices.diff()

    # 2. Separate Gains (U) and Losses (D)
    # clip(lower=0) keeps positive values, replaces negatives with 0
    gain = delta.clip(lower=0)
    # clip(upper=0) keeps negative values.
    # We negate them to get positive losses.
    loss = -delta.clip(upper=0)

    # 3. Calculate Smoothed Averages (Wilder's Method)
    # alpha = 1/period implies com = period - 1
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    # 4. Calculate RS and RSI

    # We handle division by zero if avg_loss == 0.
    safe_avg_loss = avg_loss.replace(0, EPSILON)
    rs = avg_gain / safe_avg_loss
    
    # Handle division by zero implicitly (inf -> 100)
    return 100 - (100 / (1 + rs))

def macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    r"""
    Calculates the Moving Average Convergence Divergence (MACD).

    The MACD is a trend-following momentum indicator that shows the
    relationship between two exponential moving averages of an asset's price.

    Mathematical Formulas:

    - $$ \text{MACD Line} =  \text{EMA}_{fast}(P) - \text{EMA}_{slow}(P) $$
    - $$ \text{Signal Line} = \text{EMA}_{signal}(\text{MACD Line}) $$
    - $$ \text{Histogram} = \text{MACD Line} - \text{Signal Line} $$
    - $$ \text{Relative Hist} = \frac{\text{Histogram}}{P} $$

    Args:
        prices (pd.Series): Time series of asset prices.
        fast (int, optional): Fast EMA span. Defaults to 12.
        slow (int, optional): Slow EMA span. Defaults to 26.
        signal (int, optional): Signal EMA span. Defaults to 9.

    Returns:
        (pd.DataFrame): A DataFrame containing:

            * `macd_line`: The difference between fast and slow EMAs.
            * `macd_signal`: EMA of the MACD line (smoothing).
            * `macd_hist`: The distance between the MACD line and the
                            signal line.
            * `macd_rel_hist`: Price-normalized histogram
                                (stationary feature for ML).
    
    Notes:
        - **EMA Calculation**: Uses Standard EMA ($\alpha = 2/(N+1)$).
        - 'macd_rel_hist' is a custom metric: Histogram / Price. This normalizes
          the volatility relative to the asset price, useful for ML features.
        - **Why `adjust=False`?** This mimics the recursive formula used in most
          trading platforms: $y_t = (1-\alpha)y_{t-1} + \alpha x_t$.
          Using `adjust=True` (pandas default) would calculate weights based on
          finite history, leading to values that diverge from standard market
          indicators.

    ---
    """
    # 1. Calculate Fast and Slow EMAs (Standard EMA uses 'span')
    ema_fast = prices.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, adjust=False, min_periods=slow).mean()

    # 2. MACD Line (Velocity)
    macd_line = ema_fast - ema_slow

    # 3. Signal Line (Smoothed Velocity)
    signal_line = macd_line.ewm(
        span=signal,
        adjust=False,
        min_periods=signal
    ).mean()

    # 4. Histogram (Acceleration / Momentum)
    # Represents the distance between the MACD and its signal.
    histogram = macd_line - signal_line

    # 5. Normalized MACD (User Custom Feature)
    # Normalizes the momentum relative to the asset price.
    # We handle division by zero if avg_loss == 0.
    safe_prices = prices.replace(0, EPSILON)
    macd_norm = histogram / safe_prices

    return pd.DataFrame({
        'macd_line': macd_line,
        'macd_signal': signal_line,
        'macd_hist': histogram,
        'macd_rel_hist': macd_norm
    }, index=prices.index)

def relative_volume(
    volume: pd.Series, window: int = 20
) -> pd.Series:
    r"""
    Calculates Relative Volume (RVol).

    RVol measures the current trading activity relative to its historical average. 
    It is a key feature for identifying "smart money" institutional activity 
    and confirming price breakouts.

    Mathematical Formula:
        $$ \text{RVol}_t = \frac{V_t}{\frac{1}{n} \sum_{i=0}^{n-1} V_{t-i}} $$

    Where:
        - $V_t$: Trading volume at time $t$.
        - $n$: Lookback window (moving average period).

    Args:
        volume (pd.Series): Time series of trading volume.
        window (int, optional): Lookback window. Defaults to 20
                                (approx. 1 trading month).

    Returns:
        (pd.Series): A ratio indicating relative volume.

            - Values **> 1.0**: High abnormal activity (Surge in interest).
            - Values **< 1.0**: Low activity (Typical of consolidations).

    Notes:
        - **Why 20 days?** Represents approximately one trading month.
        - **Stationarity**: RVol is a stationary feature, making it highly 
          suitable for Machine Learning models without further differencing.
        - **Numerical Stability**: Uses `EPSILON` (1e-6) replacement for zero 
          moving averages to prevent `inf` values in low-liquidity assets.

    ---
    """
    # 1. Calculate the average volume (SMA)
    # We use our own library's function
    sma_vol = rolling_mean(volume, window=window)

    # We handle division by zero if sma_vol == 0.
    safe_sma_vol = sma_vol.replace(0, EPSILON)

    # 2. Calculate Ratio
    # We add epsilon (1e-6) to avoid DivisionByZero errors
    return volume / safe_sma_vol