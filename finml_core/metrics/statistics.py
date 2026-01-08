# External Libraries
import numpy as np
import pandas as pd

# Imported from our library
from ..config.constants import EPSILON

def simple_returns(prices: pd.Series, period: int = 1) -> pd.Series:
    r"""
    Calculates the arithmetic (simple) returns.

    $$R_t = \frac{P_t}{P_{t-n}} - 1$$

    Args:
        prices (pd.Series): Time series of asset prices.
        period (int, optional): The shift period. Defaults to 1.

    Returns:
        (pd.Series): Series of simple returns.
    
    ---
    """
    prev_prices = prices.shift(periods=period)
    safe_prev_prices = prev_prices.replace(0, EPSILON)

    return (prices / safe_prev_prices) - 1

def log_returns(prices: pd.Series, period: int = 1) -> pd.Series:
    r"""
    Calculates the logarithmic (continuously compounded) returns.

    Computes the log return as the difference between the natural logarithm
    of the price at time t and the natural logarithm of the price at time
    t-period.

    $$r_t = \ln(P_t) - \ln(P_{t-n})$$

    Args:
        prices (pd.Series): Time series of asset prices. Must be strictly
                            positive.
        period (int, optional): The shift period to calculate returns. 
            Defaults to 1 (daily returns if data is daily).

    Returns:
        (pd.Series): Series of log returns. The first 'period' values will be NaN.

    Notes:
        Log returns are preferred in quantitative finance because:

        1. Time Additivity: Sum of log returns equals the total period return.
        2. Statistical Properties: They are often assumed to be normally
                                   distributed.
                
    ---
    """
    # We handle division by zero if prices == 0 (though rare in prices).
    safe_prices = prices.replace(0, EPSILON)

    return np.log(safe_prices).diff(periods=period)

def volatility(
    returns: pd.Series, annualize: bool = True, scale: int = 252
) -> float:
    r"""
    Calculates the volatility (sample standard deviation) of a return series.
    
    Optionally applies an annualization factor.

    $$ \sigma_{annual} = \sigma_{period} \times \sqrt{T} $$

    Args:
        returns (pd.Series): Time series of asset returns (log or simple).
        annualize (bool, optional): If True, scales the volatility to an 
            annual figure. Defaults to True.
        scale (int, optional): The annualization factor. 
            Use 252 for daily data, 12 for monthly data. Defaults to 252.

    Returns:
        (float): The standard deviation of the series.

    Notes:
        This function uses N-1 degrees of freedom (sample standard deviation),
        which is the default behavior in pandas.std().
    """
    # ddof=1 is default in pandas (sample std dev),
    # which is correct for finance.
    vol = returns.std()
    
    if annualize:
        return vol * np.sqrt(scale)
    
    return vol

def rolling_mean(prices: pd.Series, window: int) -> pd.Series:
    r"""
    Calculates the Simple Moving Average (SMA).

    Computes the unweighted mean of the previous 'window' data points.

    $$ \mu_t = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i} $$

    Args:
        prices (pd.Series): Time series data.
        window (int): The size of the moving window.

    Returns:
        (pd.Series): The rolling mean. The first 'window-1' values will be NaN.
    
    ---
    """
    return prices.rolling(window=window).mean()

def rolling_std(prices: pd.Series, window: int) -> pd.Series:
    r"""
    Calculates the Moving Standard Deviation.

    Computes the standard deviation of the previous 'window' data points.
    Often used as a measure of dynamic volatility (e.g., Bollinger Bands width).

    $$\sigma_t = \sqrt{\frac{1}{n-1} \sum_{i=0}^{n-1} (P_{t-i} - \bar{x}_t)^2}$$

    Where:

    - $\sigma_t$: Rolling standard deviation at time $t$.
    - $n$: Lookback period (window size).
    - $P_{t-i}$: Observation at time $t-i$.
    - $\bar{x}_t$: Moving average (mean) of the window at time $t$.

    Args:
        prices (pd.Series): Time series data.
        window (int): The size of the moving window.

    Returns:
        (pd.Series): The rolling standard deviation.
    
    ---
    """
    # ddof=1 is default in pandas (sample std dev),
    # which is correct for finance.
    return prices.rolling(window=window).std()