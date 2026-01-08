# External Libraries
import pandas as pd
import numpy as np

# Imported from our library
from ..config.constants import EPSILON

def sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.0, 
    periods: int = 252
) -> float:
    r"""
    Calculates the annualized Sharpe Ratio (Ex-post).

    The Sharpe Ratio evaluates the risk-adjusted performance of an investment 
    by subtracting the risk-free rate from the investment's return and 
    dividing the result by the investment's standard deviation (volatility).

    Mathematical Formula:
        $$ S = \frac{R_p - R_f}{\sigma_p} \cdot \sqrt{T} $$

    Where:
        - $R_p$: Average period return.
        - $R_f$: Risk-free rate per period.
        - $\sigma_p$: Standard deviation of period returns.
        - $T$: Annualization factor (e.g., 252 for daily data).

    Args:
        returns (pd.Series): Time series of asset returns 
                             (simple returns recommended).
        risk_free_rate (float, optional): Annualized risk-free rate
                                          (e.g., 0.04 for 4%). Defaults to 0.0.
        periods (int, optional): Annualization factor. (252 for daily, 12 for
                                 monthly). Defaults to 252.

    Returns:
        (float): The annualized Sharpe Ratio.

    Interpretability:
        - **< 1.0**: Suboptimal risk-adjusted return.
        - **1.0 - 1.9**: Acceptable / Good.
        - **2.0 - 2.9**: Superior / Very Good.
        - **> 3.0**: Exceptional (often seen in High-Frequency Trading).
    """
    if len(returns) < 2:
        return np.nan

    # 1. Calculate Annualized Mean Return
    # We multiply the average daily return by the number of trading days
    avg_period_return = returns.mean()

    # 2. Calculate Annualized Volatility
    # We multiply standard deviation by sqrt(time)
    std_period_return = returns.std()

    if pd.isna(std_period_return) | (std_period_return < EPSILON):
        return 0.0
    

    # 3. Calculate Daily Sharpe
    # Adjusting Risk Free Rate to "period"
    period_rf = risk_free_rate / periods
    period_sharpe = (avg_period_return -period_rf) / std_period_return

    return period_sharpe * np.sqrt(periods)