import pandas as pd
import yfinance as yf
from datetime import datetime as dt
from typing import List, Optional

class MarketDataLoader:
    """
    ## Automated Data Ingestor and Standardizer.

    This class acts as the primary gateway for raw market data. Its core mission 
    is to abstract the complexity of external APIs, transforming heterogeneous 
    data sources into the internal standardized **Long-Format MultiIndex** required by the entire library ecosystem.

    The engine ensures that regardless of the provider, the final output strictly 
    adheres to a `[Date, Ticker]` MultiIndex structure with standardized OCHLV columns.
    """

    def __init__(self, source: str = 'yfinance'):
        """
        The constructor initializes the loader by selecting a specific data provider 
        registry. This setup determines the internal mapping logic that will be 
        applied during the standardization process.

        Args:
            source (str): Provider name from the 
                [`PROVIDERS`][finml_core.config.providers] registry.

        ---
        """
        self.source = source
    
    def get_financial_data(
        self,
        tickers: List[str],
        etf: str,
        start: str,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
            Fetches and standardizes financial data from the configured source.

            This method acts as a high-level factory. It ensures that regardless of the 
            source's original API format, the output is consistently aligned and 
            ready for the `DatasetGenerator` or any other standalone class within the 
            library (e.g., `FeatureGenerator`, `TripleBarrierLabeling`).

            Args:
                tickers (List[str]): List of asset symbols to fetch.
                etf (str): Reference ETF (e.g., '^GSPC') used to define the 
                    market calendar and align trading days.
                start (str): Start date/time. Supports various resolutions
                    (e.g., `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`) depending on the
                    providers capability.
                end (str, optional): End date/time. Supports the same resolutions 
                    as `start`. Defaults to today.

            Returns:
                (pd.DataFrame): A standardized MultiIndex DataFrame (Date, Ticker) 
                    containing adjusted OHLCV data, ready for quantitative analysis.

            Raises:
                KeyError: If the `source` provided during initialization is not 
                    yet implemented in the data fetching logic.
            """
        if self.source == 'yfinance':
            df = self._get_yfinance_data(
                tickers=tickers,
                etf=etf,
                start=start,
                end=end
            )
        else:
            raise KeyError(
                f"data_source={self.source} not supported. "
                "Current supported data sources: ['yfinance']"
            )

        return df

    def _get_yfinance_data(
        self,
        tickers: List[str],
        etf: str,
        start: str,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """Downloads, stacks, and cleans data from Yahoo Finance."""

        if not end:
            end = dt.today().strftime('%Y-%m-%d')

        print(f"Downloading {len(tickers)} tickers + {etf} reference...")

        # Download including ETF (Reference for Market Days)
        download_list = [etf] + tickers

        df = yf.download(
            tickers=download_list,
            start=start,
            end=end,
            auto_adjust=True
        )

        if df.empty:
            raise ValueError("No data downloaded. Check tickers or internet.")

        # Remove ETF (Reference)
        # We used it to force the date range, now we drop it.
        df = df.drop(columns=[etf], level=1)

        # Stacking -> Multi Index (Date, Ticker)
        df = df.stack(level=1, future_stack=True)
        df.columns.name = None

        # Sorting Values
        df = df.sort_index()

        return df