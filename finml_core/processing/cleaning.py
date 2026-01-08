# External Libraries
import pandas as pd
import numpy as np

# From our module
from ..utils.validation import validate_date_level, validate_ticker_level

class DataCleaner:
    """
    Module responsible for data sanitization and stability.
    
    Standardizes the handling of infinities (infs) and null values (NaNs) 
    to ensure the numerical stability of the pipeline before ML ingestion.
    
    Key Features:
        1. Multi-Asset Safety: Applies operations per-ticker to prevent data
                               leakage.
        2. Look-ahead Bias Prevention: Uses conservative filling methods
                                       (ffill limit=1).
        3. Strict Validation: Raises errors if data quality standards
                              are not met.
    """

    def __init__(
            self,
            ticker_level: str,
            date_level: str,
            method: str = 'ffill'
    ):
        """
        Args:
            ticker_level (str): Name of the MultiIndex level containing
                the asset identifiers (e.g., 'Ticker').

            date_level (str): Name of the MultiIndex level containing
                the timestamps (e.g., 'Date'). Required for re-indexing 
                during edge trimming.

            method (str): Strategy for handling NaNs. for now only 'ffill'
                          (Forward Fill) is supported as it is standard in
                          in financial time series to handle minor gaps
                          (e.g., holidays) without look-ahead bias.
        ---
        """
        self.method = method
        self.ticker_level = ticker_level
        self.date_level = date_level
    
    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        **Execution Protocol: Strict Sanitization.**
        
        Runs the data through a 4-stage quality gate to ensure
        numerical integrity.
        
        Protocol Steps:
            1. Trim Edges: Removes leading/trailing NaNs (warm-up periods).
            2. Fill Gaps: Applies limited forward fill for internal gaps.
            3. NaN Validation: checks for remaining Nulls;
                               raises Error if found.
            4. Inf Validation: checks for Infinite values;
                               raises Error if found.

        Args:
            df (pd.DataFrame): Input DataFrame (Raw or Feature Matrix).
                               Must have a MultiIndex (Date, Ticker).

        Returns:
            (pd.DataFrame): A clean, numerically stable DataFrame ready for 
                          modeling.
        
        Raises:
            ValueError: If NaNs persist after cleaning or if Infs are detected.
        """
        df_clean = df.copy()

        # Validations before computing
        validate_ticker_level(self.ticker_level, df_clean)
        validate_date_level(self.date_level, df_clean)

        # First Step: Trim Edges if needed. Else return dataset
        if df_clean.isnull().values.any():
            df_clean = (
                df_clean.groupby(self.ticker_level,group_keys=False)
                .apply(self._nan_trim_edges)
            )
        else:
            return df_clean
        
        # 2nd Step: Forward fill gaps if needed. Else return dataset.
        if df_clean.isnull().values.any():
            df_clean = (
                df_clean.groupby(self.ticker_level,group_keys=False)
                .apply(self._nan_fill_forward)
            )
        else:
            return df_clean
        
        if df_clean.isnull().values.any():
            msg = (
                "CRITICAL: Data still contains NaNs after Trim & Ffill.\n"
                "Possible causes: Gaps > 1 timestep, "
                "or feature calculation errors.\n\n"
                f"NaN's Report:\n{self._nan_report(df_clean)}"

            )

            raise ValueError(msg)
        
        if np.isinf(df_clean).values.any():
            msg = (
                "Infinite values detected."
                f"Inf's Report:\n{self._inf_report(df_clean)}"
            )

            raise ValueError(msg)
        else:
            return df_clean

    def _nan_trim_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        **Internal Strategy: Subset Edge Trimming**
        
        Trims NaNs from the beginning (leading) and end (trailing) of the 
        provided DataFrame subset.
        
        Usage Context:
        Designed to be called within a `groupby().apply()` operation to ensure
        trimming occurs per-ticker.
        
        Logic:
        1. Temporarily aligns the index to the Date level.
        2. Identifies the global valid time range (First Valid Index to 
           Last Valid Index) across all columns in the subset.
        3. Slices the data to keep only the valid core.

        Args:
            df (pd.DataFrame): A subset of data (e.g., a single ticker's history).

        Returns:
            pd.DataFrame: The trimmed subset.
        """
        # Replacing Index to Date instead of (Date, Ticker)
        calc_df = df.copy()
        calc_df.index = calc_df.index.get_level_values(self.date_level)

        # Find first and last valid index for each column
        starts = calc_df.apply(pd.Series.first_valid_index)
        ends = calc_df.apply(pd.Series.last_valid_index)

        # Find global first and last index to use as range
        global_start = starts.max()
        global_end = ends.min()

        # Checking for empty columns or invalid range
        if starts.hasnans | (global_start >= global_end):
            return df.iloc[:0]
        
        return df[global_start:global_end]

    def _nan_fill_forward(
            self,
            df: pd.DataFrame,
            limit: int = 1
    ) -> pd.DataFrame:
        """
        INTERNAL STRATEGY: Subset Forward Fill.
        
        Fills internal NaN gaps using Forward Fill (ffill) on the provided 
        DataFrame subset.
        
        Usage Context:
        Must be applied to a single asset's history (e.g., via groupby) 
        to prevent data leakage (mixing prices of Asset A with Asset B).

        Args:
            df (pd.DataFrame): A subset of data containing internal gaps.
            limit (int): Max number of consecutive NaNs to fill (default=1).

        Returns:
            pd.DataFrame: The subset with minor gaps filled.
        """
        df_clean = df.copy()

        # Forward Fill: Propagate last valid observation forward
        return df_clean.ffill(limit=limit)

    def _inf_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DIAGNOSTIC TOOL: Anomaly Reporter.
        
        Generates a filtered view of the dataset focusing only on 
        problematic elements (Infs).
        
        Logic:
        1. Aggregates boolean mask by Ticker.
        2. Filters out clean tickers and clean columns.
        3. Returns a matrix showing exactly where the errors are.

        Args:
            df (pd.DataFrame): The dirty DataFrame.

        Returns:
            (pd.DataFrame): A subset of the data containing only the 
                          counts/locations of errors.
        """
        # Infs Diagnostic
        infs = np.isinf(df).groupby(self.ticker_level).sum()
        infs = infs.loc[infs.sum(axis=1) > 0, infs.sum(axis=0) > 0]

        return infs

    def _nan_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DIAGNOSTIC TOOL: Anomaly Reporter.
        
        Generates a filtered view of the dataset focusing only on 
        problematic elements (NaNs).
        
        Logic:
        1. Aggregates boolean mask by Ticker.
        2. Filters out clean tickers and clean columns.
        3. Returns a matrix showing exactly where the errors are.

        Args:
            df (pd.DataFrame): The dirty DataFrame.

        Returns:
            (pd.DataFrame): A subset of the data containing only the 
                          counts/locations of errors.
        """
        # NaNs Diagnostic
        nans = df.isnull().groupby(self.ticker_level).sum()
        nans = nans.loc[nans.sum(axis=1) > 0, nans.sum(axis=0) > 0]

        return nans