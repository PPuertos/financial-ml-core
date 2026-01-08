# Importing External Libraries
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# Importing Local Modules
from ..data.loader import MarketDataLoader
from ..processing.cleaning import DataCleaner
from ..processing.features import FeatureGenerator
from ..processing.labeling import TripleBarrierLabeling
from ..config.providers import PROVIDERS
from ..utils.validation import validate_df_columns

class DatasetGenerator:
    """
    Master Orchestrator for Financial Machine Learning Pipelines.

    This class centralizes the end-to-end workflow of a quantitative trading strategy:
        1. **Data Acquisition**: Either via automated download or manual injection.
        2. **Cleaning**: Handles MultiIndex alignment, NaNs, and infinite values.
        3. **Feature Engineering**: Generates technical indicators and temporal lags.
        4. **Labeling**: Executes the Triple Barrier Method for target definition.
        5. **Consolidation**: Aligns $X$ and $y$ by removing inconsistent 'warm-up' periods.

    # Operation Modes
    The generator adapts to two primary user workflows:

    * **Automated Mode (Beginner-Friendly)**: Uses pre-configured mappings from the 
        internal registry. You only need to provide a `data_source` (e.g., 'yfinance') 
        and the list of tickers in the `run` method.
    * **Custom Mode (Pro/Injected)**: Designed for users with proprietary datasets 
        or unsupported APIs. By providing a `custom_provider` dictionary, the 
        orchestrator switches to an injection-only logic, requiring a pre-loaded 
        DataFrame in the `run` method.

    Attributes:
        analysis_data (pd.DataFrame): Full dataset available after `run()`, 
            containing original OHLCV, all indicators, and labeling details.
    """

    def __init__(
        self,
        data_source: str = 'yfinance',
        custom_provider: Optional[Dict[str, Any]] = None,
        feature_config: Optional[Dict[str, Dict[str, Any]]] = None,
        feature_selection: Optional[Dict[str, List[int]]] = None,
        label_config: Optional[Dict[str, Any]] = None
    ):
        """
        # Pipeline Configuration & Engine Setup

        Instantiating this class initializes the core modular engines required for 
        the full data lifecycle. This constructor acts as a factory, setting up 
        the internal instances of `MarketDataLoader`, `FeatureGenerator`, 
        `TripleBarrierLabeling`, and `DataCleaner` based on the selected mode.

        It determines whether the pipeline will operate in **'Automated'** mode 
        (leveraging predefined provider registries) or **'Custom'** mode 
        (handling user-injected datasets and coordinate maps).

        Args:
            data_source (str):
                Name of the provider in
                [ `PROVIDERS` ][finml_core.config.providers]
                   registry. Defaults to 'yfinance'.
            
            custom_provider (Dict[str, Any], optional):
                Custom coordinate map. If provided, ignores `data_source`. 
                Must contain `col_map`, `ticker_level`, and `date_level`.

                Example:

                ```python
                {
                    'col_map': {
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'
                    },
                    'ticker_level': 'Ticker',
                    'date_level': 'Date'
                }
                ```
                which is 'yfinance' mapping.
            
            feature_config (Dict[str, Dict[str, Any]], optional):
                Technical indicator parameters (e.g., `rsi`, `bollinger`). 
                If None, uses 
                [ `compute_indicators` ][finml_core.processing.features.FeatureGenerator.compute_indicators] defaults
                from `FeatureGenerator` class.
            
            feature_selection (Dict[str, List[int]], optional):
                The final 'filter' for the matrix. Defines which metrics and 
                how many lags to include in $X$. If None, uses 
                [ `construct_feature_matrix` ][finml_core.processing.features.FeatureGenerator.construct_feature_matrix] defaults
                from `FeatureGenerator` class.
            
            label_config (Dict[str, Any], optional):
                Hyperparameters for the Triple Barrier Method. If None, uses
                [ `compute_outcomes` ][finml_core.processing.labeling.TripleBarrierLabeling.compute_outcomes] defaults
                from `TripleBarrierLabeling` class.

        ---
        """
        if custom_provider:
            provider = custom_provider

            self.custom_user_input = True
        else:
            # PROVIDER'S INFO
            provider = PROVIDERS.get(data_source)

            if not provider:
                raise KeyError(
                    f"Provider '{data_source}' not found."
                    "Provide 'custom_provider' dict or select a supported"
                    f"provider: {PROVIDERS.keys()}"
                )
            
            self.custom_user_input = False
        
        # Required parameters for initializing classes
        col_map = provider.get('col_map')
        ticker_level = provider.get('ticker_level')
        date_level = provider.get('date_level')

        # Saving col_map to validate columns are mapping to dataset
        self.col_map = col_map

        # Feature selection
        self.feature_selection = feature_selection
        
        # Initialize components
        if self.custom_user_input == False:
            self.loader = None
        else:
            self.loader = MarketDataLoader(source=data_source)

        self.feature_gen = FeatureGenerator(
            col_map=col_map,
            ticker_level=ticker_level,
            config=feature_config
        )
        
        self.labeler = TripleBarrierLabeling(
            col_map=col_map,
            ticker_level=ticker_level,
            date_level=date_level,
            config=label_config
        )

        self.cleaner = DataCleaner(
            ticker_level=ticker_level,
            date_level=date_level,
            method='ffill'
        )

        self.analysis_data = None

    def run(
        self,
        tickers: Optional[List[str]] = None,
        etf_reference: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        df_input: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the modeling pipeline to produce model-ready matrices.

        This method orchestrates the full data lifecycle. It follows a strict 
        **Priority Logic**:
        
        1. If `custom_provider` was used at `__init__`, it **requires** `df_input`.
        2. If `data_source` was used, it attempts to download data using 
            `tickers`, `etf_reference`, and `start_date`.

        !!! danger "Data Structure Requirement (MultiIndex)"
            Whether injected via `df_input` or downloaded, the internal engine 
            **requires** a MultiIndex DataFrame with levels corresponding to 
            `ticker_level` and `date_level`. 
            
            Standard format: `[Date (DatetimeIndex), Ticker (str)]`.

        Args:
            tickers (List[str], optional): 
                Asset symbols to fetch. Required if `df_input` is None and
                `custom_provider` was specified.
            etf_reference (str, optional): 
                Reference symbol (e.g., '^GSPC') to align market holidays.
            start_date (str, optional): 
                Start of the historical period (YYYY-MM-DD hh:mm:ss).
            end_date (str, optional): 
                End of the period. Defaults to current date.
            df_input (pd.DataFrame, optional): 
                Pre-loaded MultiIndex DataFrame.

        Returns:
            (Tuple):
                - **X** (pd.DataFrame): Final Feature Matrix with selected 
                    metrics and lags.
                - **y** (pd.DataFrame): Target labels (target_side) aligned 
                    with $X$.

        !!! info "Index Alignment & Trimming:"
            The pipeline performs a bidirectional trim 
                to ensure a clean, MultiIndex-aligned dataset. It removes the 
                **Warm-up Period** at the beginning (caused by rolling metrics like 
                volatility and indicators) and the **Horizon Gap** at the end (due to 
                the forward-looking nature of triple-barrier labels). This ensures 
                that $X$ and $y$ contain only fully realized, non-null observations 
                ready for machine learning.
        """
        # priotitizing custom provider if user provided it
        if self.custom_user_input == True:
            if df_input is None:
                raise KeyError(
                    "Provided 'custom_provider' parameter when instantiating "
                    "class, must provide 'df_input'"
                )
            
            # Validating if users column mappings are in the df columns
            validate_df_columns(df_input, self.col_map)

            df_raw = df_input.copy()
        
        else:
            print(f"--- 1. Starting Ingestion ({len(tickers)} tickers) ---")
            # Get raw data from the specified API
            df_raw = self.loader.get_financial_data(
                tickers=tickers,
                etf=etf_reference,
                start=start_date, 
                end=end_date
            )

        # Validate not inf's and not nan's
        df_raw_clean = self.cleaner.validate_and_clean(df_raw)


        print("--- 2. Generating Features (X) ---")
        # Calculate financial metrics
        df_indicators = self.feature_gen.compute_indicators(df_raw_clean)
        
        # Select features from financial metrics and lags
        X = self.feature_gen.construct_feature_matrix(
            df_indicators,
            selection=self.feature_selection
        )

        print("--- 3. Generating Labels (y) ---")
        # Label our data using the triple-barrier labeling method
        triple_barrier_outcomes = self.labeler.compute_outcomes(df_raw_clean)

        print("--- 4. Consolidating & Cleaning ---")

        model_data = pd.concat([
            X, # Model Input (Features)
            triple_barrier_outcomes[['target_side']] # Model Output (Labels)
        ], axis=1)

        # Model Data Cleaned Valid Idx's
        valid_index = self.cleaner.validate_and_clean(model_data).index

        # Final feature matrix
        X_clean = X.loc[valid_index].copy()

        # Final label serie
        y_clean = triple_barrier_outcomes.loc[valid_index, 'target_side'].copy()

        # Full context data
        analysis_data_clean = pd.concat([
            df_indicators, # OHLCV & Indicators
            triple_barrier_outcomes # Triple-barrier outcomes
        ], axis=1)

        self.analysis_data = analysis_data_clean.loc[valid_index].copy()

        print(f"--- Pipeline Finished. Final Dataset: {X_clean.shape[0]} rows ---")
        return X_clean, y_clean