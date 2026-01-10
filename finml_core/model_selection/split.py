# External Libraries
import pandas as pd
import numpy as np
from typing import Tuple, Generator

# External Machine Learning Library
from sklearn.model_selection import KFold, train_test_split

# --- PRIVATE UTILS ---

def _purge_training_multi_indexes(
        training_t1: pd.Series,
        test_intervals: Tuple[pd.Timestamp, pd.Timestamp],
        date_level: str
) -> pd.MultiIndex:
    """
    Core Purging Logic.

    Identifies overlap using the geometric condition:

    Overlap exists if: (Train_Start <= Test_End) & (Train_End >= Test_Start)

    Args:
        training_t1 (pd.Series): Training set label spans.
        test_intervals (Tuple): (Test_Start, Test_End_with_Embargo).

    Returns:
        pd.MultiIndex: The subset of training indices that DO NOT overlap.
    """
    testing_start_date = test_intervals[0]
    testing_end_date = test_intervals[1]

    # Training Entry (Index)
    training_entry = training_t1.index.get_level_values(date_level)
    # Training Exit (Values)
    training_exit = training_t1

    # Logic: Drop if Training Interval overlaps with Test Interval
    overlap_mask = (
        # Train starts before test ends
        (training_entry <= testing_end_date) &
        # Train ends after test starts
        (training_exit >= testing_start_date)
    )

    # Return only the indices that are NOT overlaping
    return training_t1.loc[~overlap_mask].index

# --- PUBLIC INTERFACE ---

def _apply_embargo(
    unique_dates: pd.DatetimeIndex,
    test_t1: pd.Series,
    pct_embargo: float,
    date_level: str
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Calculates the embargo timeframe.

    Embargo is calculated as a % of the TOTAL dataset duration, 
    added to the end of the test set.

    Returns:
        Tuple: (Test_Start_Date, Test_End_Date + Embargo_Delta)
    """
    # Calculate the global time span
    global_start = min(unique_dates)
    global_end = max(unique_dates)
    total_duration = global_end - global_start

    # Calculate the global embargo delta
    embargo_h = total_duration * pct_embargo

    # Test entry dates
    test_entry_dates = test_t1.index.get_level_values(date_level)
    # Test exit dates
    test_exit_dates = test_t1

    # Identify test boundaries
    test_start = min(test_entry_dates)
    test_end = max(test_exit_dates)

    # Apply embargo
    target_embargo_date = test_end + embargo_h

    # Logical safety
    if target_embargo_date > global_end:
        target_embargo_date = global_end

    return (test_start, target_embargo_date)


class PurgedKFold():
    r"""
    Cross-Validation with Purging and Embargo for Financial Time Series.
    
    Implements the methodology proposed by Marcos López de Prado in 
    "Advances in Financial Machine Learning" (Chapter 7). Standard K-Fold 
    CV assumes IID data, which is false in finance due to overlapping labels 
    and serial correlation.

    This class prevents data leakage through two mechanisms:
        1. **Purging**: Removes training observations whose labels overlap in time 
        with the test set.
        2. **Embargo**: Eliminates a period immediately following the test set 
        to handle auto-correlated residuals.

    Mathematical Overlap Condition:
        A training observation is purged if its interval $[t_{i,0}, t_{i,1}]$ 
        overlaps with the test interval $[T_{j,0}, T_{j,1}]$:
        $$ (t_{i,0} \le T_{j,1}) \land (t_{i,1} \ge T_{j,0}) $$
    """
    def __init__(
            self,
            n_splits: int,
            t1: pd.Series,
            date_level: str,
            pct_embargo: float = 0.01
    ):
        """
        Attributes:
            n_splits (int): Number of folds.
            t1 (pd.Series): End timestamps of labels (Vertical Barriers).
            date_level (str): MultiIndex level name for timestamps.
            pct_embargo (float): Percentage of total timeframe for the embargo.

        Note:
            Sklearn Compatibility: Works as a drop-in replacement for KFold in 
            GridSearchCV and cross_val_score.

        ---
        """
        self.n_splits = n_splits
        self.t1 = t1
        self.date_level = date_level
        self.pct_embargo = pct_embargo

        self.purging_t1 = []
        self.embargo_t1 = []

    def split(
        self, X: pd.DataFrame, y: pd.Series, groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates indices to split data into training and test sets.

        Process:
            1. **Temporal Grouping**: Groups data by unique dates to maintain 
            chronological integrity.
            2. **Interval Analysis**: Evaluates the `t1` (label span) of each 
            observation.
            3. **Purge & Embargo**: Drops training indices that fail the 
            non-overlap condition against the current test fold.

        Args:
            X (pd.DataFrame): Features dataset with MultiIndex.
            y (pd.Series, optional): Target variable. Defaults to None.
            groups (): Compatibility placeholder.

        Yields:
            (np.ndarray): Integer indices for the training set.
            (np.ndarray): Integer indices for the test set.
        """
        # We work on a copy to avoid side effects
        t1 = self.t1.copy()
        
        # Extract unique dates to split by time
        unique_dates = X.index.get_level_values(
             self.date_level
        ).unique().sort_values()

        kf = KFold(n_splits=self.n_splits, shuffle=False)

        # KFold returns indices of the 'unique_dates' array
        split_generator = kf.split(unique_dates)

        for train_idx, test_idx in split_generator:
            # Resolve Dates from indices
            train_dates = unique_dates[train_idx]
            test_dates = unique_dates[test_idx]

            # Create masks based on the date level
            training_mask = X.index.get_level_values(
                 self.date_level
            ).isin(train_dates)

            testing_mask = X.index.get_level_values(
                 self.date_level
            ).isin(test_dates)

            # Susbets t1 (label spans) for current splits
            training_t1 = t1.loc[training_mask]
            testing_t1 = t1.loc[testing_mask]

            # Define the extended test interval (Test Range + Embargo)
            test_intervals = _apply_embargo(
                unique_dates,
                testing_t1,
                self.pct_embargo,
                self.date_level
            )

            # Purge: Identify training samples that overlap with Test+Embargo
            purged_training_idx = _purge_training_multi_indexes(
                training_t1,
                test_intervals,
                self.date_level
            )

            # purging+embargo exit dates 
            overlap_t1 = training_t1.loc[
                 ~training_t1.index.isin(purged_training_idx)
            ]
            
            # last test exit date
            test_end_date = max(testing_t1)

            # purging mask
            purging_mask = (
                 overlap_t1.index.get_level_values(self.date_level) <=
                 test_end_date
            )

            # purging exit dates
            self.purging_t1.append(overlap_t1.loc[purging_mask])

            # embargo mask
            embargo_mask = (
                 overlap_t1.index.get_level_values(self.date_level) >
                 test_end_date
            )

            # embargo exit dates
            self.embargo_t1.append(overlap_t1.loc[embargo_mask])

            # Map valid MultiIndex back to Integer Indices (for Sklearn)
            # We create a boolean mask of the whole X
            # Set True only for valid rows
            purged_training_mask = X.index.isin(purged_training_idx)

            # Return integer positions
            yield np.where(purged_training_mask)[0], np.where(testing_mask)[0]
    
def purged_train_test_split(
          X: pd.DataFrame,
          y: pd.Series,
          t1: pd.Series,
          date_level: str,
          test_size: float = 0.2,
          pct_embargo: float = 0.01
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Chronological split with financial purging and embargo.

    Performs a non-shuffled temporal split (Past -> Train, Future -> Test) 
    and applies a quarantine period to ensure the test set is strictly 
    independent of the training data.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target labels.
        t1 (pd.Series): Timestamps of label ends (Vertical Barriers).
        date_level (str): MultiIndex level name for timestamps.
        test_size (float): Proportion of data for testing. Defaults to 0.2.
        pct_embargo (float): Total timeline percentage for embargo. 
                             Defaults to 0.01 (1%).

    Returns:
        (Tuple):
            - **X_train** (pd.DataFrame): Purged features.
            - **y_train** (pd.Series): Purged labels.
            - **X_test** (pd.DataFrame): Test features.
            - **y_test** (pd.Series): Test labels.
    """
    # Extract unique dates to split by time
    unique_dates = X.index.get_level_values(
            date_level
    ).unique().sort_values()

    # spliting into train and test arrays
    train_dates, test_dates = train_test_split(
        unique_dates,
        test_size=test_size,
        shuffle=False
    )

    # Create masks based on the date level
    training_mask = X.index.get_level_values(
        date_level
    ).isin(train_dates)

    testing_mask = X.index.get_level_values(
        date_level
    ).isin(test_dates)

    # Susbets t1 (label spans) for current splits
    training_t1 = t1.loc[training_mask]
    testing_t1 = t1.loc[testing_mask]

    # Define the extended test interval (Test Range + Embargo)
    test_intervals = _apply_embargo(
        unique_dates,
        testing_t1,
        pct_embargo,
        date_level
    )

    # Purge: Identify training samples that overlap with Test+Embargo
    purged_training_idx = _purge_training_multi_indexes(
        training_t1,
        test_intervals,
        date_level
    )
    
    X_train, y_train, X_test, y_test = (
        X.loc[purged_training_idx], y.loc[purged_training_idx],
        X.loc[testing_mask], y.loc[testing_mask]
    )

    # Return train and test subsets
    return X_train, y_train, X_test, y_test