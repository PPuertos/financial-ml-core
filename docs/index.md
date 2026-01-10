# 🏛️ Financial ML Core: The Foundation

> "To push innovation forward, one must first master the foundations that got us here."

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/){:target="_blank"}
[![Scikit-Learn Compatible](https://img.shields.io/badge/sklearn-compatible-orange.svg)](https://scikit-learn.org/){:target="_blank"}
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT){:target="_blank"}
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/PPuertos/financial-ml-core)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-informational?logo=linkedin)](https://www.linkedin.com/in/francisco-puertos-rumayor/){:target="_blank"}

---

**`finml-core`** is an institutional-grade framework engineered from first principles to address the inherent uncertainty and structural complexity of financial time series. This library was born from the necessity to move beyond "black-box" API calls, providing a transparent and mathematically rigorous system that respects the unique statistical nature of market data.

Rather than offering a collection of standard tools, **`finml-core`** provides a governed pipeline designed to prevent the most common pitfalls in Quantitative Finance—such as look-ahead bias, information leakage, and the misapplication of standard cross-validation to non-IID data.

`finml-core` is a high-performance Python library designed for quantitative researchers and financial engineers. It implements rigorous methodologies proposed by Marcos López de Prado in *"Advances in Financial Machine Learning"*, ensuring that ML models for trading are free from common pitfalls like look-ahead bias and serial correlation leakage.

---

## 📐 The Philosophy of Financial Data Geometry

In Quantitative Finance, the data structure is not just a container, it is a **Coordinate System**. Most generic machine learning frameworks fail because they treat financial series as flat tables, ignoring the relationship between time and specific assets. 

**`finml-core`** enforces a strict 3-dimensional perspective—**Time × Asset × Feature**—through its mandatory **MultiIndex Architecture**.

### Why [Date, Ticker] is Non-Negotiable
Financial time series are characterized by cross-sectional dependence and heterogeneous scales. By enforcing a `pd.MultiIndex` with `Date` and `Ticker` levels as the primary index, the framework achieves three critical objectives:

1. **Group-Wise Vectorization**: Statistical operations (like volatility estimation or RSI) are calculated *per asset* in a vectorized manner. This ensures that a high-volatility ticker (like TSLA) never influences the normalization or feature scaling of a low-volatility one (like KO).
2. **Temporal Alignment**: It ensures that during the Labeling Process, the "Vertical Barrier" (time-limit) is synchronized across the entire universe, even when dealing with assets that have different trading lifespans or missing bars.
3. **Prevention of Cross-Asset Leakage**: By maintaining the asset identity at the index level, the Purging and Embargo mechanisms can precisely identify which specific observations must be removed without destroying the integrity of the rest of the universe.

```python
# The finml-core Standard for Data Integrity
# High-level example of index setting:
df.index = pd.MultiIndex.from_tuples(tuples, names=['Date', 'Ticker'])
```

---

## ⚙️ Core Engines & Architectural Modules

The framework is organized into specialized engines, each designed to solve a specific challenge in the Financial ML pipeline. While they can be used independently, they are seamlessly orchestrated by the **DatasetGenerator**.

### 1. **The Data Factory Architecture** [`DatasetGenerator`](reference/pipelines/data_factory.md)

The central nervous system of the library, abstracting the complexity of the entire pipeline into a unified interface.

* **Dual-Mode Operation**: Supports **Provider Mode** for automated fetching via integrated APIs (e.g., `yfinance`) and **Custom Mode** for ingesting user-provided DataFrames.
* **Validation & Integrity**: Acts as a final gatekeeper, validating **MultiIndex** geometry and ensuring that features and labels are perfectly synchronized in time before model ingestion.

### 2. **Market Data Loader** [`MarketDataLoader`](reference/data/loader.md)
The abstraction layer for rigorous data acquisition.

* **Unified Schema**: Translates heterogeneous data from various providers into a consistent internal format, eliminating "Garbage In, Garbage Out" at the source.
* **Market Context**: Integrated with market calendars to handle session gaps and holidays, ensuring the dataset reflects actual trading reality.

### 3. The Sanitization Engine [`DataCleaner`](reference/processing/cleaning.md)
Financial data is noisy and prone to leakage. This engine enforces a **Strict Sanitization Protocol**.

* **Bidirectional Trimming**: Automatically manages the "warm-up" period required for technical indicators and the "look-ahead" horizon required for future labels.
* **Arrow of Time Integrity**: Implements conservative forward-filling to handle liquidity gaps without introducing future information.

### 4. Feature Engineering [`FeatureGenerator`](reference/processing/features.md)
Moves beyond simple technical analysis to create **Statistically Sound Features**.

* **Multi-Lag Expansion**: Automatically generates temporal sequences (lags) for every feature to capture market memory and autocorrelation.
* **Relative Intelligence**: Enables features to be calculated relative to a benchmark (e.g., an ETF), isolating idiosyncratic asset performance from broader market noise.

### 5. Structural Labeling [`TripleBarrierLabeling`](reference/processing/labeling.md)
A path-dependent labeling engine that respects the reality of institutional risk management.

* **Dynamic Boundaries**: Uses realized volatility to set adaptive take-profit and stop-loss barriers.
* **Holding Constraints**: Implements vertical barriers (time-limits) to account for the cost of capital and opportunity costs.

### 6. Model Selection [`PurgedKFold`](reference/model_selection/split.md)
Our safeguard against the "Overfitting Trap" in non-IID financial data.

* **Leakage Prevention**: Implements **Purging** to remove training observations whose labels overlap with the test set's timeframe.
* **Memory Embargo**: Applies a **Quarantine Zone** (Embargo) after the test set to neutralize the effects of slow-decaying serial correlation.

---

### 🚀 Getting Started

To ensure a clean and reproducible environment, follow these steps:

#### 1. Setup Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

#### 2. Installation
First, install the framework directly from the source. This will automatically trigger the installation of core dependencies defined in `pyproject.toml`:

```bash
# Intall financial-ml-core library
# Windows
python -m pip install git+https://github.com/PPuertos/financial-ml-core.git
# MacOS/Linux
python3 -m pip install git+https://github.com/PPuertos/financial-ml-core.git
```

<details>
<summary><b>🛠️ Are you a developer or planning to collaborate?</b></summary>
<br>
If you want to contribute to the project, run tests, or modify the source code, install the full development requirements to ensure you have all necessary tooling (like pytest and setuptools):
<br><br>

```bash
# 1. Clone the repository
git clone https://github.com/PPuertos/financial-ml-core.git
cd financial-ml-core

# 2. Install in editable mode with development dependencies
# Windows
python -m pip install -e ".[dev]"

# MacOS/Linux
python3 -m pip install -e ".[dev]"
```
</details>

---

## 🧪 Usage Example: Basic Financial ML Pipeline
The following example demonstrates the complete workflow: from automated
Multi-Asset Ingestion and Feature Engineering (including temporal lags) to
Triple Barrier Labeling and rigorous Sanitization. The result is a
production-ready dataset, free from look-ahead bias and numerical instability.

This library provides the essential Purged Cross-Validation engines required
to tune any Scikit-Learn model without falling into the "overfitting trap"
caused by serial correlation, ensuring that your model's performance is
statistically grounded.

### 1. Orchestrating the Data Factory
The `DatasetGenerator` is the core orchestrator that handles ingestion,
sanitization, feature engineering, and labeling in a single synchronized flow.

#### Automated Ingestion
In its simplest form, you only need to specify your tickers and the data source
(from supported [`PROVIDERS`](reference/config/providers.md)).
The framework will use institutional-grade defaults for feature engineering and labeling.

```python
# --- AUTOMATED MODE ---
from finml_core.pipelines.data_factory import DatasetGenerator

# 1. Input `data_source` parameter refering to the provider
generator = DatasetGenerator(data_source='yfinance')

# 2. Execute the pipeline
# Automatically handles Ingestion, Sanitization, Features, and Labeling
X, y = generator.run(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    etf_reference='^GSPC',
    start_date='2018-01-01'
)
```

<details> <summary><b>📂 Advanced: Using your own Custom Dataset</b></summary>

If you already have a dataset (e.g., from a local CSV or database), you can
inject it directly. The only requirement is that your DataFrame must follow the
<b>MultiIndex [Date, Ticker]</b> geometry.

```python
# --- CUSTOM MODE ---
from finml_core.pipelines.data_factory import DatasetGenerator

# Assume 'my_df' is a MultiIndex (Date, Ticker) DataFrame with OHLCV columns

# Specifying 'my_df' dataset structure
custom_provider = {
    'col_map': {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    },
    'ticker_level': 'Ticker',
    'date_level': 'Date'
}

generator = DatasetGenerator(
    custom_provider=custom_provider
)

# Pass your DataFrame directly to the run method
X, y = generator.run(df_input=my_df)
```
</details>

### 2. Statistical Split (Purging & Embargo)
Standard random splits or naive chronological splits are insufficient in finance.
Because our labels (via TBM) have a time duration (from $t_0$ to $t_1$),
training and testing observations can overlap, causing **information leakage**. 

We use [`purged_train_test_split`](reference/model_selection/split.md) to
execute a rigorous separation:

* **Purging**: Removes training samples whose evaluation period overlaps with the test set.
* **Embargo**: Adds a "quarantine" buffer immediately after the test set to neutralize any residual serial correlation (market memory).

```python
from finml_core.model_selection.split import purged_train_test_split

# Extracting t1 (label end dates for Purged CV)
analysis_data = generator.analysis_data
t1 = analysis_data['t1']

# Split Data into Train & Test (Purged & Embargoed)
X_train, y_train, X_test, y_test = purged_train_test_split(
    X, y, t1, 
    date_level='Date',
    test_size=.2
)

print(f"\nTrain size: {X_train.shape} | Test size: {X_test.shape}")
print(f"Purging + Embargo dates = {X.shape[0] - X_train.shape[0] - X_test.shape[0]}")
```

### 3. Leakage-Free Validation (Purged K-Fold)
Standard Cross-Validation assumes that observations are independent and
identically distributed (I.I.D.). In finance, this assumption is false.
To evaluate a model or prepare for hyperparameter tuning without "lying" to
ourselves, we implement `PurgedKFold`. 

This cross-validator acts as a drop-in replacement for Scikit-Learn's K-Fold,
ensuring that every training fold is mathematically isolated from the validation
set by respecting the temporal overlap of labels.

```python
from finml_core.model_selection.split import PurgedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# 1. Create the Purged Cross-Validator using the training 't1' dates.
# This ensures that each fold is separated by purging and an embargo period.
purged_cv = PurgedKFold(
    n_splits=5,
    t1=analysis_data.loc[X_train.index, 't1'],
    date_level='Date',
    pct_embargo=0.01
)

# 2. Define your model (e.g., RandomForest)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)

# 3. Compute scores using the Purged CV engine.
# This setup is also compatible with sklearn.model_selection.GridSearchCV.
scores = cross_val_score(
    rf_model, 
    X_train, 
    y_train, 
    cv=purged_cv.split(X_train, y_train), 
    scoring='accuracy',
    n_jobs=-1
)

print(f"Purged CV Accuracy: {np.mean(scores):.2%} (+/- {np.std(scores):.4f})")
```

---
## ⚖️ License & Ethical Note

This project is licensed under the MIT License. **`finml-core`** is an open-source contribution to the quantitative finance community. It is designed for research and educational purposes. Always remember: *Backtesting is not forecasting, and past performance does not guarantee future results.*

---

## 🏛️ Why `finml-core`?

* **Integrity First**: Every line of code is written to prevent data leakage, the #1 killer of financial ML models.
* **Built for Production**: Unlike fragmented notebooks, this is a structured library with modular engines.
* **Theoretical Rigor**: We don't just "fit models"; we follow the path-dependent nature of financial markets as defined by the industry's leading researchers.

---

## 📚 References
* López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

---

<div align="center" style="margin-top: 50px;">
  <p style="margin-bottom: 15px; font-weight: bold; color: #555; font-family: sans-serif;">
    Developed with passion and mathematical rigor by
  </p>
  
  <a href="https://www.linkedin.com/in/francisco-puertos-rumayor/" 
     target="_blank" 
     style="display: inline-flex; 
            align-items: center; 
            background-color: #0077b5; 
            color: white; 
            padding: 10px 20px; 
            text-decoration: none; 
            border-radius: 4px; 
            font-family: -apple-system, system-ui, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', 'Fira Sans', Ubuntu, Oxygen, 'Oxygen Sans', Cantarell, 'Droid Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Lucida Grande', Helvetica, Arial, sans-serif; 
            font-weight: 600;
            font-size: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            transition: background-color 0.2s ease, box-shadow 0.2s ease;">
    
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="white" style="margin-right: 10px;">
      <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-1.337-.025-3.062-1.865-3.062-1.867 0-2.153 1.459-2.153 2.966v5.7h-3v-11h2.88v1.503h.04c.401-.76 1.381-1.56 2.841-1.56 3.039 0 3.601 2.001 3.601 4.602v5.455z"/>
    </svg>
    
    Francisco Puertos Rumayor
  </a>
</div>

---