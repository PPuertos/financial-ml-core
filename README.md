# 🏛️ Financial ML Core: The Foundation

> "To push innovation forward, one must first master the foundations that got us here."

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn Compatible](https://img.shields.io/badge/sklearn-compatible-orange.svg)](https://scikit-learn.org/)
[![Documentation](https://img.shields.io/badge/docs-MkDocs-blue.svg)](https://ppuertos.github.io/financial-ml-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-informational?logo=linkedin)](https://www.linkedin.com/in/francisco-puertos-rumayor/)

---
**`finml-core`** is not just a tool for building models, it is a framework built from first principles to handle the inherent "uncertainty" of financial data. This library was born from the necessity to understand the mechanics of high-level quantitative finance, moving beyond simple API calls to engineer a system that respects the statistical nature of market series.

## 🔬 First-Principles Engineering

In Quantitative Finance, the "how" is irrelevant if the "why" is flawed. Most failures in Financial Machine Learning stem from a lack of respect for the data's structure—serial correlation, non-stationarity, and look-ahead bias. 

This library addresses these gaps by focusing on three architectural pillars:

### 1. The Multi-Asset Coordinate System
Financial data is 3-dimensional (Time, Asset, Feature). Instead of flattening this complexity, **`finml-core`** uses a strict **MultiIndex (Date, Ticker)** architecture. This ensures that every operation—from technical indicators to data cleaning—is aware of its asset-specific context, preventing cross-asset data leakage.

### 2. Automated Ingestion & Standardization
Before processing, data must be governed. Our **Market Data Loader** acts as a unified abstraction layer. By leveraging a centralized **Providers Registry**, it translates heterogeneous external APIs into a consistent internal schema. This eliminates the "Garbage In, Garbage Out" problem by enforcing alignment between tickers and market calendars from the very first step.

### 3. The Sanitization Protocol
Data gaps in finance are not just "missing values", they are signals of market closures or liquidity issues. Our `DataCleaner` doesn't just fill holes, it executes a **Strict Sanitization Protocol**:
* **Bidirectional Trimming**: Managing warm-up periods (for indicators) and horizon gaps (for labels) to ensure every training row is statistically valid.
* **Conservative Filling**: Forward-filling with strict limits to respect the arrow of time and avoid look-ahead bias.

### 4. Structural Labeling
We implement the **Triple Barrier Method**, recognizing that markets don't just move up or down—they move against time and volatility. By dynamically adjusting barriers to realized volatility, we transform raw price action into labels that capture the true economic reality of a trade.

### 5. Statistical Integrity in Validation
Standard Cross-Validation fails in finance due to the overlapping nature of labels and serial correlation. To address this, we implement **Purged K-Fold Cross-Validation**. This pillar ensures that training and testing sets are separated by "Purging" and "Embargo" periods, preventing information leakage and ensuring that our model's performance is not a mere artifact of data overlap.

---

## 🏗️ The Data Factory Architecture

The library is structured as a pipeline of specialized engines, all orchestrated by the `DatasetGenerator`. This allows for a modular but integrated workflow:

* **Ingestion Engine**: Standardizes heterogeneous sources into a unified internal format.
* **Processing Engine**: Handles the mathematical heavy lifting of feature engineering.
* **Quality Gate**: A final check for numerical stability (Infs and NaNs) before model ingestion.

### 🚀 Getting Started

To ensure a clean and reproducible environment, follow these steps:

#### 1. Setup Environment
```bash
# Create a virtual environment
# Windows
python -m venv venv
# MacOS/Linux
python3 -m venv venv

# Activate it
# Windows
venv\Scripts\activae
# MacOS/Linux
source venv/bin/Activate
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

#### 🛠️ Basic Usage
The `DatasetGenerator` orchestrates the entire pipeline. Transform raw tickers into a production-ready dataset with a single call:

```python
# --- AUTOMATED MODE ---
from finml_core.pipelines.data_factory import DatasetGenerator
from finml_core.model_selection.split import purged_train_test_split

# 1. Input `data_source` parameter refering to the provider
generator = DatasetGenerator(data_source='yfinance')

# 2. Execute the pipeline
# Automatically handles Ingestion, Sanitization, Features, and Labeling
X, y = generator.run(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    etf_reference='^GSPC',
    start_date='2018-01-01'
)

# 3. Extracting t1 (label end dates for Purged CV)
analysis_data = generator.analysis_data
t1 = analysis_data['t1']

# 4. Split Data into Train & Test (Purged & Embargoed)
X_train, y_train, X_test, y_test = purged_train_test_split(
    X, y, t1, 
    date_level='Date',
    test_size=.2
)

print(f"\nTrain size: {X_train.shape} | Test size: {X_test.shape}")
print(f"Purging + Embargo dates = {X.shape[0] - X_train.shape[0] - X_test.shape[0]}")
```
---

## 🎯 Intended Use

This framework is designed for researchers and engineers who challenge the status quo. It is built for those who value **statistical integrity** over ease of use, providing a transparent and rigorous path from raw uncertainty to strategic value.

---

## 🛠️ Roadmap & Future Developments

The journey doesn't end here. Guided by first principles, the following modules are currently under development to further enhance the library's analytical power:

1. **Bar Sampling**: Moving beyond time-based sampling to implement Volume and Dollar Bars, aiming for IID (Independent and Identically Distributed) properties and near-Normal distributions.
2. **Feature Importance**: Specialized methods for financial series, accounting for substitution effects and non-linearity.
3. **Hyperparameter Tuning**: Integrated with Purged CV to optimize models without falling into the "overfitting trap."
4. **Backtesting Engine**: A rigorous framework for strategy evaluation, strictly following López de Prado's methodologies to avoid backtest overfitting.
5. **Unit Testing Framework**: Ensuring the mathematical precision of every indicator and labeling protocol.