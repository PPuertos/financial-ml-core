# 🏛️ Financial ML Core: The Foundation

> "To push innovation forward, one must first master the foundations that got us here."

[![Documentation](https://img.shields.io/badge/docs-MkDocs-blue.svg)](https://ppuertos.github.io/financial-ml-core/docs)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-informational?logo=linkedin)](https://www.linkedin.com/in/francisco-puertos-rumayor/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn Compatible](https://img.shields.io/badge/sklearn-compatible-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
**FinML-Core** is not just a tool for building models; it is a framework built from first principles to handle the inherent "uncertainty" of financial data. This library was born from the necessity to understand the mechanics of high-level quantitative finance, moving beyond simple API calls to engineer a system that respects the statistical nature of market series.

> 📘 **Full Documentation**: For detailed API references, architectural deep dives, and tutorials, visit our [Documentation Site](https://ppuertos.github.io/financial-ml-core/docs).

## 🔬 First-Principles Engineering

In Quantitative Finance, the "how" is irrelevant if the "why" is flawed. Most failures in Financial Machine Learning stem from a lack of respect for the data's structure—serial correlation, non-stationarity, and look-ahead bias. 

This library addresses these gaps by focusing on three architectural pillars:

### 1. The Multi-Asset Coordinate System
Financial data is 3-dimensional (Time, Asset, Feature). Instead of flattening this complexity, **FinML-Core** uses a strict **MultiIndex (Date, Ticker)** architecture. This ensures that every operation—from technical indicators to data cleaning—is aware of its asset-specific context, preventing cross-asset data leakage.

### 2. Automated Ingestion & Standardization
Before processing, data must be governed. Our **Market Data Loader** acts as a unified abstraction layer. By leveraging a centralized **Providers Registry**, it translates heterogeneous external APIs into a consistent internal schema. This eliminates the "Garbage In, Garbage Out" problem by enforcing alignment between tickers and market calendars from the very first step.

### 3. The Sanitization Protocol
Data gaps in finance are not just "missing values"; they are signals of market closures or liquidity issues. Our `DataCleaner` doesn't just fill holes; it executes a **Strict Sanitization Protocol**:
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
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

#### 2. Installation
First, install the framework directly from the source. This will automatically trigger the installation of core dependencies defined in `pyproject.toml`:

```bash
pip install git+[https://github.com/PPuertos/financial-ml-core.git](https://github.com/PPuertos/financial-ml-core.git)
```

<details>
<summary><b>🛠️ Are you a developer or planning to collaborate?</b></summary>
<br>
If you want to contribute to the project, run tests, or modify the source code, install the full development requirements to ensure you have all necessary tooling (like pytest and setuptools):
<br><br>

```bash
# Install additional tools for development and testing
pip install -r requirements.txt
```
</details>

#### 🛠️ Basic Usage
The `DatasetGenerator` orchestrates the entire pipeline. Transform raw tickers into a production-ready dataset with a single call:

```python
from finml_core.pipelines.data_factory import DatasetGenerator
from finml_core.model_selection.split import purged_train_test_split

# 1. Define your universe and parameters
generator = DatasetGenerator(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    etf='^GSPC',                
    start='2020-01-01',       
    end='2023-12-31'
)

# 2. Execute the pipeline
# Automatically handles Ingestion, Sanitization, Features, and Labeling
X, y = generator.create_dataset()

# 3. Extracting t1 (label end dates for Purged CV)
analysis_data = generator.analysis_data
t1 = analysis_data['t1']

# 4. Split Data into Train & Test (Purged & Embargoed)
X_train, y_train, X_test, y_test = purged_train_test_split(
    X, y, t1, 
    date_column='Date'
)

print(f"Train size: {X_train.shape} | Test size: {X_test.shape}")
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

---

## 📞 Connect & Contribute

This project is a continuous evolution of my research in Algorithmic Trading and Risk Analytics. If you share this passion for first-principles thinking:

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 12px; margin: 40px 0;">

  <a href="https://www.linkedin.com/in/francisco-puertos-rumayor/" class="md-button" style="display: inline-flex; align-items: center; gap: 8px; margin: 0;">
    <svg style="width: 20px; fill: currentColor;" viewBox="0 0 448 512"><path d="M416 32H31.9C14.3 32 0 46.5 0 64.3v383.4C0 465.5 14.3 480 31.9 480H416c17.6 0 32-14.5 32-32.3V64.3c0-17.8-14.4-32.3-32-32.3zM135.4 416H69V202.2h66.5V416zm-33.2-243c-21.3 0-38.5-17.3-38.5-38.5S80.9 96 102.2 96c21.2 0 38.5-17.3 38.5 38.5 0 21.3-17.2 38.5-38.5 38.5zm282.1 243h-66.4V312c0-24.8-.5-56.7-34.5-56.7-34.6 0-39.9 27-39.9 54.9V416h-66.4V202.2h63.7v29.2h.9c8.9-16.8 30.6-34.5 62.9-34.5 67.2 0 79.7 44.3 79.7 101.9V416z"></path></svg>
    LinkedIn
  </a>

  <a href="mailto:paco_puertos11@hotmail.com" class="md-button" style="display: inline-flex; align-items: center; gap: 8px; margin: 0;">
    <svg style="width: 20px; fill: currentColor;" viewBox="0 0 512 512"><path d="M48 64C21.5 64 0 85.5 0 112c0 15.1 7.1 29.3 19.2 38.4L236.8 313.6c11.4 8.5 27 8.5 38.4 0L492.8 150.4c12.1-9.1 19.2-23.3 19.2-38.4c0-26.5-21.5-48-48-48H48zM0 176V384c0 35.3 28.7 64 64 64H448c35.3 0 64-28.7 64-64V176L294.4 339.2c-22.8 17.1-54 17.1-76.8 0L0 176z"/></svg>
    Email
  </a>

  <a href="https://ppuertos.github.io/financial-ml-core/docs" class="md-button" style="display: inline-flex; align-items: center; gap: 8px; margin: 0;">
    <svg style="width: 20px; fill: currentColor;" viewBox="0 0 24 24"><path d="M12 3L1 9l11 6 9-4.91V17h2V9M5 13.18v4L12 21l7-3.82v-4L12 17l-7-3.82Z"/></svg>
    Documentation
  </a>

</div>