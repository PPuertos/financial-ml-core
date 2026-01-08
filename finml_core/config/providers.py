"""
This module centralizes the configuration for all supported data sources. Its primary 
goal is to enable **total automation**: if a provider is registered here, the 
`MarketDataLoader` knows exactly how to interpret its columns without any user 
intervention.

---

## 🚀 Automated Workflow

This registry acts as the bridge connecting the [Loader](../data/loader.md) with the 
rest of the library. By selecting a valid `data_source`, the system automatically 
configures column mappings and MultiIndex levels (Ticker and Date).

## 🛠️ Current Support

Currently, we provide official support for a single provider, though the architecture 
is designed to be easily scalable:

* **yfinance**: Configured by default to extract OHLCV data using the standard Yahoo Finance schema.

```python
PROVIDERS = {
    'yfinance': {
        'col_map': {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        },
        'ticker_level': 'Ticker',
        'date_level': 'Date'
    }
}
```

!!! question "Which provider would you like to see next?" 
    We are actively working to expand this list to achieve "plug-and-play"
    integration with more sources. If you use a specific provider (such as
    Alpaca, Binance, or Bloomberg) and would like the workflow to be 100%
    automated for them, we would love to hear your suggestions to add them to
    the registry!

    You can reach out by:

    * 🛠️ <a href="https://github.com/PPuertos/financial-ml-core/issues/new" target="_blank">**Open GitHub Issue**</a>
    * ✉️ <a href="mailto:paco_puertos11@hotmail.com" target="_blank">**Email**</a>
    * 💼 <a href="https://www.linkedin.com/in/francisco-puertos-rumayor/" target="_blank">**LinkedIn**</a>
"""

# Master Data for Providers Config
PROVIDERS = {
    'yfinance': {
        'col_map': {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        },
        'ticker_level': 'Ticker',
        'date_level': 'Date'
    }
}