# Binance Data Pipeline

A comprehensive data pipeline for fetching, processing, and analyzing market data from Binance with a focus on funding arbitrage strategies.

## Overview

The Binance Data Pipeline is a modular Python framework designed to:

- Fetch various types of market data from Binance (spot, futures, funding rates)
- Build futures term structure and analyze contango/backwardation
- Consolidate multiple data sources into a unified market view
- Support funding arbitrage and other trading strategies

## Features

- **Unified CLI Interface**: Simple command-line tools for all data operations
- **Multiple Data Sources**: Support for spot markets, perpetual futures, quarterly contracts, and funding rates
- **Flexible Time Intervals**: Process data at various granularities (1d, 8h, 1h, etc.)
- **Futures Curve Building**: Calculate term structure with configurable roll periods
- **Consolidated Market Data**: Combine multiple data sources with consistent column naming

## Installation

### Prerequisites

- Python 3.8+
- Binance API credentials

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/binance-data-pipeline.git
   cd binance-data-pipeline
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Binance API credentials:
   ```
   BINANCE_API_KEY=your_api_key_here
   BINANCE_SECRET_KEY=your_secret_key_here
   ```

## Project Structure

```
binance_data_pipeline/
├── bin/                      # CLI scripts
│   ├── fetch.py              # Data fetching script
│   └── pipeline.py           # Data processing script
├── binance_data_pipeline/    # Main package
│   ├── core/                 # Core components
│   │   ├── config.py         # Configuration management
│   │   ├── client.py         # Binance API client
│   │   └── logger.py         # Logging setup
│   ├── fetchers/             # Data fetchers
│   │   ├── base.py           # Base fetcher class
│   │   ├── spot.py           # Spot market data fetcher
│   │   ├── futures.py        # Futures data fetcher
│   │   ├── funding_rates.py  # Funding rates fetcher
│   │   └── contract_details.py # Contract specifications fetcher
│   ├── processors/           # Data processors
│   │   ├── futures_curve.py  # Futures term structure processor
│   │   └── market_builder.py # Market data consolidation
│   └── utils/                # Utility functions
│       ├── date_utils.py     # Date/time utilities
│       ├── file_utils.py     # File operations
│       └── binance_utils.py  # Binance-specific utilities
├── data/                     # Data storage
│   ├── raw/                  # Raw data from Binance
│   ├── processed/            # Processed data (futures curves, etc.)
│   ├── markets/              # Consolidated market data
│   └── contracts/            # Contract specifications
└── tests/                    # Test suite
```

## Usage

### Fetching Data

```bash
# Fetch spot data
python bin/fetch.py spot --symbol BTCUSDT --interval 1d --start 2023-06-01 --end 2023-07-01

# Fetch futures data (both perpetual and quarterly contracts)
python bin/fetch.py futures --symbol BTCUSDT

# Fetch funding rates
python bin/fetch.py funding --symbol BTCUSDT

# Fetch contract details
python bin/fetch.py contracts --type both

# Fetch all data types for a single symbol
python bin/fetch.py all --symbol BTCUSDT --interval 1d
```

### Processing Data

```bash
# Run the complete data processing pipeline
python bin/pipeline.py BTCUSDT "1d 8h 1h"

# Specify a custom futures roll period
python bin/pipeline.py BTCUSDT "1d" --futures-roll 14d
```

### Programmatic Usage

```python
from binance_data_pipeline.fetchers import SpotFetcher
from binance_data_pipeline.processors import MarketBuilder

# Fetch spot data
fetcher = SpotFetcher(symbol="BTCUSDT", intervals=["1d", "8h"])
fetcher.fetch_all()

# Build consolidated market data
builder = MarketBuilder(symbol="BTCUSDT", intervals=["1d"])
builder.build_all()
```

## Configuration

Default configuration is defined in `binance_data_pipeline/core/config.py`:

- **Default Symbol**: BTCUSDT
- **Default Intervals**:
  - Spot: 1d, 8h, 1h
  - Futures: 1d, 8h, 1h
  - Funding: 8h
- **Data Directories**: Paths for raw, processed, and market data

## Data Output

The pipeline produces several types of output files:

- **Raw Data**: `data/raw/{symbol}/{interval}/{type}.csv`
- **Futures Curve**: `data/processed/{symbol}/{interval}/futures_curve_{roll_period}_roll.csv`
- **Market Data**: `data/markets/{symbol}_{interval}.csv`
- **Contract Specs**: `data/contracts/{type}_specs.csv`

## Extension and Customization

The modular architecture allows easy extension:

1. Create a new fetcher by subclassing `BinanceFetcher`
2. Create a new processor by implementing similar interfaces to existing processors
3. Add new CLI commands by extending `bin/fetch.py` or `bin/pipeline.py`

## Dependencies

- `pandas`: Data manipulation
- `python-binance`: Official Binance API client
- `python-dotenv`: Environment variable management
