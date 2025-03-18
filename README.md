# Binance Data Pipeline

A comprehensive system for fetching, processing, and analyzing market data from Binance for statistical arbitrage strategies.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/gobbles404/perp-arb.git
cd binance_data_pipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root with your Binance API credentials:

```
BINANCE_API_KEY=your_key_here
BINANCE_SECRET_KEY=your_secret_here
```

## Usage

### Fetch Market Data

To fetch data for the default market (BTCUSDT):

```bash
# Fetch all data types (spot, futures, funding rates) for BTCUSDT with default settings
python binance_data_pipeline/bin/fetch.py all --symbol BTCUSDT
```

### Fetch Specific Data Types

```bash
# Fetch only spot data
python binance_data_pipeline/bin/fetch.py spot --symbol BTCUSDT --interval 1d

# Fetch only futures data
python binance_data_pipeline/bin/fetch.py futures --symbol BTCUSDT

# Fetch only funding rates
python binance_data_pipeline/bin/fetch.py funding --symbol BTCUSDT
```

### Process Data

After fetching, the data pipeline will automatically run to:

1. Build a futures curve term structure
2. Consolidate all market data into a single dataset

You can also run the pipeline manually:

```bash
python binance_data_pipeline/bin/pipeline.py BTCUSDT "1d 8h 1h"
```

## Data Output

- Raw data: `data/raw/{symbol}/{interval}/`
- Processed data: `data/processed/{symbol}/{interval}/`
- Consolidated market data: `data/markets/{symbol}_{interval}.csv`

## Requirements

Create a `requirements.txt` file with these dependencies:

```
pandas>=1.3.0
python-binance>=1.0.16
python-dotenv>=0.19.0
matplotlib>=3.4.0
```

## Default Settings

- Default symbol: BTCUSDT
- Default intervals: 1d, 8h, 1h
- Data directory: ./binance_data_pipeline/data/

## Directory Structure

```
binance_data_pipeline/
├── bin/
│   ├── fetch.py        # CLI tool for data fetching
│   └── pipeline.py     # Data processing pipeline
├── binance_data_pipeline/
│   ├── core/           # Core components (config, client, etc.)
│   ├── fetchers/       # Data fetchers for different sources
│   ├── processors/     # Data processors and consolidators
│   └── utils/          # Utility functions
└── data/               # Generated data (created on first run)
    ├── raw/            # Raw fetched data
    ├── processed/      # Processed data
    └── markets/        # Consolidated market data
```
