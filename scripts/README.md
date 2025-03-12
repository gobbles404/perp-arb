# Binance Data Fetcher

This is a unified system for fetching various types of data from Binance.

## Usage

The main entry point is the `fetch.py` script, which supports various commands for different data types.

### Fetch Spot Market Data

```bash
python fetch.py spot --symbol BTCUSDT --interval 1d,8h,1h --start 2023-01-01 --end 2023-12-31
```
