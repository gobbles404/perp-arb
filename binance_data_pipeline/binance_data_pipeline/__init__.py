"""
Binance Data Pipeline

A comprehensive system for fetching, processing, and analyzing market data from Binance.

This package provides a unified interface for:
- Fetching different types of market data (spot, futures, funding rates)
- Building futures term structure
- Consolidating multiple data sources into a single market view
- Supporting funding arbitrage strategies

Usage:
    # Fetch spot data
    from binance_data_pipeline.fetchers import SpotFetcher
    fetcher = SpotFetcher(symbol="BTCUSDT", intervals=["1d", "8h"])
    fetcher.fetch_all()

    # Process data
    from binance_data_pipeline.processors import MarketBuilder
    builder = MarketBuilder(symbol="BTCUSDT", intervals=["1d"])
    builder.build_all()
"""

__version__ = "0.1.0"

# Export core components
from .core.config import config
from .core.logger import get_logger

# Export fetcher classes for easy import
from .fetchers import (
    SpotFetcher,
    FuturesFetcher,
    FundingRatesFetcher,
    ContractDetailsFetcher,
)

# Export processor classes for easy import
from .processors import (
    FuturesCurveProcessor,
    MarketBuilder,
)

# Define package-level exports
__all__ = [
    # Core components
    "config",
    "get_logger",
    # Fetchers
    "SpotFetcher",
    "FuturesFetcher",
    "FundingRatesFetcher",
    "ContractDetailsFetcher",
    # Processors
    "FuturesCurveProcessor",
    "MarketBuilder",
]
