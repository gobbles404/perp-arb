# binance_data_pipeline/fetchers/__init__.py
"""
Fetchers for Binance market data.

This package provides specialized fetchers for different types of
market data from the Binance exchange API.
"""

from .base import BinanceFetcher
from .spot import SpotFetcher
from .futures import FuturesFetcher
from .funding_rates import FundingRatesFetcher
from .contract_details import ContractDetailsFetcher

__all__ = [
    "BinanceFetcher",
    "SpotFetcher",
    "FuturesFetcher",
    "FundingRatesFetcher",
    "ContractDetailsFetcher",
]
