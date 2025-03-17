# binance_data_pipeline/processors/__init__.py
"""
Processors for manipulating and consolidating market data.

This package provides processors to transform raw market data
into more useful and consolidated formats.
"""

from .futures_curve import FuturesCurveProcessor
from .market_builder import MarketBuilder

__all__ = [
    "FuturesCurveProcessor",
    "MarketBuilder",
]
