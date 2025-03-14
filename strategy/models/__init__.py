"""
Trading strategy models module.

This module contains strategy implementations and building blocks for
creating cryptocurrency trading strategies, with a focus on funding arbitrage.
"""

# Import key strategy implementations
from .funding_arb import FundingArbStrategy
from .adaptive_funding_arb import AdaptiveFundingArbStrategy

# Import submodules
from . import base
from . import builder
from . import strategies

# Define what gets imported with "from models import *"
__all__ = [
    # Core strategies
    "FundingArbStrategy",
    "AdaptiveFundingArbStrategy",
    # Submodules
    "base",
    "builder",
    "strategies",
]
