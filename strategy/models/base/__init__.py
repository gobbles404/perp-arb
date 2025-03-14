"""
Base components for the strategy framework.

This package contains the foundational classes for building trading strategies.
"""

# Import the most commonly used classes for convenience
from .base_strategy import BaseStrategy
from .signals import (
    Signal,
    EntrySignal,
    ExitSignal,
    CompositeSignal,
    FundingRateSignal,
    VolatilitySignal,
)
from .position_sizer import PositionSizer, EqualNotionalSizer
from .risk_manager import RiskManager
from .context import StrategyContext

# Version
__version__ = "0.1.0"
