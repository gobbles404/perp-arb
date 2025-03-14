"""
Strategy implementations module.

This module contains concrete strategy implementations that can be used with the backtesting engine.
"""

from .beta import BetaStrategy
from .enhanced_beta import EnhancedBetaStrategy

__all__ = ["BetaStrategy", "EnhancedBetaStrategy"]
