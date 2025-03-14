"""
Strategy implementations module.

This module contains concrete strategy implementations that can be used with the backtesting engine.
"""

from .beta_strategy import BetaStrategy
from .enhanced_beta_strategy import EnhancedBetaStrategy

__all__ = ["BetaStrategy", "EnhancedBetaStrategy"]
