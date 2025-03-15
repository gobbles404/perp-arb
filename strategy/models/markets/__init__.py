# models/markets/__init__.py
"""
Market structures module.

This module contains classes that define different market structures
which strategies can use to trade.
"""

from .spot_perp import SpotPerpMarket

__all__ = ["SpotPerpMarket"]
