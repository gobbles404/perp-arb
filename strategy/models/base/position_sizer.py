"""
Position Sizing Components - Classes for determining trade size.
"""

from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

from .base_strategy import StrategyContext


# todo: this is where delta warehousing might go
# todo: this is where incorporation of futures margining goes


class PositionSizer(ABC):
    """Base class for position sizing logic."""

    @abstractmethod
    def calculate_position_size(
        self,
        data_row: pd.Series,
        context: StrategyContext,
        capital: float,
        leverage: float,
    ) -> Dict[str, float]:
        """
        Calculate position sizes for all instruments.

        Args:
            data_row: Current market data
            context: Strategy context
            capital: Available capital
            leverage: Leverage multiplier

        Returns:
            Dict mapping instrument names to position sizes
        """
        pass


class EqualNotionalSizer(PositionSizer):
    """Allocates equal notional value to each side of the trade."""

    def calculate_position_size(
        self,
        data_row: pd.Series,
        context: StrategyContext,
        capital: float,
        leverage: float,
    ) -> Dict[str, float]:
        """
        Calculate position sizes with equal notional value.

        For funding arb, this typically means:
        - Long spot position
        - Short perp position of equal notional value
        """
        position_size = capital * leverage / 2  # Split between spot and perp

        spot_price = data_row.get("spot_close", 0)
        perp_price = data_row.get("perp_close", 0)

        if spot_price <= 0 or perp_price <= 0:
            return {"spot_quantity": 0, "perp_quantity": 0}

        spot_quantity = position_size / spot_price
        perp_quantity = position_size / perp_price

        return {
            "spot_quantity": spot_quantity,
            "perp_quantity": perp_quantity,
            "long_notional": spot_quantity * spot_price,
            "short_notional": perp_quantity * perp_price,
        }
