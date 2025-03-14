"""
Risk Management Components - Classes for controlling position risk.
"""

from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

from .base_strategy import StrategyContext


class RiskManager(ABC):
    """Base class for risk management components."""

    @abstractmethod
    def check_risk_limits(
        self,
        position_sizing: Dict[str, float],
        data_row: pd.Series,
        context: StrategyContext,
    ) -> Dict[str, float]:
        """
        Check if position sizing meets risk constraints.

        Args:
            position_sizing: Proposed position sizes
            data_row: Current market data
            context: Strategy context

        Returns:
            Adjusted position sizes that meet risk constraints
        """
        pass


class VolatilityBasedRiskManager(RiskManager):
    """Adjusts position sizes based on market volatility."""

    def __init__(
        self,
        lookback: int = 20,
        max_risk_pct: float = 2.0,
        price_column: str = "spot_close",
    ):
        """
        Initialize the volatility-based risk manager.

        Args:
            lookback: Periods to look back for volatility calculation
            max_risk_pct: Maximum risk as percentage of capital
            price_column: Column name for price data
        """
        self.lookback = lookback
        self.max_risk_pct = max_risk_pct
        self.price_column = price_column

    def check_risk_limits(
        self,
        position_sizing: Dict[str, float],
        data_row: pd.Series,
        context: StrategyContext,
    ) -> Dict[str, float]:
        """
        Adjust position sizes based on current volatility.

        Higher volatility leads to smaller positions.
        """
        window = context.get_data_window(self.lookback)

        if self.price_column not in window.columns or len(window) < 2:
            return position_sizing  # No adjustment if insufficient data

        returns = window[self.price_column].pct_change().dropna()

        if len(returns) < 2:
            return position_sizing  # No adjustment if insufficient data

        # Calculate daily volatility and annualize
        daily_vol = returns.std()
        annualized_vol = daily_vol * (252**0.5)

        # Determine volatility scaling factor
        # Higher volatility = lower scaling factor
        base_vol = 0.2  # 20% annualized is considered "normal"
        vol_scaling = min(1.0, base_vol / max(annualized_vol, 0.001))

        # Apply scaling to all position sizes
        adjusted_sizing = {}
        for key, value in position_sizing.items():
            if key.endswith("_quantity"):
                adjusted_sizing[key] = value * vol_scaling
            else:
                adjusted_sizing[key] = value

        # Recalculate notional values
        if "spot_quantity" in adjusted_sizing and "spot_close" in data_row:
            spot_price = data_row["spot_close"]
            adjusted_sizing["long_notional"] = (
                adjusted_sizing["spot_quantity"] * spot_price
            )

        if "perp_quantity" in adjusted_sizing and "perp_close" in data_row:
            perp_price = data_row["perp_close"]
            adjusted_sizing["short_notional"] = (
                adjusted_sizing["perp_quantity"] * perp_price
            )

        return adjusted_sizing
