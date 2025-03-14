"""
Enhanced funding rate capture strategy.

This strategy extends the base Beta strategy by adding term futures contracts
to improve capital efficiency while maintaining the same risk profile.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union

from .beta import BetaStrategy
from ..base.signals import EntrySignal, ExitSignal
from ..base.position_sizer import PositionSizer
from ..base.risk_manager import RiskManager


class EnhancedBetaStrategy(BetaStrategy):
    """
    Enhanced funding rate strategy that adds futures contracts.

    This strategy:
    1. Extends the basic Beta strategy
    2. Adds the ability to use term futures for more capital-efficient positioning
    3. Optimizes the allocation between spot, perp, and futures
    """

    def __init__(
        self,
        entry_signals: Optional[List[EntrySignal]] = None,
        exit_signals: Optional[List[ExitSignal]] = None,
        position_sizer: Optional[PositionSizer] = None,
        risk_manager: Optional[RiskManager] = None,
        use_futures: bool = True,
        futures_allocation: float = 0.5,
        name: str = "EnhancedBetaStrategy",
    ):
        """
        Initialize the Enhanced Beta Strategy.

        Args:
            entry_signals: Signals that determine entry conditions
            exit_signals: Signals that determine exit conditions
            position_sizer: Component that determines position sizes
            risk_manager: Component that enforces risk limits
            use_futures: Whether to use futures contracts
            futures_allocation: Portion of short side allocated to futures (vs perp)
            name: Strategy name
        """
        super().__init__(
            entry_signals=entry_signals,
            exit_signals=exit_signals,
            position_sizer=position_sizer,
            risk_manager=risk_manager,
            name=name,
        )
        self.use_futures = use_futures
        self.futures_allocation = futures_allocation

    def initialize_position(
        self, data: pd.DataFrame, capital: float, leverage: float, fee_rate: float
    ) -> Dict[str, Any]:
        """
        Initialize a new position with spot, perp, and futures.

        This overrides the parent method to add futures contract handling.

        Args:
            data: Market data DataFrame
            capital: Initial capital
            leverage: Leverage multiplier
            fee_rate: Trading fee rate

        Returns:
            Dict with position details
        """
        # First call the parent implementation to handle basic entry logic
        position = super().initialize_position(data, capital, leverage, fee_rate)

        # If we didn't enter a position or futures are disabled, just return parent result
        if not self.is_position_open or not self.use_futures:
            return position

        # Check if futures data is available
        first_row = data.iloc[0]
        if not all(col in first_row for col in ["prompt_close", "next_close"]):
            # No futures data available, fallback to perp-only
            return position

        # Recalculate position with futures
        # Keep the spot side the same, but split the short side between perp and futures
        spot_quantity = position["spot_quantity"]
        total_short_notional = spot_quantity * first_row["spot_close"]

        # Allocate short notional between perp and futures
        perp_notional = total_short_notional * (1 - self.futures_allocation)
        futures_notional = total_short_notional * self.futures_allocation

        # Calculate quantities
        perp_quantity = perp_notional / first_row["perp_close"]
        futures_quantity = futures_notional / first_row["prompt_close"]

        # Calculate total fees including futures
        total_notional = (
            (spot_quantity * first_row["spot_close"])
            + (perp_quantity * first_row["perp_close"])
            + (futures_quantity * first_row["prompt_close"])
        )
        total_fee = total_notional * fee_rate

        # Update position
        enhanced_position = {
            "entry_date": position["entry_date"],
            "spot_entry": position["spot_entry"],
            "perp_entry": position["perp_entry"],
            "futures_entry": first_row["prompt_close"],
            "futures_contract": first_row.get("prompt_contract", "Unknown"),
            "spot_quantity": spot_quantity,
            "perp_quantity": perp_quantity,
            "futures_quantity": futures_quantity,
            "capital": capital - total_fee,
            "entry_fee": total_fee,
            "total_notional": total_notional,
            "use_futures": True,
            "futures_allocation": self.futures_allocation,
        }

        # Update the position
        self.position = enhanced_position

        # Update current trade record
        if self.current_trade:
            self.current_trade.update(
                {
                    "futures_entry": first_row["prompt_close"],
                    "futures_contract": first_row.get("prompt_contract", "Unknown"),
                    "futures_quantity": futures_quantity,
                    "perp_quantity": perp_quantity,  # Updated perp quantity
                    "fees": total_fee,
                }
            )

        return enhanced_position

    def calculate_pnl(self, data_row: pd.Series) -> Dict[str, Any]:
        """
        Calculate PnL including futures contracts.

        Args:
            data_row: Current market data

        Returns:
            Dict with PnL components and position status
        """
        # First get the basic PnL calculation from parent
        result = super().calculate_pnl(data_row)

        # If no position or futures disabled, return parent result
        if (
            not self.is_position_open
            or not self.use_futures
            or "futures_quantity" not in self.position
        ):
            return result

        # Check if futures data is available
        if not all(col in data_row for col in ["prompt_close"]):
            return result  # Return without futures calculation if data missing

        # Calculate futures PnL
        futures_price = data_row["prompt_close"]
        futures_quantity = self.position["futures_quantity"]
        futures_entry = self.position["futures_entry"]

        # For short futures position, profit when price goes down
        futures_pnl = futures_quantity * (futures_entry - futures_price)

        # Add futures-specific fields to result
        result.update(
            {
                "futures_pnl": futures_pnl,
                "futures_price": futures_price,
                # Add futures to total notional
                "total_notional": result["total_notional"]
                + (futures_quantity * futures_price),
            }
        )

        # Adjust net market PnL to include futures
        net_market_pnl = result["spot_pnl"] + result["perp_pnl"] + futures_pnl
        result["net_market_pnl"] = net_market_pnl

        return result

    def close_position(self, data_row: pd.Series, fee_rate: float) -> Dict[str, Any]:
        """
        Close the position including futures contracts.

        Args:
            data_row: Current market data
            fee_rate: Trading fee rate

        Returns:
            Dict with position exit details
        """
        # First call the parent implementation for basic close logic
        exit_data = super().close_position(data_row, fee_rate)

        # If the position wasn't open or futures disabled, just return parent result
        if not exit_data["final_total_notional"] or not self.use_futures:
            return exit_data

        # Check if we have futures data in the position and market data
        if "futures_quantity" not in self.position or "prompt_close" not in data_row:
            return exit_data  # Return without futures calculation if data missing

        # Calculate futures exit details
        futures_exit_price = data_row["prompt_close"]
        futures_quantity = self.position["futures_quantity"]
        futures_entry = self.position["futures_entry"]

        # For short futures position, profit when price goes down
        futures_pnl = futures_quantity * (futures_entry - futures_exit_price)
        futures_notional = futures_quantity * futures_exit_price

        # Update exit data to include futures
        enhanced_exit_data = {
            **exit_data,
            "futures_exit": futures_exit_price,
            "futures_pnl": futures_pnl,
            "final_futures_notional": futures_notional,
            "final_total_notional": exit_data["final_total_notional"]
            + futures_notional,
            # Recalculate exit fee to include futures
            "exit_fee": (exit_data["final_total_notional"] + futures_notional)
            * fee_rate,
            # Update net PnL to include futures
            "net_pnl": exit_data["net_pnl"] + futures_pnl,
        }

        # Update the trade record
        if self.current_trade:
            self.current_trade.update(
                {
                    "futures_exit": futures_exit_price,
                    "futures_pnl": futures_pnl,
                    "net_pnl": self.current_trade.get("net_pnl", 0) + futures_pnl,
                    "fees": (exit_data["final_total_notional"] + futures_notional)
                    * fee_rate,
                }
            )

        return enhanced_exit_data
