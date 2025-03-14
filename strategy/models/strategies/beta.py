"""
Pure funding rate capture strategy (Beta Strategy).

This strategy captures funding rate by going long spot and short perpetual futures
with equal notional value. It serves as the benchmark for funding arbitrage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from ..base.base_strategy import BaseStrategy
from ..base.signals import EntrySignal, ExitSignal
from ..base.position_sizer import PositionSizer
from ..base.risk_manager import RiskManager
from ..builder.strategy_builder import StrategyContext


class BetaStrategy(BaseStrategy):
    """
    Pure funding rate capture strategy (benchmark).

    This strategy:
    1. Enters when funding rate signal is positive
    2. Maintains equal notional value on spot and perp
    3. Exits when exit signal triggers
    """

    def __init__(
        self,
        entry_signals: Optional[List[EntrySignal]] = None,
        exit_signals: Optional[List[ExitSignal]] = None,
        position_sizer: Optional[PositionSizer] = None,
        risk_manager: Optional[RiskManager] = None,
        name: str = "BetaStrategy",
    ):
        """
        Initialize the Beta Strategy.

        Args:
            entry_signals: Signals that determine entry conditions
            exit_signals: Signals that determine exit conditions
            position_sizer: Component that determines position sizes
            risk_manager: Component that enforces risk limits
            name: Strategy name
        """
        super().__init__(name=name)

        # Store the components
        self.entry_signals = entry_signals or []
        self.exit_signals = exit_signals or []
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager

        # Track periods held and trade history
        self.periods_held = 0
        self.trade_history = []
        self.current_trade = None
        self.is_position_open = False

        # Set default funding periods multiplier
        self.funding_periods_multiplier = 1

    def initialize_position(
        self, data: pd.DataFrame, capital: float, leverage: float, fee_rate: float
    ) -> Dict[str, Any]:
        """
        Initialize a new position if entry signals allow.

        Args:
            data: Market data DataFrame
            capital: Initial capital
            leverage: Leverage multiplier
            fee_rate: Trading fee rate

        Returns:
            Dict with position details
        """
        # Initialize context
        self.context = StrategyContext(data, current_index=0)
        first_row = data.iloc[0]

        # Check if we should enter based on entry signals
        should_enter = True
        for signal in self.entry_signals:
            if not signal.evaluate(first_row, self.context):
                should_enter = False
                break

        if not should_enter:
            # Return empty position
            return {
                "entry_date": first_row["Timestamp"],
                "spot_entry": first_row["spot_close"],
                "perp_entry": first_row["perp_close"],
                "spot_quantity": 0,
                "perp_quantity": 0,
                "capital": capital,
                "entry_fee": 0,
                "total_notional": 0,
            }

        # Calculate position sizes
        if self.position_sizer:
            position_sizes = self.position_sizer.calculate_position_size(
                first_row, self.context, capital, leverage
            )
        else:
            # Default position sizing - equal notional on both sides
            position_size = capital * leverage / 2  # Split between spot and perp
            spot_quantity = position_size / first_row["spot_close"]
            perp_quantity = position_size / first_row["perp_close"]

            position_sizes = {
                "spot_quantity": spot_quantity,
                "perp_quantity": perp_quantity,
                "long_notional": spot_quantity * first_row["spot_close"],
                "short_notional": perp_quantity * first_row["perp_close"],
            }

        # Apply risk management if configured
        if self.risk_manager:
            position_sizes = self.risk_manager.check_risk_limits(
                position_sizes, first_row, self.context
            )

        # Calculate fees
        entry_fee = (
            position_sizes["long_notional"] + position_sizes["short_notional"]
        ) * fee_rate
        total_notional = (
            position_sizes["long_notional"] + position_sizes["short_notional"]
        )

        # Update strategy state
        self.is_position_open = True
        self.periods_held = 1

        # Record trade entry
        self.current_trade = {
            "entry_date": first_row["Timestamp"],
            "spot_entry": first_row["spot_close"],
            "perp_entry": first_row["perp_close"],
            "spot_quantity": position_sizes["spot_quantity"],
            "perp_quantity": position_sizes["perp_quantity"],
            "entry_funding": first_row["funding_rate"],
            "entry_capital": capital,
            "fees": entry_fee,
        }

        # Update position data
        self.position = {
            "entry_date": first_row["Timestamp"],
            "spot_entry": first_row["spot_close"],
            "perp_entry": first_row["perp_close"],
            "spot_quantity": position_sizes["spot_quantity"],
            "perp_quantity": position_sizes["perp_quantity"],
            "capital": capital - entry_fee,
            "entry_fee": entry_fee,
            "total_notional": total_notional,
        }

        return self.position

    def calculate_pnl(self, data_row: pd.Series) -> Dict[str, Any]:
        """
        Calculate PnL and check if we should exit the position.

        Args:
            data_row: Current market data

        Returns:
            Dict with PnL components and position status
        """
        # Update context
        if self.context:
            self.context.current_index += 1

        # Initialize result
        result = {
            "date": data_row["Timestamp"],
            "spot_pnl": 0,
            "perp_pnl": 0,
            "funding_payment": 0,
            "funding_rate": data_row["funding_rate"],
            "should_exit": False,
            "total_notional": 0,
        }

        # If no position is open, return empty result
        if not self.is_position_open or self.position is None:
            return result

        # Get current prices
        spot_price = data_row["spot_close"]
        perp_price = data_row["perp_close"]
        funding_rate = data_row["funding_rate"]

        # Calculate PnL components
        spot_pnl = self.position["spot_quantity"] * (
            spot_price - self.position["spot_entry"]
        )
        perp_pnl = self.position["perp_quantity"] * (
            self.position["perp_entry"] - perp_price
        )

        # Calculate funding payment
        funding_payment = (
            self.position["perp_quantity"]
            * perp_price
            * funding_rate
            * self.funding_periods_multiplier
        )

        # Calculate total notional value
        total_notional = (self.position["spot_quantity"] * spot_price) + (
            self.position["perp_quantity"] * perp_price
        )

        # Update periods held
        self.periods_held += 1

        # Check if we should exit based on exit signals
        should_exit = False
        for signal in self.exit_signals:
            if signal.evaluate(data_row, self.context):
                should_exit = True
                break

        # Update result
        result.update(
            {
                "spot_pnl": spot_pnl,
                "perp_pnl": perp_pnl,
                "funding_payment": funding_payment,
                "funding_rate": funding_rate,
                "should_exit": should_exit,
                "total_notional": total_notional,
            }
        )

        return result

    def close_position(self, data_row: pd.Series, fee_rate: float) -> Dict[str, Any]:
        """
        Close the current position and record the trade.

        Args:
            data_row: Current market data
            fee_rate: Trading fee rate

        Returns:
            Dict with position exit details
        """
        # If no position is open, return empty result
        if not self.is_position_open or self.position is None:
            return {
                "exit_date": data_row["Timestamp"],
                "exit_spot": data_row["spot_close"],
                "exit_perp": data_row["perp_close"],
                "exit_fee": 0,
                "final_total_notional": 0,
            }

        # Get exit prices
        exit_spot = data_row["spot_close"]
        exit_perp = data_row["perp_close"]

        # Calculate final notional and fees
        final_spot_notional = self.position["spot_quantity"] * exit_spot
        final_perp_notional = self.position["perp_quantity"] * exit_perp
        final_total_notional = final_spot_notional + final_perp_notional
        exit_fee = final_total_notional * fee_rate

        # Calculate PnLs for the trade
        spot_pnl = self.position["spot_quantity"] * (
            exit_spot - self.position["spot_entry"]
        )
        perp_pnl = self.position["perp_quantity"] * (
            self.position["perp_entry"] - exit_perp
        )
        net_pnl = spot_pnl + perp_pnl

        # Complete the current trade record
        if self.current_trade is not None:
            self.current_trade.update(
                {
                    "exit_date": data_row["Timestamp"],
                    "spot_exit": exit_spot,
                    "perp_exit": exit_perp,
                    "exit_funding": data_row["funding_rate"],
                    "duration": (
                        (data_row["Timestamp"] - self.current_trade["entry_date"]).days
                        if isinstance(data_row["Timestamp"], pd.Timestamp)
                        else 0
                    ),
                    "spot_pnl": spot_pnl,
                    "perp_pnl": perp_pnl,
                    "net_pnl": net_pnl,
                    "fees": self.current_trade["fees"] + exit_fee,
                }
            )

            # Add to trade history
            self.trade_history.append(self.current_trade)
            self.current_trade = None

        # Reset position state
        self.is_position_open = False
        self.periods_held = 0
        old_position = self.position
        self.position = None

        # Return exit information
        return {
            "exit_date": data_row["Timestamp"],
            "exit_spot": exit_spot,
            "exit_perp": exit_perp,
            "exit_fee": exit_fee,
            "final_spot_notional": final_spot_notional,
            "final_perp_notional": final_perp_notional,
            "final_total_notional": final_total_notional,
            "spot_pnl": spot_pnl,
            "perp_pnl": perp_pnl,
            "net_pnl": net_pnl,
            "position": old_position,
        }

    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics from trade history.

        Returns:
            Dict with trade statistics
        """
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "average_profit": 0,
                "average_loss": 0,
                "profit_factor": 0,
                "average_duration": 0,
            }

        winning_trades = [t for t in self.trade_history if t["net_pnl"] > 0]
        losing_trades = [t for t in self.trade_history if t["net_pnl"] <= 0]

        win_rate = (
            len(winning_trades) / len(self.trade_history) if self.trade_history else 0
        )

        avg_profit = (
            np.mean([t["net_pnl"] for t in winning_trades]) if winning_trades else 0
        )
        avg_loss = (
            np.mean([t["net_pnl"] for t in losing_trades]) if losing_trades else 0
        )

        total_profit = (
            sum(t["net_pnl"] for t in winning_trades) if winning_trades else 0
        )
        total_loss = sum(t["net_pnl"] for t in losing_trades) if losing_trades else 0

        profit_factor = -total_profit / total_loss if total_loss != 0 else float("inf")

        avg_duration = (
            np.mean([t["duration"] for t in self.trade_history])
            if self.trade_history
            else 0
        )

        return {
            "total_trades": len(self.trade_history),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "average_profit": float(avg_profit),
            "average_loss": float(avg_loss),
            "profit_factor": (
                float(profit_factor) if not np.isinf(profit_factor) else float(0)
            ),
            "average_duration": float(avg_duration),
        }
