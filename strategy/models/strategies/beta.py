"""
Pure funding rate capture strategy (Beta Strategy).

This strategy captures funding rate by going long spot and short perpetual futures
with equal notional value. It serves as the benchmark for funding arbitrage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from ..base.base_strategy import BaseStrategy, StrategyContext
from ..base.signals import EntrySignal, ExitSignal
from ..base.position_sizer import PositionSizer
from ..base.risk_manager import RiskManager
from ..markets.spot_perp import SpotPerpMarket


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
        market: Optional[SpotPerpMarket] = None,
    ):
        """
        Initialize the Beta Strategy.

        Args:
            entry_signals: Signals that determine entry conditions
            exit_signals: Signals that determine exit conditions
            position_sizer: Component that determines position sizes
            risk_manager: Component that enforces risk limits
            name: Strategy name
            market: SpotPerpMarket instance (optional, created internally if not provided)
        """
        super().__init__(name=name)

        # Store the components
        self.entry_signals = entry_signals or []
        self.exit_signals = exit_signals or []
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager
        self.market = market

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

        # Create market if not provided
        if self.market is None:
            self.market = SpotPerpMarket(
                data=data, capital=capital, leverage=leverage, fee_rate=fee_rate
            )
            self.market.set_funding_periods_multiplier(self.funding_periods_multiplier)

        # Check if we should enter based on entry signals
        should_enter = True
        for signal in self.entry_signals:
            if not signal.evaluate(first_row, self.context):
                should_enter = False
                break

        # Initialize position through market
        position = self.market.initialize_position(first_row, should_enter)

        # Update strategy state
        self.is_position_open = self.market.is_position_open
        self.periods_held = 1 if self.is_position_open else 0

        # Record trade entry if position opened
        if self.is_position_open:
            self.current_trade = {
                "entry_date": first_row["Timestamp"],
                "spot_entry": position["spot_entry"],
                "perp_entry": position["perp_entry"],
                "spot_quantity": position["spot_quantity"],
                "perp_quantity": position["perp_quantity"],
                "entry_funding": first_row["funding_rate"],
                "entry_capital": capital,
                "fees": position["entry_fee"],
            }

        # Store the position for backward compatibility
        self.position = position

        return position

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

        # Calculate PnL through market
        result = self.market.calculate_pnl(data_row)

        # Check if we should exit based on exit signals
        should_exit = False
        for signal in self.exit_signals:
            if signal.evaluate(data_row, self.context):
                should_exit = True
                break

        # Update periods held
        if self.is_position_open:
            self.periods_held += 1

        # Add exit signal to result
        result["should_exit"] = should_exit

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
        # Close position through market
        exit_data = self.market.close_position(data_row)

        # Complete the current trade record
        if self.current_trade is not None and self.is_position_open:
            self.current_trade.update(
                {
                    "exit_date": data_row["Timestamp"],
                    "spot_exit": exit_data["exit_spot"],
                    "perp_exit": exit_data["exit_perp"],
                    "exit_funding": data_row["funding_rate"],
                    "duration": (
                        (data_row["Timestamp"] - self.current_trade["entry_date"]).days
                        if isinstance(data_row["Timestamp"], pd.Timestamp)
                        else 0
                    ),
                    "spot_pnl": exit_data["spot_pnl"],
                    "perp_pnl": exit_data["perp_pnl"],
                    "net_pnl": exit_data["net_pnl"],
                    "fees": self.current_trade["fees"] + exit_data["exit_fee"],
                }
            )

            # Add to trade history
            self.trade_history.append(self.current_trade)
            self.current_trade = None

        # Reset strategy state
        self.is_position_open = False
        self.periods_held = 0
        old_position = self.position
        self.position = None

        return exit_data

    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics from trade history.

        Returns:
            Dict with trade statistics
        """
        # This method remains unchanged as it's not market-specific
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
