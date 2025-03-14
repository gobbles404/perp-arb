"""
Adaptive Funding Arbitrage Strategy implementation.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class AdaptiveFundingArbStrategy:
    """
    Strategy that adaptively enters and exits positions based on funding rates.

    This strategy:
    1. Enters a position (long spot, short perp) when funding rate exceeds the funding_threshold
    2. Exits the position when funding rate drops below the exit_threshold
    3. Maintains each position for at least min_holding_periods
    """

    def __init__(
        self,
        timeframe="1d",
        funding_threshold=0.0,
        exit_threshold=0.0,
        min_holding_periods=1,
        funding_periods_multiplier=1,
    ):
        """
        Initialize the adaptive funding arbitrage strategy.

        Parameters:
        -----------
        timeframe : str
            Timeframe of the data (e.g., '1d', '8h')
        funding_threshold : float
            Minimum funding rate to enter a position (default: 0.0)
        exit_threshold : float
            Funding rate threshold to exit a position (default: 0.0)
        min_holding_periods : int
            Minimum number of periods to hold a position (default: 1)
        funding_periods_multiplier : int
            Multiplier to adjust funding for different timeframes
        """
        self.timeframe = timeframe
        self.funding_threshold = funding_threshold
        self.exit_threshold = exit_threshold
        self.min_holding_periods = min_holding_periods

        # Set funding periods multiplier based on timeframe
        self.funding_periods_multiplier = funding_periods_multiplier
        if timeframe == "1d":
            self.funding_periods_multiplier = 3  # 3 funding periods per day
        elif timeframe == "8h":
            self.funding_periods_multiplier = 1  # 1 funding period per 8h
        elif timeframe == "4h":
            self.funding_periods_multiplier = 0.5  # 0.5 funding period per 4h

        # Initialize position tracking
        self.is_position_open = False
        self.entry_data = None
        self.position_size = 0
        self.spot_quantity = 0
        self.perp_quantity = 0
        self.periods_held = 0
        self.trade_history = []
        self.current_trade = None

    def initialize_position(self, data, initial_capital, leverage, fee_rate):
        """
        Initialize position based on strategy rules.

        Parameters:
        -----------
        data : pandas.DataFrame
            Market data
        initial_capital : float
            Initial capital amount
        leverage : float
            Leverage multiplier
        fee_rate : float
            Trading fee rate

        Returns:
        --------
        dict
            Position details
        """
        # Reset position state
        self.is_position_open = False
        self.entry_data = None
        self.spot_quantity = 0
        self.perp_quantity = 0
        self.periods_held = 0

        # Get first row to check if we should enter
        first_row = data.iloc[0]

        # Check if the initial funding rate meets our threshold
        if first_row["funding_rate"] >= self.funding_threshold:
            # Enter position
            return self._enter_position(first_row, initial_capital, leverage, fee_rate)
        else:
            # Return empty position
            return {
                "entry_date": first_row["Timestamp"],
                "spot_entry": first_row["spot_close"],
                "perp_entry": first_row["perp_close"],
                "spot_quantity": 0,
                "perp_quantity": 0,
                "capital": initial_capital,
                "entry_fee": 0,
                "total_notional": 0,
            }

    def _enter_position(self, row, capital, leverage, fee_rate):
        """
        Enter a new position.
        """
        # Calculate position size
        position_size = capital * leverage / 2  # Split between spot and perp

        # Calculate quantities
        spot_entry = row["spot_close"]
        perp_entry = row["perp_close"]
        spot_quantity = position_size / spot_entry
        perp_quantity = position_size / perp_entry

        # Calculate fees
        entry_fee = position_size * 2 * fee_rate  # fees for both spot and perp

        # Calculate notional value
        total_notional = (spot_quantity * spot_entry) + (perp_quantity * perp_entry)

        # Update state
        self.is_position_open = True
        self.entry_data = {
            "timestamp": row["Timestamp"],
            "spot_entry": spot_entry,
            "perp_entry": perp_entry,
            "spot_quantity": spot_quantity,
            "perp_quantity": perp_quantity,
            "entry_capital": capital,
            "entry_fee": entry_fee,
        }
        self.spot_quantity = spot_quantity
        self.perp_quantity = perp_quantity
        self.periods_held = 1

        # Start tracking current trade
        self.current_trade = {
            "entry_date": row["Timestamp"],
            "spot_entry": spot_entry,
            "perp_entry": perp_entry,
            "spot_quantity": spot_quantity,
            "perp_quantity": perp_quantity,
            "entry_funding": row["funding_rate"],
            "entry_capital": capital,
            "fees": entry_fee,
        }

        return {
            "entry_date": row["Timestamp"],
            "spot_entry": spot_entry,
            "perp_entry": perp_entry,
            "spot_quantity": spot_quantity,
            "perp_quantity": perp_quantity,
            "capital": capital - entry_fee,  # Subtract entry fee from available capital
            "entry_fee": entry_fee,
            "total_notional": total_notional,
        }

    def calculate_pnl(self, row):
        """
        Calculate PnL for the current position based on given data row.

        Parameters:
        -----------
        row : pandas.Series
            Current data row

        Returns:
        --------
        dict
            PnL calculation results and position status
        """
        # Get current prices
        spot_price = row["spot_close"]
        perp_price = row["perp_close"]
        funding_rate = row["funding_rate"]

        # Initialize result
        result = {
            "date": row["Timestamp"],
            "spot_pnl": 0,
            "perp_pnl": 0,
            "funding_payment": 0,
            "funding_rate": funding_rate,
            "should_exit": False,
            "total_notional": 0,
        }

        # If no position is open, return empty result
        if not self.is_position_open or self.entry_data is None:
            return result

        # Calculate PnL components
        spot_pnl = self.spot_quantity * (spot_price - self.entry_data["spot_entry"])
        perp_pnl = self.perp_quantity * (self.entry_data["perp_entry"] - perp_price)

        # Calculate funding payment (perp position * perp price * funding rate * multiplier)
        funding_payment = (
            self.perp_quantity
            * perp_price
            * funding_rate
            * self.funding_periods_multiplier
        )

        # Calculate total notional value
        total_notional = (self.spot_quantity * spot_price) + (
            self.perp_quantity * perp_price
        )

        # Update periods held
        self.periods_held += 1

        # Determine if we should exit
        # Exit if funding rate drops below threshold and we've held for minimum periods
        should_exit = (
            funding_rate < self.exit_threshold
            and self.periods_held >= self.min_holding_periods
        )

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

    def close_position(self, row, fee_rate):
        """
        Close the current position.

        Parameters:
        -----------
        row : pandas.Series
            Current data row
        fee_rate : float
            Trading fee rate

        Returns:
        --------
        dict
            Position exit details
        """
        # If no position is open, return empty result
        if not self.is_position_open or self.entry_data is None:
            return {
                "exit_date": row["Timestamp"],
                "exit_spot": row["spot_close"],
                "exit_perp": row["perp_close"],
                "exit_fee": 0,
                "final_total_notional": 0,
            }

        # Get exit prices
        exit_spot = row["spot_close"]
        exit_perp = row["perp_close"]

        # Calculate fees and notional
        final_total_notional = (self.spot_quantity * exit_spot) + (
            self.perp_quantity * exit_perp
        )
        exit_fee = final_total_notional * fee_rate

        # Calculate PnLs for the trade
        spot_pnl = self.spot_quantity * (exit_spot - self.entry_data["spot_entry"])
        perp_pnl = self.perp_quantity * (self.entry_data["perp_entry"] - exit_perp)
        net_pnl = spot_pnl + perp_pnl

        # Complete the current trade record
        if self.current_trade is not None:
            self.current_trade.update(
                {
                    "exit_date": row["Timestamp"],
                    "spot_exit": exit_spot,
                    "perp_exit": exit_perp,
                    "exit_funding": row["funding_rate"],
                    "duration": (
                        row["Timestamp"] - self.current_trade["entry_date"]
                    ).days,
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
        self.spot_quantity = 0
        self.perp_quantity = 0
        self.periods_held = 0

        return {
            "exit_date": row["Timestamp"],
            "exit_spot": exit_spot,
            "exit_perp": exit_perp,
            "exit_fee": exit_fee,
            "final_total_notional": final_total_notional,
        }

    def should_enter_position(self, row, available_capital):
        """
        Determine if we should enter a new position.

        Parameters:
        -----------
        row : pandas.Series
            Current data row
        available_capital : float
            Available capital

        Returns:
        --------
        bool
            True if we should enter a position, False otherwise
        """
        # Don't enter if we already have a position
        if self.is_position_open:
            return False

        # Enter if funding rate meets our threshold
        return row["funding_rate"] >= self.funding_threshold

    def get_trade_statistics(self):
        """
        Calculate statistics from trade history.

        Returns:
        --------
        dict
            Trade statistics
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
