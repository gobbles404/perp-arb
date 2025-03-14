"""
Backtesting engine for adaptive trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class AdaptiveBacktester:
    """
    Class for backtesting a strategy that can dynamically enter and exit positions.
    """

    def __init__(
        self, strategy, data, initial_capital=10000, leverage=1.0, fee_rate=0.0004
    ):
        self.strategy = strategy
        self.data = data.copy().reset_index(drop=True)
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.results = None
        self.current_capital = initial_capital

    def run(self):
        """Run the backtest and return results."""
        if len(self.data) < 2:
            print("Error: Insufficient data points for backtest")
            return None

        # Initialize tracking arrays
        dates = []
        equity_curve = []
        funding_income = []
        cumulative_funding = []
        spot_pnl = []
        perp_pnl = []
        total_pnl = []
        net_market_pnl = []
        notional_values = []
        funding_rates = []
        annualized_funding_rates = []
        position_status = []  # 1 for in position, 0 for out of position

        # Initialize position (may not open if first funding rate is negative)
        position = self.strategy.initialize_position(
            self.data, self.current_capital, self.leverage, self.fee_rate
        )

        # Record initial state
        dates.append(position["entry_date"])
        equity_curve.append(self.current_capital)
        funding_income.append(0)
        cumulative_funding.append(0)
        spot_pnl.append(0)
        perp_pnl.append(0)
        total_pnl.append(0)
        net_market_pnl.append(0)
        notional_values.append(position["total_notional"])
        funding_rates.append(self.data["funding_rate"].iloc[0])
        annualized_funding_rates.append(
            self.data["funding_rate"].iloc[0] * 3 * 365 * 100
        )
        position_status.append(1 if self.strategy.is_position_open else 0)

        # Update capital based on initial entry fee
        self.current_capital -= (
            position["entry_fee"] if self.strategy.is_position_open else 0
        )

        # Simulate over time
        for i in range(1, len(self.data)):
            # Get current data point
            current_row = self.data.iloc[i]

            # Calculate PnL for current state
            result = self.strategy.calculate_pnl(current_row)

            # Handle potential NaN in funding payment
            funding_payment = result["funding_payment"]
            if np.isnan(funding_payment):
                funding_payment = 0

            # Update cumulative metrics
            current_funding = (
                (cumulative_funding[-1] + funding_payment)
                if cumulative_funding
                else funding_payment
            )

            # Calculate market PnLs
            current_spot_pnl = result["spot_pnl"]
            current_perp_pnl = result["perp_pnl"]

            # Handle potential NaN in PnL components
            if np.isnan(current_spot_pnl):
                current_spot_pnl = 0
            if np.isnan(current_perp_pnl):
                current_perp_pnl = 0

            current_net_market_pnl = current_spot_pnl + current_perp_pnl

            # Check if we need to exit the position
            if self.strategy.is_position_open and result["should_exit"]:
                # Close the position
                exit_data = self.strategy.close_position(current_row, self.fee_rate)

                # Update capital with exit fee
                self.current_capital -= exit_data["exit_fee"]

            # Check if we should enter a new position
            elif (
                not self.strategy.is_position_open
                and self.strategy.should_enter_position(
                    current_row, self.current_capital
                )
            ):
                # Initialize a new position
                position = self.strategy.initialize_position(
                    pd.DataFrame([current_row]),
                    self.current_capital,
                    self.leverage,
                    self.fee_rate,
                )

                # Update capital with entry fee
                self.current_capital -= (
                    position["entry_fee"] if self.strategy.is_position_open else 0
                )

            # Calculate total PnL and current capital
            current_total_pnl = current_spot_pnl + current_perp_pnl + current_funding
            self.current_capital = self.initial_capital + current_total_pnl

            # Append to tracking arrays
            dates.append(current_row["Timestamp"])
            equity_curve.append(self.current_capital)
            funding_income.append(funding_payment)
            cumulative_funding.append(current_funding)
            spot_pnl.append(current_spot_pnl)
            perp_pnl.append(current_perp_pnl)
            total_pnl.append(current_total_pnl)
            net_market_pnl.append(current_net_market_pnl)
            notional_values.append(result["total_notional"])

            # Handle potential NaN in funding rate
            funding_rate = current_row["funding_rate"]
            if np.isnan(funding_rate):
                funding_rate = 0

            funding_rates.append(funding_rate)

            # Calculate annualized funding rate
            annualized_rate = funding_rate * 3 * 365 * 100  # 3 funding periods per day
            annualized_funding_rates.append(annualized_rate)

            # Record if we're in a position
            position_status.append(1 if self.strategy.is_position_open else 0)

        # Close any open position at the end
        final_exit_data = self.strategy.close_position(
            self.data.iloc[-1], self.fee_rate
        )
        final_capital = self.current_capital - (
            final_exit_data["exit_fee"] if self.strategy.is_position_open else 0
        )

        # Get trade statistics
        trade_stats = self.strategy.get_trade_statistics()

        # Compile all results
        self.results = {
            "initial_capital": self.initial_capital,
            "final_capital": final_capital,
            "entry_fee": (
                sum(t["fees"] for t in self.strategy.trade_history)
                if hasattr(self.strategy, "trade_history")
                else 0
            ),
            "exit_fee": 0,  # This is included in the trade history fees
            "dates": dates,
            "equity_curve": equity_curve,
            "funding_income": funding_income,
            "cumulative_funding": cumulative_funding,
            "spot_pnl": spot_pnl,
            "perp_pnl": perp_pnl,
            "total_pnl": total_pnl,
            "net_market_pnl": net_market_pnl,
            "notional_values": notional_values,
            "funding_rates": funding_rates,
            "annualized_funding_rates": annualized_funding_rates,
            "position_status": position_status,
            "data": self.data,
            "trades": (
                self.strategy.trade_history
                if hasattr(self.strategy, "trade_history")
                else []
            ),
            "trade_stats": trade_stats,
            # For compatibility with metrics calculation
            "entry_date": dates[0],
            "exit_date": dates[-1],
            "spot_quantity": 0,  # Multiple positions, so this is not meaningful
            "perp_quantity": 0,  # Multiple positions, so this is not meaningful
            "initial_notional": notional_values[0],
            "final_notional": notional_values[-1],
        }

        return self.results
