"""
Backtesting engine for trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class Backtester:
    """
    Class for backtesting a strategy on historical data.
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

    def run(self):
        """Run the backtest and return results."""
        if len(self.data) < 2:
            print("Error: Insufficient data points for backtest")
            return None

        # Initialize position
        position = self.strategy.initialize_position(
            self.data, self.initial_capital, self.leverage, self.fee_rate
        )

        # Initialize tracking arrays
        dates = [position["entry_date"]]
        equity_curve = [position["capital"]]
        funding_income = [0]
        cumulative_funding = [0]
        spot_pnl = [0]
        perp_pnl = [0]
        total_pnl = [0]
        net_market_pnl = [0]
        notional_values = [position["total_notional"]]
        funding_rates = [0]
        annualized_funding_rates = [0]

        # Simulate over time
        for i in range(1, len(self.data)):
            # Get current data point
            current_row = self.data.iloc[i]

            # Calculate PnL
            result = self.strategy.calculate_pnl(current_row)

            # Handle potential NaN in funding payment
            funding_payment = result["funding_payment"]
            if np.isnan(funding_payment):
                funding_payment = 0

            # Update cumulative metrics
            current_funding = cumulative_funding[-1] + funding_payment

            # Calculate market PnLs
            current_spot_pnl = result["spot_pnl"]
            current_perp_pnl = result["perp_pnl"]

            # Handle potential NaN in PnL components
            if np.isnan(current_spot_pnl):
                current_spot_pnl = 0
            if np.isnan(current_perp_pnl):
                current_perp_pnl = 0

            current_net_market_pnl = current_spot_pnl + current_perp_pnl
            current_total_pnl = current_net_market_pnl + current_funding
            current_capital = (
                self.initial_capital + current_total_pnl - position["entry_fee"]
            )

            # Append to tracking arrays
            dates.append(result["date"])
            equity_curve.append(current_capital)
            funding_income.append(funding_payment)
            cumulative_funding.append(current_funding)
            spot_pnl.append(current_spot_pnl)
            perp_pnl.append(current_perp_pnl)
            total_pnl.append(current_total_pnl)
            net_market_pnl.append(current_net_market_pnl)
            notional_values.append(result["total_notional"])

            # Handle potential NaN in funding rate
            funding_rate = result["funding_rate"]
            if np.isnan(funding_rate):
                funding_rate = 0

            funding_rates.append(funding_rate)

            # Calculate annualized funding rate
            annualized_rate = funding_rate * 3 * 365 * 100  # 3 funding periods per day
            annualized_funding_rates.append(annualized_rate)

        # Close position at the end
        exit_data = self.strategy.close_position(self.data.iloc[-1], self.fee_rate)
        final_capital = equity_curve[-1] - exit_data["exit_fee"]

        # Compile all results
        self.results = {
            "entry_date": position["entry_date"],
            "exit_date": exit_data["exit_date"],
            "initial_capital": self.initial_capital,
            "final_capital": final_capital,
            "entry_fee": position["entry_fee"],
            "exit_fee": exit_data["exit_fee"],
            "spot_quantity": position["spot_quantity"],
            "perp_quantity": position["perp_quantity"],
            "spot_entry": position["spot_entry"],
            "perp_entry": position["perp_entry"],
            "spot_exit": exit_data["exit_spot"],
            "perp_exit": exit_data["exit_perp"],
            "initial_notional": position["total_notional"],
            "final_notional": exit_data["final_total_notional"],
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
            "data": self.data,
        }

        return self.results
