"""
Backtesting engine for trading strategies using vectorbt.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import vectorbt as vbt


class VectorbtBacktester:
    """
    Class for backtesting a strategy on historical data using vectorbt.
    Maintains the same interface as the original Backtester for compatibility.
    """

    def __init__(
        self, strategy, data, initial_capital=10000, leverage=1.0, fee_rate=0.0004
    ):
        """
        Initialize the backtester.

        Parameters:
        - strategy: Strategy class instance (not directly used in vectorbt but kept for interface)
        - data: DataFrame with market data
        - initial_capital: Initial capital amount
        - leverage: Leverage multiplier
        - fee_rate: Trading fee rate
        """
        self.strategy = strategy
        self.data = data.copy().reset_index(drop=True)
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.results = None

    def run(self):
        """Run the backtest using vectorbt and return results."""
        if len(self.data) < 2:
            print("Error: Insufficient data points for backtest")
            return None

        # Extract price data
        spot_price = self.data["spot_close"].values
        perp_price = self.data["perp_close"].values
        funding_rate = self.data["funding_rate"].values
        dates = self.data["Timestamp"].values

        # Determine timeframe multiplier for funding
        if hasattr(self.strategy, "funding_periods_multiplier"):
            funding_multiplier = self.strategy.funding_periods_multiplier
        else:
            # Default to 1 if not specified
            funding_multiplier = 1

        # Adjust funding rate for timeframe if needed
        adjusted_funding_rate = funding_rate * funding_multiplier

        # Calculate position size
        position_size = (
            self.initial_capital * self.leverage / 2
        )  # Split between spot and perp

        # Calculate quantities
        spot_quantity = position_size / spot_price[0]
        perp_quantity = position_size / perp_price[0]

        # Calculate entry fees
        entry_fee = position_size * 2 * self.fee_rate

        # Create a vectorized calculation of P&L
        # For spot: long position, so profit when price goes up
        spot_pnl = spot_quantity * (spot_price - spot_price[0])

        # For perp: short position, so profit when price goes down
        perp_pnl = perp_quantity * (perp_price[0] - perp_price)

        # Calculate funding payments (perp position * perp price * funding rate)
        funding_payments = perp_quantity * perp_price * adjusted_funding_rate

        # Calculate cumulative funding
        cumulative_funding = np.cumsum(funding_payments)

        # Calculate net market P&L (spot + perp)
        net_market_pnl = spot_pnl + perp_pnl

        # Calculate total P&L including funding
        total_pnl = net_market_pnl + cumulative_funding

        # Calculate equity curve
        equity_curve = self.initial_capital + total_pnl - entry_fee

        # Calculate notional values over time
        spot_notional = spot_quantity * spot_price
        perp_notional = perp_quantity * perp_price
        total_notional = spot_notional + perp_notional

        # Calculate exit fee
        exit_fee = total_notional[-1] * self.fee_rate

        # Calculate annualized funding rates
        annualized_funding_rates = funding_rate * 3 * 365 * 100  # Convert to percentage

        # Store results in the same format as the original backtester
        self.results = {
            "entry_date": dates[0],
            "exit_date": dates[-1],
            "initial_capital": self.initial_capital,
            "final_capital": float(equity_curve[-1] - exit_fee),
            "entry_fee": float(entry_fee),
            "exit_fee": float(exit_fee),
            "spot_quantity": float(spot_quantity),
            "perp_quantity": float(perp_quantity),
            "spot_entry": float(spot_price[0]),
            "perp_entry": float(perp_price[0]),
            "spot_exit": float(spot_price[-1]),
            "perp_exit": float(perp_price[-1]),
            "initial_notional": float(total_notional[0]),
            "final_notional": float(total_notional[-1]),
            "dates": dates.tolist(),
            "equity_curve": equity_curve.tolist(),
            "funding_income": funding_payments.tolist(),
            "cumulative_funding": cumulative_funding.tolist(),
            "spot_pnl": spot_pnl.tolist(),
            "perp_pnl": perp_pnl.tolist(),
            "total_pnl": total_pnl.tolist(),
            "net_market_pnl": net_market_pnl.tolist(),
            "notional_values": total_notional.tolist(),
            "funding_rates": funding_rate.tolist(),
            "annualized_funding_rates": annualized_funding_rates.tolist(),
            "data": self.data,
        }

        # Add vectorbt portfolio object for additional analysis if needed
        try:
            # Create a vectorbt Portfolio object from the returns
            returns = pd.Series(
                np.diff(equity_curve) / equity_curve[:-1],
                index=pd.DatetimeIndex(dates[1:]),
            )

            portfolio = vbt.Portfolio.from_returns(
                returns,
                init_capital=self.initial_capital,
                freq="D",  # Assuming daily frequency, adjust if different
            )

            self.results["vbt_portfolio"] = portfolio
        except Exception as e:
            print(f"Warning: Could not create vectorbt portfolio: {e}")
            self.results["vbt_portfolio"] = None

        return self.results
