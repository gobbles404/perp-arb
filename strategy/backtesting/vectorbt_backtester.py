"""
Backtesting engine for trading strategies using vectorbt.
Enhanced with risk monitoring capabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import vectorbt as vbt


class VectorbtBacktester:
    """
    Class for backtesting a strategy on historical data using vectorbt.
    Maintains the same interface as the original Backtester for compatibility.
    Enhanced with risk monitoring metrics.
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

        # Extract margin requirements from data if available
        self.maint_margin_pct = self._get_maint_margin_pct()
        self.initial_margin_pct = self._get_initial_margin_pct()
        self.liquidation_fee_pct = self._get_liquidation_fee_pct()

    def _get_maint_margin_pct(self):
        """Extract maintenance margin percentage from data."""
        if "perpetual_maint" in self.data.columns:
            return self.data["perpetual_maint"].iloc[0]
        return 2.5  # Default value of 2.5%

    def _get_initial_margin_pct(self):
        """Extract initial margin percentage from data."""
        if "perpetual_initial" in self.data.columns:
            return self.data["perpetual_initial"].iloc[0]
        return 5.0  # Default value of 5.0%

    def _get_liquidation_fee_pct(self):
        """Extract liquidation fee percentage from data."""
        if "perpetual_liquidation_fee" in self.data.columns:
            return self.data["perpetual_liquidation_fee"].iloc[0]
        return 0.5  # Default value of 0.5%

    def run(self):
        """Run the backtest using vectorbt and return results with risk metrics."""
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

        # ----- RISK METRICS CALCULATION -----

        # Arrays to store risk metrics
        health_factors = []
        liquidation_prices = []
        buffer_percentages = []
        basis_percentages = []

        # Calculate risk metrics for each time step
        for i in range(len(dates)):
            # Calculate current equity
            current_equity = self.initial_capital + total_pnl[i] - entry_fee

            # Calculate maintenance margin requirement
            current_perp_notional = perp_notional[i]
            maint_margin_requirement = current_perp_notional * (
                self.maint_margin_pct / 100
            )

            # Calculate health factor
            health_factor = (
                current_equity / maint_margin_requirement
                if maint_margin_requirement > 0
                else float("inf")
            )
            health_factors.append(health_factor)

            # Calculate liquidation price for short perp position
            # Liquidation occurs when: equity <= maintenance margin
            # For a short position, price increase leads to losses
            liquidation_price = perp_price[0] * (
                1 + (self.maint_margin_pct / 100) / self.leverage
            )
            liquidation_prices.append(liquidation_price)

            # Calculate buffer to liquidation (as percentage)
            buffer_pct = (
                ((liquidation_price - perp_price[i]) / perp_price[i]) * 100
                if perp_price[i] > 0
                else 0
            )
            buffer_percentages.append(buffer_pct)

            # Calculate current basis between spot and perp
            basis_pct = (
                ((perp_price[i] / spot_price[i]) - 1) * 100 if spot_price[i] > 0 else 0
            )
            basis_percentages.append(basis_pct)

        # Find times when health factor approached danger level
        close_calls = sum(1 for hf in health_factors if hf is not None and hf < 1.2)

        # Find minimum health factor
        min_health = min([h for h in health_factors if h is not None], default=None)

        # Store results in the same format as the original backtester with added risk metrics
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
            # Risk metrics
            "health_factors": health_factors,
            "liquidation_prices": liquidation_prices,
            "buffer_percentages": buffer_percentages,
            "basis_percentages": basis_percentages,
            "latest_health_factor": health_factors[-1] if health_factors else None,
            "latest_liquidation_price": (
                liquidation_prices[-1] if liquidation_prices else None
            ),
            "min_health_factor": min_health,
            "close_calls": close_calls,
            "maintenance_margin_pct": self.maint_margin_pct,
            "initial_margin_pct": self.initial_margin_pct,
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
