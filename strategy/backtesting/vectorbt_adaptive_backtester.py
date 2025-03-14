"""
Adaptive backtesting engine using vectorbt.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import vectorbt as vbt

# Removed the Direction import since it's causing issues


class VectorbtAdaptiveBacktester:
    """
    Class for backtesting a strategy that can dynamically enter and exit positions,
    implemented using vectorbt for better performance.
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
        """Run the backtest using vectorbt and return results with adaptive logic."""
        if len(self.data) < 2:
            print("Error: Insufficient data points for backtest")
            return None

        # Extract price and funding data
        df = self.data.copy()

        # Determine timeframe multiplier for funding
        if hasattr(self.strategy, "funding_periods_multiplier"):
            funding_multiplier = self.strategy.funding_periods_multiplier
        else:
            # Default to 1 if not specified
            funding_multiplier = 1

        # Generate entry and exit signals based on funding rate sign
        entries = df["funding_rate"] > 0
        exits = df["funding_rate"] <= 0

        # If first entry is a negative funding rate, fix the entries and exits
        # to avoid starting with an exit signal
        if not entries.iloc[0]:
            entries.iloc[0] = False

        # Create vectorbt signals
        entry_signals = entries.to_numpy()
        exit_signals = exits.to_numpy()

        # Create portfolio with long spot position
        spot_pf = vbt.Portfolio.from_signals(
            df["spot_close"],
            entries=entry_signals,
            exits=exit_signals,
            init_cash=self.initial_capital * self.leverage / 2,
            fees=self.fee_rate,
            freq="D",  # Assuming daily data, adjust if different
        )

        # Create portfolio with short perp position (use inverse signals for short)
        # For shorts, we use inverse entry/exit logic with vectorbt
        short_entries = entries.to_numpy()
        short_exits = exits.to_numpy()

        # Use string 'short' instead of Direction enum
        perp_pf = vbt.Portfolio.from_signals(
            df["perp_close"],
            entries=short_entries,
            exits=short_exits,
            init_cash=self.initial_capital * self.leverage / 2,
            fees=self.fee_rate,
            freq="D",  # Assuming daily data
            direction="short",  # Use string 'short' instead of enum
        )

        # Calculate funding income when in position
        funding_rates = df["funding_rate"].to_numpy() * funding_multiplier

        # Get position status (1 when in position, 0 when out)
        position_status = np.zeros(len(df))

        # Use vectorbt position status to determine when we're in the market
        active_position_mask = (
            spot_pf.position_mask
        )  # This is True when position is active
        position_status[active_position_mask] = 1

        # Calculate perp value at each time point (for funding calculation)
        perp_values = df["perp_close"].to_numpy() * perp_pf.position_size.to_numpy()

        # Calculate funding payments (only applied when position is active)
        funding_payments = np.zeros(len(df))
        funding_payments[active_position_mask] = (
            perp_values[active_position_mask] * funding_rates[active_position_mask]
        )

        # Accumulate funding payments
        cumulative_funding = np.cumsum(funding_payments)

        # Calculate total equity curve (spot + perp + funding)
        # We need to adjust for initial cash to avoid double counting
        equity_curve = (
            spot_pf.equity + perp_pf.equity - self.initial_capital + cumulative_funding
        )

        # Extract trade information
        trades = []

        # Extract spot trades
        spot_trades = spot_pf.trades.records_readable
        perp_trades = perp_pf.trades.records_readable

        # Zip together spot and perp trades to create complete trade records
        if not spot_trades.empty and not perp_trades.empty:
            for i in range(min(len(spot_trades), len(perp_trades))):
                spot_trade = spot_trades.iloc[i]
                perp_trade = perp_trades.iloc[i]

                # Create trade record
                trade = {
                    "entry_date": spot_trade["Entry Date"],
                    "exit_date": spot_trade["Exit Date"],
                    "duration": (
                        spot_trade["Exit Date"] - spot_trade["Entry Date"]
                    ).days,
                    "entry_funding": (
                        df.loc[
                            df["Timestamp"] == spot_trade["Entry Date"], "funding_rate"
                        ].values[0]
                        if len(
                            df.loc[
                                df["Timestamp"] == spot_trade["Entry Date"],
                                "funding_rate",
                            ].values
                        )
                        > 0
                        else 0
                    ),
                    "exit_funding": (
                        df.loc[
                            df["Timestamp"] == spot_trade["Exit Date"], "funding_rate"
                        ].values[0]
                        if len(
                            df.loc[
                                df["Timestamp"] == spot_trade["Exit Date"],
                                "funding_rate",
                            ].values
                        )
                        > 0
                        else 0
                    ),
                    "spot_entry": spot_trade["Entry Price"],
                    "perp_entry": perp_trade["Entry Price"],
                    "spot_exit": spot_trade["Exit Price"],
                    "perp_exit": perp_trade["Exit Price"],
                    "spot_pnl": spot_trade["PnL"],
                    "perp_pnl": perp_trade["PnL"],
                    "net_pnl": spot_trade["PnL"] + perp_trade["PnL"],
                    "fees": spot_trade["Fees"] + perp_trade["Fees"],
                }
                trades.append(trade)

        # Calculate trade statistics
        if trades:
            winning_trades = [t for t in trades if t["net_pnl"] > 0]
            losing_trades = [t for t in trades if t["net_pnl"] <= 0]

            total_profit = sum(t["net_pnl"] for t in trades)
            total_fees = sum(t["fees"] for t in trades)

            avg_profit = (
                np.mean([t["net_pnl"] for t in winning_trades]) if winning_trades else 0
            )
            avg_loss = (
                np.mean([t["net_pnl"] for t in losing_trades]) if losing_trades else 0
            )

            avg_duration = np.mean([t["duration"] for t in trades])

            trade_stats = {
                "total_trades": len(trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": len(winning_trades) / len(trades) if trades else 0,
                "average_profit": float(avg_profit),
                "average_loss": float(avg_loss),
                "total_profit": float(total_profit),
                "total_fees": float(total_fees),
                "average_duration": float(avg_duration),
            }
        else:
            trade_stats = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "average_profit": 0,
                "average_loss": 0,
                "total_profit": 0,
                "total_fees": 0,
                "average_duration": 0,
            }

        # Calculate PnL components over time
        spot_pnl = spot_pf.asset_value - spot_pf.init_cash
        perp_pnl = perp_pf.asset_value - perp_pf.init_cash
        net_market_pnl = spot_pnl + perp_pnl

        # Prepare notional values
        spot_notional = np.abs(spot_pf.asset_value.to_numpy())
        perp_notional = np.abs(perp_pf.asset_value.to_numpy())
        total_notional = spot_notional + perp_notional

        # Calculate annualized funding rates
        annualized_funding_rates = (
            funding_rates * 3 * 365 * 100
        )  # Convert to percentage

        # Store results in the same format as the original backtester
        self.results = {
            "entry_date": df["Timestamp"].iloc[0],
            "exit_date": df["Timestamp"].iloc[-1],
            "initial_capital": self.initial_capital,
            "final_capital": float(equity_curve.iloc[-1]),
            "entry_fee": float(total_fees / 2 if trades else 0),  # Estimate
            "exit_fee": float(total_fees / 2 if trades else 0),  # Estimate
            "spot_quantity": 0,  # Not meaningful for adaptive strategy
            "perp_quantity": 0,  # Not meaningful for adaptive strategy
            "initial_notional": float(total_notional[0]),
            "final_notional": float(total_notional[-1]),
            "dates": df["Timestamp"].tolist(),
            "equity_curve": equity_curve.tolist(),
            "funding_income": funding_payments.tolist(),
            "cumulative_funding": cumulative_funding.tolist(),
            "spot_pnl": spot_pnl.to_numpy().tolist(),
            "perp_pnl": perp_pnl.to_numpy().tolist(),
            "total_pnl": (net_market_pnl + cumulative_funding).to_numpy().tolist(),
            "net_market_pnl": net_market_pnl.to_numpy().tolist(),
            "notional_values": total_notional.tolist(),
            "funding_rates": df["funding_rate"].tolist(),
            "annualized_funding_rates": annualized_funding_rates.tolist(),
            "position_status": position_status.tolist(),
            "data": df,
            "trades": trades,
            "trade_stats": trade_stats,
            "vbt_portfolio_spot": spot_pf,
            "vbt_portfolio_perp": perp_pf,
        }

        return self.results
