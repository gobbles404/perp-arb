"""
Adaptive backtesting engine using vectorbt.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import vectorbt as vbt


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

        # Use -1 for direction instead of "short" string
        perp_pf = vbt.Portfolio.from_signals(
            df["perp_close"],
            entries=short_entries,
            exits=short_exits,
            init_cash=self.initial_capital * self.leverage / 2,
            fees=self.fee_rate,
            freq="D",  # Assuming daily data
            direction=-1,  # Use -1 for short position
        )

        # Calculate funding income when in position
        funding_rates = df["funding_rate"].to_numpy() * funding_multiplier

        # Get position status (1 when in position, 0 when out)
        position_status = np.zeros(len(df))

        # Handle position_mask correctly based on if it's a function or property
        if hasattr(spot_pf, "position_mask"):
            if callable(spot_pf.position_mask):
                active_position_mask = spot_pf.position_mask().to_numpy()
            else:
                active_position_mask = spot_pf.position_mask.to_numpy()
        else:
            # Fallback: create a mask from entries and exits manually
            active_position_mask = np.zeros(len(df), dtype=bool)
            current_position = False
            for i in range(len(df)):
                if entry_signals[i] and not current_position:
                    current_position = True
                elif exit_signals[i] and current_position:
                    current_position = False
                active_position_mask[i] = current_position

        position_status[active_position_mask] = 1

        # Calculate perp value at each time point for funding calculation
        # We'll use the asset value divided by price to estimate position size
        # This is a workaround for missing position_size attribute

        # Calculate estimated perp position sizes
        perp_prices = df["perp_close"].to_numpy()
        perp_position_sizes = np.zeros(len(df))

        # Estimate position size based on asset value and price
        if hasattr(perp_pf, "asset_value"):
            if callable(perp_pf.asset_value):
                perp_asset_values = perp_pf.asset_value().to_numpy()
            else:
                perp_asset_values = perp_pf.asset_value.to_numpy()

            # Estimate position size as asset value / price (when we're in position)
            for i in range(len(df)):
                if active_position_mask[i] and perp_prices[i] > 0:
                    # For short positions, we expect negative asset values
                    # So we divide by price to get quantity (negative for shorts)
                    perp_position_sizes[i] = perp_asset_values[i] / perp_prices[i]
        else:
            # If we can't get asset values, estimate based on cash changes
            # This is a rough estimate but should be close enough for funding calculations
            perp_position_sizes[active_position_mask] = (
                -(self.initial_capital * self.leverage / 2)
                / perp_prices[active_position_mask]
            )

        # Calculate funding payments (only applied when position is active)
        funding_payments = np.zeros(len(df))
        for i in range(len(df)):
            if active_position_mask[i]:
                # Funding payment = perp position size * perp price * funding rate
                funding_payments[i] = (
                    perp_position_sizes[i] * perp_prices[i] * funding_rates[i]
                )

        # Accumulate funding payments
        cumulative_funding = np.cumsum(funding_payments)

        # Calculate PnL components over time - handle as function or property correctly
        if hasattr(spot_pf, "asset_value"):
            if callable(spot_pf.asset_value):
                spot_asset_value = spot_pf.asset_value()
            else:
                spot_asset_value = spot_pf.asset_value

            if callable(perp_pf.asset_value):
                perp_asset_value = perp_pf.asset_value()
            else:
                perp_asset_value = perp_pf.asset_value

            spot_pnl = spot_asset_value - spot_pf.init_cash
            perp_pnl = perp_asset_value - perp_pf.init_cash
            net_market_pnl = spot_pnl + perp_pnl
        else:
            # Fallback if asset_value isn't available - use cash changes
            spot_pnl = pd.Series(np.zeros(len(df)))
            perp_pnl = pd.Series(np.zeros(len(df)))
            net_market_pnl = pd.Series(np.zeros(len(df)))

        # Calculate equity curve manually
        # First check if equity is available
        if hasattr(spot_pf, "equity") and hasattr(perp_pf, "equity"):
            # Use existing equity attribute
            if callable(spot_pf.equity):
                spot_equity = spot_pf.equity()
            else:
                spot_equity = spot_pf.equity

            if callable(perp_pf.equity):
                perp_equity = perp_pf.equity()
            else:
                perp_equity = perp_pf.equity

            # Combined equity + funding
            equity_curve = (
                spot_equity + perp_equity - self.initial_capital + cumulative_funding
            )
        else:
            # Need to calculate equity manually
            # Check if cash is available
            if (
                hasattr(spot_pf, "cash")
                and hasattr(spot_pf, "asset_value")
                and hasattr(perp_pf, "cash")
                and hasattr(perp_pf, "asset_value")
            ):
                # Get cash values
                if callable(spot_pf.cash):
                    spot_cash = spot_pf.cash()
                else:
                    spot_cash = spot_pf.cash

                if callable(perp_pf.cash):
                    perp_cash = perp_pf.cash()
                else:
                    perp_cash = perp_pf.cash

                # Equity = Cash + Asset Value
                spot_equity = spot_cash + spot_asset_value
                perp_equity = perp_cash + perp_asset_value

                # Combined equity + funding
                equity_curve = (
                    spot_equity
                    + perp_equity
                    - self.initial_capital
                    + cumulative_funding
                )
            else:
                # Last resort: estimate equity from PnL + initial capital
                estimated_market_pnl = np.zeros(len(df))
                for i in range(len(df)):
                    if active_position_mask[i]:
                        # For long spot position
                        spot_change = (
                            df["spot_close"].iloc[i] / df["spot_close"].iloc[0] - 1
                        )
                        # For short perp position (negative PnL when price goes up)
                        perp_change = -(
                            df["perp_close"].iloc[i] / df["perp_close"].iloc[0] - 1
                        )
                        # Combined PnL as percentage of initial position
                        estimated_market_pnl[i] = (
                            (spot_change + perp_change)
                            * self.initial_capital
                            * self.leverage
                            / 2
                        )

                # Create a pandas Series for the equity curve
                equity_curve = pd.Series(
                    self.initial_capital + estimated_market_pnl + cumulative_funding
                )

        # Prepare notional values from the estimated position sizes
        spot_prices = df["spot_close"].to_numpy()
        spot_position_sizes = np.zeros(len(df))

        # Estimate spot position size based on asset value and price (similar to perp)
        if hasattr(spot_pf, "asset_value"):
            if callable(spot_pf.asset_value):
                spot_asset_values = spot_pf.asset_value().to_numpy()
            else:
                spot_asset_values = spot_pf.asset_value.to_numpy()

            # Estimate position size as asset value / price (when we're in position)
            for i in range(len(df)):
                if active_position_mask[i] and spot_prices[i] > 0:
                    spot_position_sizes[i] = spot_asset_values[i] / spot_prices[i]
        else:
            # If we can't get asset values, estimate based on the initial investment
            spot_position_sizes[active_position_mask] = (
                self.initial_capital * self.leverage / 2
            ) / spot_prices[active_position_mask]

        # Calculate notional values
        spot_notional = np.abs(spot_position_sizes * spot_prices)
        perp_notional = np.abs(perp_position_sizes * perp_prices)
        total_notional = spot_notional + perp_notional

        # Calculate annualized funding rates
        annualized_funding_rates = (
            funding_rates * 3 * 365 * 100
        )  # Convert to percentage

        # Extract trade information with better error handling
        trades = []

        # Helper function to safely get trades from portfolio
        def get_trades_from_portfolio(portfolio):
            try:
                if hasattr(portfolio, "trades"):
                    if hasattr(portfolio.trades, "records_readable"):
                        return portfolio.trades.records_readable
                    elif hasattr(portfolio.trades, "records"):
                        return portfolio.trades.records
                return pd.DataFrame()  # Empty DataFrame if nothing found
            except Exception as e:
                print(f"Warning: Could not extract trades: {e}")
                return pd.DataFrame()

        # Get trades records from both portfolios
        spot_trades = get_trades_from_portfolio(spot_pf)
        perp_trades = get_trades_from_portfolio(perp_pf)

        # Calculate total fees from trades
        total_fees = 0

        # Now process trades with maximum flexibility for column names
        if not spot_trades.empty and not perp_trades.empty:
            # Print column names to debug
            print("Spot trades columns:", spot_trades.columns.tolist())

            # Helper function to safely get a column value with multiple possible names
            def get_column_value(df_row, possible_names, default=None):
                for name in possible_names:
                    if name in df_row:
                        return df_row[name]
                return default

            # Determine column names for entry/exit dates
            entry_date_cols = ["Entry Date", "entry_date", "entry_idx", "entry_time"]
            exit_date_cols = ["Exit Date", "exit_date", "exit_idx", "exit_time"]

            # Determine column names for prices
            entry_price_cols = ["Entry Price", "entry_price", "entry_val"]
            exit_price_cols = ["Exit Price", "exit_price", "exit_val"]

            # Determine column names for PnL and fees
            pnl_cols = ["PnL", "pnl", "return"]
            fees_cols = ["Fees", "fees", "fee"]

            # Try to calculate total fees
            for i in range(min(len(spot_trades), len(perp_trades))):
                try:
                    spot_fee = get_column_value(spot_trades.iloc[i], fees_cols, 0)
                    perp_fee = get_column_value(perp_trades.iloc[i], fees_cols, 0)
                    total_fees += spot_fee + perp_fee
                except Exception as e:
                    print(f"Warning: Error calculating fees: {e}")

            # Process each pair of trades
            for i in range(min(len(spot_trades), len(perp_trades))):
                spot_row = spot_trades.iloc[i]
                perp_row = perp_trades.iloc[i]

                try:
                    # Get entry and exit dates
                    entry_date = get_column_value(spot_row, entry_date_cols)
                    exit_date = get_column_value(spot_row, exit_date_cols)

                    # Handle index-based dates
                    if isinstance(entry_date, (int, np.integer)):
                        entry_date = df.iloc[entry_date]["Timestamp"]
                    if isinstance(exit_date, (int, np.integer)):
                        exit_date = df.iloc[exit_date]["Timestamp"]

                    # Calculate duration
                    try:
                        duration = (exit_date - entry_date).days
                    except:
                        duration = 0

                    # Get entry and exit prices
                    spot_entry_price = get_column_value(spot_row, entry_price_cols, 0)
                    spot_exit_price = get_column_value(spot_row, exit_price_cols, 0)
                    perp_entry_price = get_column_value(perp_row, entry_price_cols, 0)
                    perp_exit_price = get_column_value(perp_row, exit_price_cols, 0)

                    # Get PnL
                    spot_pnl_val = get_column_value(spot_row, pnl_cols, 0)
                    perp_pnl_val = get_column_value(perp_row, pnl_cols, 0)

                    # Get fees
                    spot_fees = get_column_value(spot_row, fees_cols, 0)
                    perp_fees = get_column_value(perp_row, fees_cols, 0)

                    # Try to find funding rates at entry and exit
                    try:
                        # Safe way to find entry funding rate
                        if isinstance(entry_date, pd.Timestamp):
                            entry_date_str = entry_date.strftime("%Y-%m-%d")
                            entry_matches = (
                                df["Timestamp"].dt.strftime("%Y-%m-%d")
                                == entry_date_str
                            )
                        else:
                            entry_matches = df["Timestamp"] == entry_date

                        if sum(entry_matches) > 0:
                            entry_funding = df.loc[entry_matches, "funding_rate"].iloc[
                                0
                            ]
                        else:
                            entry_funding = 0

                        # Safe way to find exit funding rate
                        if isinstance(exit_date, pd.Timestamp):
                            exit_date_str = exit_date.strftime("%Y-%m-%d")
                            exit_matches = (
                                df["Timestamp"].dt.strftime("%Y-%m-%d") == exit_date_str
                            )
                        else:
                            exit_matches = df["Timestamp"] == exit_date

                        if sum(exit_matches) > 0:
                            exit_funding = df.loc[exit_matches, "funding_rate"].iloc[0]
                        else:
                            exit_funding = 0
                    except Exception as e:
                        print(f"Warning: Could not extract funding rates: {e}")
                        entry_funding = 0
                        exit_funding = 0

                    # Create trade record
                    trade = {
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "duration": duration,
                        "entry_funding": entry_funding,
                        "exit_funding": exit_funding,
                        "spot_entry": spot_entry_price,
                        "perp_entry": perp_entry_price,
                        "spot_exit": spot_exit_price,
                        "perp_exit": perp_exit_price,
                        "spot_pnl": spot_pnl_val,
                        "perp_pnl": perp_pnl_val,
                        "net_pnl": spot_pnl_val + perp_pnl_val,
                        "fees": spot_fees + perp_fees,
                    }
                    trades.append(trade)
                except Exception as e:
                    print(f"Warning: Could not process trade {i}: {e}")
                    continue

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

        # Ensure equity_curve is a pandas Series for compatibility
        if not isinstance(equity_curve, pd.Series):
            equity_curve = pd.Series(equity_curve)

        # Convert Series to lists for results
        equity_list = equity_curve.tolist()

        # Make sure pandas Series are converted to numpy arrays before using tolist()
        if isinstance(spot_pnl, pd.Series):
            spot_pnl_list = spot_pnl.to_numpy().tolist()
        else:
            spot_pnl_list = spot_pnl.tolist()

        if isinstance(perp_pnl, pd.Series):
            perp_pnl_list = perp_pnl.to_numpy().tolist()
        else:
            perp_pnl_list = perp_pnl.tolist()

        if isinstance(net_market_pnl, pd.Series):
            net_market_pnl_list = net_market_pnl.to_numpy().tolist()
            total_pnl_list = (
                (net_market_pnl + pd.Series(cumulative_funding)).to_numpy().tolist()
            )
        else:
            net_market_pnl_list = net_market_pnl.tolist()
            total_pnl_list = (net_market_pnl + cumulative_funding).tolist()

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
            "equity_curve": equity_list,
            "funding_income": funding_payments.tolist(),
            "cumulative_funding": cumulative_funding.tolist(),
            "spot_pnl": spot_pnl_list,
            "perp_pnl": perp_pnl_list,
            "total_pnl": total_pnl_list,
            "net_market_pnl": net_market_pnl_list,
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
