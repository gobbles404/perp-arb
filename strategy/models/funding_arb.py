"""
Core funding arbitrage strategy implementation.
"""


class FundingArbStrategy:
    """
    Simple long spot, short perp strategy that harvests funding rate.
    """

    def __init__(self):
        """Initialize the strategy."""
        self.position = None
        self.metrics = {}

    def initialize_position(self, data, capital, leverage, fee_rate):
        """Initialize trading position with given parameters."""
        # Get first data point
        entry_date = data["Timestamp"].iloc[0]
        spot_entry = data["spot_close"].iloc[0]
        perp_entry = data["perp_close"].iloc[0]

        # Calculate position size (in USD value)
        position_size = capital * leverage / 2  # Split capital between long and short

        # Calculate position quantities
        spot_quantity = position_size / spot_entry
        perp_quantity = position_size / perp_entry

        # Calculate notional values
        long_notional = spot_quantity * spot_entry
        short_notional = perp_quantity * perp_entry
        total_notional = long_notional + short_notional

        # Calculate entry fee
        entry_fee = position_size * 2 * fee_rate

        # Create position object
        self.position = {
            "entry_date": entry_date,
            "spot_entry": spot_entry,
            "perp_entry": perp_entry,
            "spot_quantity": spot_quantity,
            "perp_quantity": perp_quantity,
            "long_notional": long_notional,
            "short_notional": short_notional,
            "total_notional": total_notional,
            "entry_fee": entry_fee,
            "initial_capital": capital,
            "capital": capital - entry_fee,
        }

        # Print position details
        print(f"Entering position on {entry_date}:")
        print(
            f"  Long {spot_quantity:.6f} BTC at ${spot_entry:.2f} (${long_notional:.2f})"
        )
        print(
            f"  Short {perp_quantity:.6f} BTC-PERP at ${perp_entry:.2f} (${short_notional:.2f})"
        )
        print(f"  Total Notional: ${total_notional:.2f}")
        print(f"  Entry fee: ${entry_fee:.2f}")

        return self.position

    def calculate_pnl(self, data_row):
        """Calculate PnL metrics for a single data point."""
        if self.position is None:
            raise ValueError("Position not initialized")

        current_date = data_row["Timestamp"]
        current_spot = data_row["spot_close"]
        current_perp = data_row["perp_close"]
        funding_rate = data_row["funding_rate"]

        # Calculate spot PnL
        spot_pnl = self.position["spot_quantity"] * (
            current_spot - self.position["spot_entry"]
        )

        # Calculate perp PnL
        perp_pnl = self.position["perp_quantity"] * (
            self.position["perp_entry"] - current_perp
        )

        # Calculate net market PnL
        net_market_pnl = spot_pnl + perp_pnl

        # Calculate funding payment
        funding_payment = self.position["perp_quantity"] * current_perp * funding_rate

        # Calculate notional values
        current_spot_notional = self.position["spot_quantity"] * current_spot
        current_perp_notional = self.position["perp_quantity"] * current_perp
        current_total_notional = current_spot_notional + current_perp_notional

        # Create result object
        result = {
            "date": current_date,
            "spot_close": current_spot,
            "perp_close": current_perp,
            "funding_rate": funding_rate,
            "spot_pnl": spot_pnl,
            "perp_pnl": perp_pnl,
            "net_market_pnl": net_market_pnl,
            "funding_payment": funding_payment,
            "spot_notional": current_spot_notional,
            "perp_notional": current_perp_notional,
            "total_notional": current_total_notional,
        }

        return result

    def close_position(self, data_row, fee_rate):
        """Close the position and calculate final metrics."""
        if self.position is None:
            raise ValueError("Position not initialized")

        exit_date = data_row["Timestamp"]
        exit_spot = data_row["spot_close"]
        exit_perp = data_row["perp_close"]

        # Calculate final values
        final_spot_notional = self.position["spot_quantity"] * exit_spot
        final_perp_notional = self.position["perp_quantity"] * exit_perp
        final_total_notional = final_spot_notional + final_perp_notional

        # Calculate exit fee
        exit_fee = (final_spot_notional + final_perp_notional) * fee_rate

        # Calculate final PnL values
        spot_pnl = self.position["spot_quantity"] * (
            exit_spot - self.position["spot_entry"]
        )
        perp_pnl = self.position["perp_quantity"] * (
            self.position["perp_entry"] - exit_perp
        )

        # Create exit data
        exit_data = {
            "exit_date": exit_date,
            "exit_spot": exit_spot,
            "exit_perp": exit_perp,
            "final_spot_notional": final_spot_notional,
            "final_perp_notional": final_perp_notional,
            "final_total_notional": final_total_notional,
            "exit_fee": exit_fee,
            "spot_pnl": spot_pnl,
            "perp_pnl": perp_pnl,
        }

        print(f"\nExiting position on {exit_date}:")
        print(
            f"  Sell {self.position['spot_quantity']:.6f} BTC at ${exit_spot:.2f} (${final_spot_notional:.2f})"
        )
        print(
            f"  Cover {self.position['perp_quantity']:.6f} BTC-PERP at ${exit_perp:.2f} (${final_perp_notional:.2f})"
        )
        print(f"  Final Notional: ${final_total_notional:.2f}")
        print(f"  Exit fee: ${exit_fee:.2f}")

        return exit_data
