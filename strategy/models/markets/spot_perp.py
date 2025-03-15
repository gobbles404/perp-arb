"""
Spot-Perpetual Market Structure Implementation.

This class represents the market structure where a strategy trades
spot and perpetual futures contracts simultaneously.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class SpotPerpMarket:
    """
    Market structure for spot and perpetual futures trading.

    This market structure implements a standard spot-perp trading setup where:
    - Long spot positions are balanced against short perpetual futures
    - Capital is allocated between the two instruments
    - Leverage is applied within exchange constraints
    - Funding payments are received/paid on the perpetual position
    """

    def __init__(
        self,
        data: pd.DataFrame,
        capital: float,
        leverage: float,
        fee_rate: float,
        allocation: str = "50-50",
        enforce_margin_limits: bool = False,
        **kwargs,
    ):
        """
        Initialize the spot-perp market structure.

        Args:
            data: Market data DataFrame
            capital: Initial capital
            leverage: Leverage multiplier
            fee_rate: Trading fee rate
            allocation: Capital allocation between spot-perp (e.g., "60-40")
            enforce_margin_limits: Whether to enforce exchange margin limits
            **kwargs: Additional keyword arguments
        """
        self.data = data
        self.capital = capital
        self.leverage = leverage
        self.fee_rate = fee_rate

        # Parse allocation
        self.allocation_str = allocation
        self.allocation = self._parse_allocation(allocation)

        # Leverage settings
        self.enforce_margin_limits = enforce_margin_limits
        self.effective_leverage = self._calculate_effective_leverage()

        # Initialize position tracking
        self.spot_quantity = 0
        self.perp_quantity = 0
        self.spot_entry_price = 0
        self.perp_entry_price = 0
        self.is_position_open = False

        # Funding multiplier (set by strategy)
        self.funding_periods_multiplier = 1

    def _parse_allocation(self, allocation_str: str) -> List[float]:
        """
        Parse allocation string into spot and perp percentages.

        Args:
            allocation_str: String like "60-40" for 60% spot, 40% perp

        Returns:
            List of allocation percentages [spot_pct, perp_pct]
        """
        try:
            parts = allocation_str.split("-")
            if len(parts) != 2:
                print(
                    f"Warning: Invalid allocation format '{allocation_str}'. Using 50-50."
                )
                return [0.5, 0.5]

            spot_pct = float(parts[0]) / 100
            perp_pct = float(parts[1]) / 100

            # Validate allocations sum to 100%
            if abs(spot_pct + perp_pct - 1.0) > 0.001:
                print(
                    f"Warning: Allocation '{allocation_str}' doesn't sum to 100%. Normalizing."
                )
                total = spot_pct + perp_pct
                spot_pct /= total
                perp_pct /= total

            return [spot_pct, perp_pct]
        except Exception as e:
            print(f"Error parsing allocation '{allocation_str}': {e}. Using 50-50.")
            return [0.5, 0.5]

    def _calculate_effective_leverage(self) -> float:
        """
        Calculate effective leverage considering exchange limits if enabled.

        Returns:
            Effective leverage to use (may be lower than requested)
        """
        effective_leverage = self.leverage

        if self.enforce_margin_limits and "perp_max_leverage" in self.data.columns:
            # Find the maximum leverage limit in first data point
            first_row = self.data.iloc[0]
            if "perp_max_leverage" in first_row and first_row["perp_max_leverage"] > 0:
                max_leverage = first_row["perp_max_leverage"]

                if effective_leverage > max_leverage:
                    print(
                        f"Requested leverage {effective_leverage}x exceeds exchange limit "
                        f"{max_leverage}x. Using maximum allowed."
                    )
                    effective_leverage = max_leverage

        return effective_leverage

    def calculate_position_sizes(self, data_row: pd.Series) -> Dict[str, float]:
        """
        Calculate position sizes for spot and perpetual futures.

        Args:
            data_row: Current market data row

        Returns:
            Dictionary with position sizes and notional values
        """
        spot_price = data_row["spot_close"]
        perp_price = data_row["perp_close"]

        if spot_price <= 0 or perp_price <= 0:
            print("Warning: Invalid prices detected. Cannot calculate position sizes.")
            return {
                "spot_quantity": 0,
                "perp_quantity": 0,
                "spot_notional": 0,
                "perp_notional": 0,
                "total_notional": 0,
            }

        # Calculate notional values
        total_notional = self.capital * self.effective_leverage
        spot_notional = total_notional * self.allocation[0]
        perp_notional = total_notional * self.allocation[1]

        # Calculate quantities
        spot_quantity = spot_notional / spot_price
        perp_quantity = perp_notional / perp_price

        return {
            "spot_quantity": spot_quantity,
            "perp_quantity": perp_quantity,
            "spot_notional": spot_notional,
            "perp_notional": perp_notional,
            "total_notional": spot_notional + perp_notional,
        }

    def initialize_position(
        self, data_row: pd.Series, should_enter: bool = True
    ) -> Dict[str, Any]:
        """
        Initialize a position in the spot-perp market.

        Args:
            data_row: Current market data row
            should_enter: Whether to actually enter the position (based on strategy signals)

        Returns:
            Dictionary with position details
        """
        # If strategy signals don't indicate entry, return empty position
        if not should_enter:
            return self._empty_position(data_row)

        # Calculate position sizes
        sizes = self.calculate_position_sizes(data_row)

        spot_quantity = sizes["spot_quantity"]
        perp_quantity = sizes["perp_quantity"]
        spot_notional = sizes["spot_notional"]
        perp_notional = sizes["perp_notional"]
        total_notional = sizes["total_notional"]

        # Calculate fees
        entry_fee = total_notional * self.fee_rate

        # Store position details
        self.spot_quantity = spot_quantity
        self.perp_quantity = perp_quantity
        self.spot_entry_price = data_row["spot_close"]
        self.perp_entry_price = data_row["perp_close"]
        self.is_position_open = True

        # Create position record
        position = {
            "entry_date": data_row["Timestamp"],
            "spot_entry": self.spot_entry_price,
            "perp_entry": self.perp_entry_price,
            "spot_quantity": spot_quantity,
            "perp_quantity": perp_quantity,
            "capital": self.capital - entry_fee,
            "entry_fee": entry_fee,
            "spot_notional": spot_notional,
            "perp_notional": perp_notional,
            "total_notional": total_notional,
            "allocation": self.allocation_str,
            "effective_leverage": self.effective_leverage,
        }

        return position

    def _empty_position(self, data_row: pd.Series) -> Dict[str, Any]:
        """
        Return an empty position structure.

        Args:
            data_row: Current market data row

        Returns:
            Empty position dictionary
        """
        return {
            "entry_date": data_row["Timestamp"],
            "spot_entry": data_row["spot_close"],
            "perp_entry": data_row["perp_close"],
            "spot_quantity": 0,
            "perp_quantity": 0,
            "capital": self.capital,
            "entry_fee": 0,
            "spot_notional": 0,
            "perp_notional": 0,
            "total_notional": 0,
            "allocation": self.allocation_str,
            "effective_leverage": self.effective_leverage,
        }

    def calculate_pnl(self, data_row: pd.Series) -> Dict[str, Any]:
        """
        Calculate PnL for the current position.

        Args:
            data_row: Current market data row

        Returns:
            Dictionary with PnL components
        """
        # Initialize result
        result = {
            "date": data_row["Timestamp"],
            "spot_pnl": 0,
            "perp_pnl": 0,
            "funding_payment": 0,
            "funding_rate": data_row["funding_rate"],
            "total_notional": 0,
        }

        # If no position is open, return empty result
        if not self.is_position_open:
            return result

        # Get current prices
        spot_price = data_row["spot_close"]
        perp_price = data_row["perp_close"]
        funding_rate = data_row["funding_rate"]

        # Spot PnL (long position)
        spot_pnl = self.spot_quantity * (spot_price - self.spot_entry_price)

        # Perp PnL (short position)
        perp_pnl = self.perp_quantity * (self.perp_entry_price - perp_price)

        # Calculate funding payment
        funding_payment = (
            self.perp_quantity
            * perp_price
            * funding_rate
            * self.funding_periods_multiplier
        )

        # Calculate current notional values
        spot_notional = self.spot_quantity * spot_price
        perp_notional = self.perp_quantity * perp_price
        total_notional = spot_notional + perp_notional

        # Update result
        result.update(
            {
                "spot_pnl": spot_pnl,
                "perp_pnl": perp_pnl,
                "funding_payment": funding_payment,
                "spot_notional": spot_notional,
                "perp_notional": perp_notional,
                "total_notional": total_notional,
            }
        )

        return result

    def close_position(self, data_row: pd.Series) -> Dict[str, Any]:
        """
        Close the current position.

        Args:
            data_row: Current market data row

        Returns:
            Dictionary with position exit details
        """
        # If no position is open, return empty result
        if not self.is_position_open:
            return {
                "exit_date": data_row["Timestamp"],
                "exit_spot": data_row["spot_close"],
                "exit_perp": data_row["perp_close"],
                "exit_fee": 0,
                "final_spot_notional": 0,
                "final_perp_notional": 0,
                "final_total_notional": 0,
            }

        # Get exit prices
        exit_spot = data_row["spot_close"]
        exit_perp = data_row["perp_close"]

        # Calculate final notional values
        final_spot_notional = self.spot_quantity * exit_spot
        final_perp_notional = self.perp_quantity * exit_perp
        final_total_notional = final_spot_notional + final_perp_notional

        # Calculate exit fee
        exit_fee = final_total_notional * self.fee_rate

        # Calculate PnLs
        spot_pnl = self.spot_quantity * (exit_spot - self.spot_entry_price)
        perp_pnl = self.perp_quantity * (self.perp_entry_price - exit_perp)
        net_pnl = spot_pnl + perp_pnl

        # Save position state before clearing
        position_state = {
            "spot_quantity": self.spot_quantity,
            "perp_quantity": self.perp_quantity,
            "spot_entry_price": self.spot_entry_price,
            "perp_entry_price": self.perp_entry_price,
        }

        # Reset position state
        self.spot_quantity = 0
        self.perp_quantity = 0
        self.spot_entry_price = 0
        self.perp_entry_price = 0
        self.is_position_open = False

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
            "position": position_state,
        }

    def set_funding_periods_multiplier(self, multiplier: float) -> None:
        """
        Set the funding periods multiplier for funding payment calculations.

        Args:
            multiplier: Multiplier to apply to funding rates
        """
        self.funding_periods_multiplier = multiplier

    def get_market_info(self) -> Dict[str, Any]:
        """
        Get information about the current market setup.

        Returns:
            Dictionary with market structure information
        """
        return {
            "market_type": "spot-perp",
            "allocation": self.allocation_str,
            "leverage": self.leverage,
            "effective_leverage": self.effective_leverage,
            "capital": self.capital,
            "is_position_open": self.is_position_open,
        }
