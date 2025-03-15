"""
Spot-Perpetual Market Structure with Capital-Efficient Leverage.

This class represents the market structure where a strategy trades
spot and perpetual futures contracts with delta neutrality, applying
leverage only to the perpetual futures side for capital efficiency.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class SpotPerpMarket:
    """
    Market structure for spot and perpetual futures trading with capital-efficient leverage.

    This market structure implements a delta-neutral spot-perp trading setup where:
    - Long spot positions are balanced against short perpetual futures
    - Leverage is applied only to the perpetual futures side
    - Capital is allocated efficiently to maintain delta neutrality
    - Funding payments are received/paid on the leveraged perpetual position
    """

    def __init__(
        self,
        data: pd.DataFrame,
        capital: float,
        leverage: float,
        fee_rate: float,
        allocation: str = "50-50",  # Legacy parameter, no longer used
        enforce_margin_limits: bool = True,
        **kwargs,
    ):
        """
        Initialize the spot-perp market structure.

        Args:
            data: Market data DataFrame
            capital: Initial capital
            leverage: Leverage multiplier (applied to perpetual futures only)
            fee_rate: Trading fee rate
            allocation: Legacy parameter, not used in capital-efficient mode
            enforce_margin_limits: Whether to enforce exchange margin limits
            **kwargs: Additional keyword arguments
        """
        self.data = data
        self.capital = capital
        self.requested_leverage = leverage  # Store original requested leverage
        self.fee_rate = fee_rate

        # Extract and store margin requirements
        self.initial_margin_pct = self._extract_initial_margin()
        self.maint_margin_pct = self._extract_maint_margin()
        self.liquidation_fee_pct = self._extract_liquidation_fee()
        self.max_leverage = self._calculate_max_leverage()

        # Validate requested leverage against exchange limits
        self.enforce_margin_limits = enforce_margin_limits
        if self.enforce_margin_limits and leverage > self.max_leverage:
            raise ValueError(
                f"Requested leverage ({leverage}x) exceeds maximum allowed ({self.max_leverage:.2f}x)"
            )

        # Set effective leverage
        self.perp_leverage = (
            min(leverage, self.max_leverage) if self.enforce_margin_limits else leverage
        )

        # Initialize position tracking
        self.spot_quantity = 0
        self.perp_quantity = 0
        self.spot_entry_price = 0
        self.perp_entry_price = 0
        self.is_position_open = False

        # Allocation tracking
        self.spot_allocation = 0  # Will be calculated during position sizing
        self.perp_allocation = 0  # Will be calculated during position sizing

        # Funding multiplier (set by strategy)
        self.funding_periods_multiplier = 1

        # Print initial configuration
        print(f"SpotPerpMarket initialized with:")
        print(f"  Capital: ${capital}")
        print(f"  Requested perp leverage: {leverage}x")
        print(f"  Exchange max leverage: {self.max_leverage:.2f}x")
        print(f"  Effective perp leverage: {self.perp_leverage:.2f}x")
        print(f"  Initial margin: {self.initial_margin_pct:.2f}%")
        print(f"  Maintenance margin: {self.maint_margin_pct:.2f}%")
        print(f"  Liquidation fee: {self.liquidation_fee_pct:.2f}%")
        print(f"  Fee rate: {fee_rate*100:.4f}%")
        print(f"  Mode: Capital-efficient delta neutral")

    def _extract_initial_margin(self) -> float:
        """
        Extract initial margin requirement from contract specifications.

        Returns:
            Initial margin percentage
        """
        if len(self.data) == 0:
            return 5.0  # Default value if no data

        first_row = self.data.iloc[0]

        # Check for the exact column name
        if "perpetual_initial" in first_row:
            return first_row["perpetual_initial"]
        else:
            return 5.0  # Default value

    def _extract_maint_margin(self) -> float:
        """
        Extract maintenance margin requirement from contract specifications.

        Returns:
            Maintenance margin percentage
        """
        if len(self.data) == 0:
            return 2.5  # Default value if no data

        first_row = self.data.iloc[0]

        # Check for the exact column name
        if "perpetual_maint" in first_row:
            return first_row["perpetual_maint"]
        else:
            return 2.5  # Default value

    def _extract_liquidation_fee(self) -> float:
        """
        Extract liquidation fee from contract specifications.

        Returns:
            Liquidation fee percentage
        """
        if len(self.data) == 0:
            return 0.5  # Default value if no data

        first_row = self.data.iloc[0]

        # Check for the exact column name
        if "perpetual_liquidation_fee" in first_row:
            return first_row["perpetual_liquidation_fee"]
        else:
            return 0.5  # Default value

    def _calculate_max_leverage(self) -> float:
        """
        Calculate maximum allowed leverage based on initial margin requirement.

        Returns:
            Maximum allowed leverage
        """
        if self.initial_margin_pct <= 0:
            return 100.0  # Arbitrary high value if margin requirement is invalid

        return 100.0 / self.initial_margin_pct

    def calculate_position_sizes(self, data_row: pd.Series) -> Dict[str, float]:
        """
        Calculate position sizes for spot and perpetual futures with capital-efficient leverage.

        This implementation:
        1. Applies leverage only to the perpetual futures position
        2. Determines optimal capital allocation for delta neutrality
        3. Ensures equal notional value on both spot and perp sides

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
                "spot_allocation": 0,
                "perp_allocation": 0,
            }

        # Calculate optimal capital allocation for delta neutrality
        # For L = leverage, we need:
        # spot_alloc + perp_alloc = capital
        # spot_alloc = perp_alloc * L
        # Solving these two equations:
        perp_allocation = self.capital / (1 + self.perp_leverage)
        spot_allocation = self.capital - perp_allocation

        # Store allocations
        self.spot_allocation = spot_allocation
        self.perp_allocation = perp_allocation

        # Calculate notional values (should be equal for delta neutrality)
        perp_notional = perp_allocation * self.perp_leverage
        spot_notional = spot_allocation

        # Double-check delta neutrality
        if abs(spot_notional - perp_notional) > 0.01:
            print(
                f"Warning: Delta neutrality not achieved. Spot: ${spot_notional:.2f}, Perp: ${perp_notional:.2f}"
            )
            # Adjust to ensure exact delta neutrality
            target_notional = min(spot_notional, perp_notional)
            spot_notional = target_notional
            perp_notional = target_notional

        # Calculate quantities
        spot_quantity = spot_notional / spot_price
        perp_quantity = perp_notional / perp_price

        # Calculate total capital deployed
        total_capital_deployed = spot_allocation + perp_allocation

        # Calculate effective portfolio leverage (total notional / capital)
        total_notional = spot_notional + perp_notional
        effective_portfolio_leverage = total_notional / self.capital

        # Return position details
        return {
            "spot_quantity": spot_quantity,
            "perp_quantity": perp_quantity,
            "spot_notional": spot_notional,
            "perp_notional": perp_notional,
            "spot_allocation": spot_allocation,
            "perp_allocation": perp_allocation,
            "perp_leverage": self.perp_leverage,
            "total_notional": total_notional,
            "effective_portfolio_leverage": effective_portfolio_leverage,
            "total_capital_deployed": total_capital_deployed,
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
        spot_allocation = sizes["spot_allocation"]
        perp_allocation = sizes["perp_allocation"]

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
            "spot_allocation": spot_allocation,
            "perp_allocation": perp_allocation,
            "perp_leverage": self.perp_leverage,
            "total_notional": total_notional,
            "effective_portfolio_leverage": sizes["effective_portfolio_leverage"],
        }

        # Log position information
        print(f"\nPosition initialized with capital-efficient leverage:")
        print(f"  Entry Date: {position['entry_date']}")
        print(f"  Capital: ${self.capital:.2f}")
        print(
            f"  Spot Allocation: ${spot_allocation:.2f} ({spot_allocation/self.capital*100:.1f}% of capital)"
        )
        print(
            f"  Perp Allocation: ${perp_allocation:.2f} ({perp_allocation/self.capital*100:.1f}% of capital)"
        )
        print(f"  Perp Leverage: {self.perp_leverage:.2f}x")
        print(f"  Spot Entry: ${position['spot_entry']:.2f}")
        print(f"  Perp Entry: ${position['perp_entry']:.2f}")
        print(f"  Spot Quantity: {position['spot_quantity']:.8f}")
        print(f"  Perp Quantity: {position['perp_quantity']:.8f}")
        print(f"  Spot Notional: ${position['spot_notional']:.2f}")
        print(f"  Perp Notional: ${position['perp_notional']:.2f}")
        print(f"  Total Notional: ${position['total_notional']:.2f}")
        print(
            f"  Effective Portfolio Leverage: {position['effective_portfolio_leverage']:.2f}x"
        )
        print(f"  Entry Fee: ${position['entry_fee']:.2f}")

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
            "spot_allocation": 0,
            "perp_allocation": 0,
            "perp_leverage": self.effective_perp_leverage,
            "total_notional": 0,
            "effective_portfolio_leverage": 1.0,
        }

    def calculate_pnl(self, data_row: pd.Series) -> Dict[str, Any]:
        """
        Calculate PnL for the current position with enhanced risk metrics.

        Args:
            data_row: Current market data row

        Returns:
            Dictionary with PnL components and risk metrics
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

        # Calculate funding payment based on leveraged perpetual position size
        # Funding payment = position size * price * funding rate * period multiplier
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

        # Calculate updated equity (capital + accumulated PnL)
        # This includes PnL from both spot and perp positions, plus accumulated funding
        net_market_pnl = spot_pnl + perp_pnl
        current_equity = self.capital + net_market_pnl + funding_payment

        # Calculate current leverage (perp notional / perp allocation)
        current_perp_leverage = (
            perp_notional / self.perp_allocation if self.perp_allocation > 0 else 0
        )

        # Calculate effective portfolio leverage (total notional / equity)
        current_portfolio_leverage = (
            total_notional / current_equity if current_equity > 0 else 0
        )

        # Calculate risk metrics

        # 1. Calculate maintenance margin requirement
        maint_margin_requirement = perp_notional * (self.maint_margin_pct / 100)

        # 2. Calculate health factor
        health_factor = (
            current_equity / maint_margin_requirement
            if maint_margin_requirement > 0
            else float("inf")
        )

        # 3. Determine risk level
        if health_factor >= 3:
            risk_level = "Very Safe"
        elif health_factor >= 1.5:
            risk_level = "Moderate Risk"
        elif health_factor >= 1:
            risk_level = "High Risk"
        else:
            risk_level = "Liquidation Imminent"

        # 4. Calculate liquidation price (for short perp position)
        # This is simplified - a more complex calculation would account for the spot position
        liquidation_price = self.perp_entry_price * (
            1 + (self.maint_margin_pct / 100) / self.perp_leverage
        )

        # 5. Calculate buffer to liquidation
        buffer_to_liquidation = ((liquidation_price - perp_price) / perp_price) * 100

        # 6. Calculate basis between spot and perp
        current_basis_pct = ((perp_price / spot_price) - 1) * 100

        # Update result with all metrics
        result.update(
            {
                "spot_pnl": spot_pnl,
                "perp_pnl": perp_pnl,
                "funding_payment": funding_payment,
                "spot_notional": spot_notional,
                "perp_notional": perp_notional,
                "total_notional": total_notional,
                "equity": current_equity,
                "current_perp_leverage": current_perp_leverage,
                "current_portfolio_leverage": current_portfolio_leverage,
                # Risk metrics
                "health_factor": health_factor,
                "risk_level": risk_level,
                "maint_margin_requirement": maint_margin_requirement,
                "liquidation_price": liquidation_price,
                "buffer_to_liquidation_pct": buffer_to_liquidation,
                "current_basis_pct": current_basis_pct,
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
            "spot_allocation": self.spot_allocation,
            "perp_allocation": self.perp_allocation,
        }

        # Reset position state
        self.spot_quantity = 0
        self.perp_quantity = 0
        self.spot_entry_price = 0
        self.perp_entry_price = 0
        self.spot_allocation = 0
        self.perp_allocation = 0
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
            "requested_leverage": self.requested_leverage,
            "perp_leverage": self.perp_leverage,
            "max_leverage": self.max_leverage,
            "initial_margin_pct": self.initial_margin_pct,
            "liquidation_fee_pct": self.liquidation_fee_pct,
            "capital": self.capital,
            "is_position_open": self.is_position_open,
            "spot_allocation": self.spot_allocation,
            "perp_allocation": self.perp_allocation,
        }


# Add these methods to the SpotPerpMarket class


def calculate_liquidation_price(self):
    """
    Calculate the price at which the position would be liquidated.

    For a delta-neutral spot-perp position, liquidation occurs when:
    - The perp position loses enough value that maintenance margin is breached
    - The combined equity (accounting for spot profit) falls below maintenance requirement

    Returns:
        Dict with liquidation details
    """
    if not self.is_position_open or self.perp_quantity == 0:
        return {
            "liquidation_price": None,
            "upside_liquidation": None,
            "downside_liquidation": None,
            "current_price": None,
            "buffer_percentage": None,
        }

    # Get current prices
    current_spot_price = self.data.iloc[-1]["spot_close"]
    current_perp_price = self.data.iloc[-1]["perp_close"]

    # Calculate initial position details
    initial_perp_notional = self.perp_quantity * self.perp_entry_price
    initial_spot_notional = self.spot_quantity * self.spot_entry_price

    # Calculate maintenance margin requirement
    maint_margin_requirement = initial_perp_notional * (self.maint_margin_pct / 100)

    # For a short perp position, liquidation occurs at higher prices
    # We need to consider the spot profit offsetting perp losses

    # Calculate upside liquidation (price increase)
    # At what price would the perp loss exceed: perp_allocation + spot_profit - maintenance_margin?
    # This is complex because spot profit offsets perp losses

    # Downside liquidation is less likely (perp profits when price falls)
    # But could happen in extreme basis expansion scenarios

    # Simplified calculation for short perp: at what price do we get liquidated?
    # liquidation_price = entry_price * (1 + (maintenance_margin_pct / perp_leverage))
    liquidation_price = self.perp_entry_price * (
        1 + (self.maint_margin_pct / 100) / self.perp_leverage
    )

    # For more accuracy, we need to account for spot position offsetting losses
    # This is more complex and would involve solving an equation

    # Calculate buffer to liquidation (as percentage)
    buffer_percentage = (
        abs(liquidation_price - current_perp_price) / current_perp_price
    ) * 100

    return {
        "liquidation_price": liquidation_price,
        "current_price": current_perp_price,
        "buffer_percentage": buffer_percentage,
        "maintenance_margin_pct": self.maint_margin_pct,
    }


def calculate_health_factor(self):
    """
    Calculate the current health factor of the position.

    Health factor = Current Equity / Maintenance Margin Requirement

    Returns:
        Dict with health metrics
    """
    if not self.is_position_open:
        return {"health_factor": None, "risk_level": "No Position"}

    # Get current prices
    current_spot_price = self.data.iloc[-1]["spot_close"]
    current_perp_price = self.data.iloc[-1]["perp_close"]

    # Calculate current notional values
    current_spot_notional = self.spot_quantity * current_spot_price
    current_perp_notional = self.perp_quantity * current_perp_price

    # Calculate PnL
    spot_pnl = self.spot_quantity * (current_spot_price - self.spot_entry_price)
    perp_pnl = self.perp_quantity * (self.perp_entry_price - current_perp_price)
    net_pnl = spot_pnl + perp_pnl

    # Calculate current equity
    current_equity = self.spot_allocation + self.perp_allocation + net_pnl

    # Calculate maintenance margin requirement
    maint_margin_requirement = current_perp_notional * (self.maint_margin_pct / 100)

    # Calculate health factor
    health_factor = (
        current_equity / maint_margin_requirement
        if maint_margin_requirement > 0
        else float("inf")
    )

    # Determine risk level
    risk_level = "Unknown"
    if health_factor >= 3:
        risk_level = "Very Safe"
    elif health_factor >= 1.5:
        risk_level = "Moderate Risk"
    elif health_factor >= 1:
        risk_level = "High Risk"
    else:
        risk_level = "Liquidation Imminent"

    return {
        "health_factor": health_factor,
        "risk_level": risk_level,
        "current_equity": current_equity,
        "maintenance_requirement": maint_margin_requirement,
        "net_pnl": net_pnl,
        "spot_pnl": spot_pnl,
        "perp_pnl": perp_pnl,
    }


def calculate_stress_test(self, price_change_pct=10):
    """
    Conduct a stress test on the current position.

    Simulates position performance under various market scenarios.

    Args:
        price_change_pct: Percentage price change to simulate (both up and down)

    Returns:
        Dict with stress test results
    """
    if not self.is_position_open:
        return {"status": "No position open"}

    # Get current prices
    current_spot_price = self.data.iloc[-1]["spot_close"]
    current_perp_price = self.data.iloc[-1]["perp_close"]

    # Calculate current basis
    current_basis_pct = ((current_perp_price / current_spot_price) - 1) * 100

    # Simulate price increases
    upside_scenarios = []
    for pct in [5, 10, 20, 50]:
        # Calculate new prices
        new_spot_price = current_spot_price * (1 + pct / 100)
        new_perp_price = current_perp_price * (1 + pct / 100)

        # Calculate PnL
        spot_pnl = self.spot_quantity * (new_spot_price - self.spot_entry_price)
        perp_pnl = self.perp_quantity * (self.perp_entry_price - new_perp_price)
        net_pnl = spot_pnl + perp_pnl

        # Calculate new equity
        new_equity = self.spot_allocation + self.perp_allocation + net_pnl

        # Calculate new health factor
        new_perp_notional = self.perp_quantity * new_perp_price
        new_maint_margin = new_perp_notional * (self.maint_margin_pct / 100)
        health_factor = (
            new_equity / new_maint_margin if new_maint_margin > 0 else float("inf")
        )

        upside_scenarios.append(
            {
                "price_change_pct": pct,
                "spot_price": new_spot_price,
                "perp_price": new_perp_price,
                "spot_pnl": spot_pnl,
                "perp_pnl": perp_pnl,
                "net_pnl": net_pnl,
                "health_factor": health_factor,
                "liquidation": health_factor < 1,
            }
        )

    # Simulate price decreases
    downside_scenarios = []
    for pct in [5, 10, 20, 50]:
        # Calculate new prices
        new_spot_price = current_spot_price * (1 - pct / 100)
        new_perp_price = current_perp_price * (1 - pct / 100)

        # Calculate PnL
        spot_pnl = self.spot_quantity * (new_spot_price - self.spot_entry_price)
        perp_pnl = self.perp_quantity * (self.perp_entry_price - new_perp_price)
        net_pnl = spot_pnl + perp_pnl

        # Calculate new equity
        new_equity = self.spot_allocation + self.perp_allocation + net_pnl

        # Calculate new health factor
        new_perp_notional = self.perp_quantity * new_perp_price
        new_maint_margin = new_perp_notional * (self.maint_margin_pct / 100)
        health_factor = (
            new_equity / new_maint_margin if new_maint_margin > 0 else float("inf")
        )

        downside_scenarios.append(
            {
                "price_change_pct": -pct,
                "spot_price": new_spot_price,
                "perp_price": new_perp_price,
                "spot_pnl": spot_pnl,
                "perp_pnl": perp_pnl,
                "net_pnl": net_pnl,
                "health_factor": health_factor,
                "liquidation": health_factor < 1,
            }
        )

    # Basis expansion scenarios (perp price increases more than spot)
    basis_expansion_scenarios = []
    for basis_change in [1, 2, 5, 10]:
        # Calculate new perp price with expanded basis
        new_basis_pct = current_basis_pct + basis_change
        new_perp_price = current_spot_price * (1 + new_basis_pct / 100)

        # Calculate PnL
        spot_pnl = 0  # Spot price unchanged
        perp_pnl = self.perp_quantity * (self.perp_entry_price - new_perp_price)
        net_pnl = perp_pnl  # No spot PnL

        # Calculate new equity
        new_equity = self.spot_allocation + self.perp_allocation + net_pnl

        # Calculate new health factor
        new_perp_notional = self.perp_quantity * new_perp_price
        new_maint_margin = new_perp_notional * (self.maint_margin_pct / 100)
        health_factor = (
            new_equity / new_maint_margin if new_maint_margin > 0 else float("inf")
        )

        basis_expansion_scenarios.append(
            {
                "basis_change_pct": basis_change,
                "new_basis_pct": new_basis_pct,
                "perp_price": new_perp_price,
                "perp_pnl": perp_pnl,
                "health_factor": health_factor,
                "liquidation": health_factor < 1,
            }
        )

    return {
        "current_basis_pct": current_basis_pct,
        "current_health": self.calculate_health_factor(),
        "upside_scenarios": upside_scenarios,
        "downside_scenarios": downside_scenarios,
        "basis_expansion_scenarios": basis_expansion_scenarios,
    }
