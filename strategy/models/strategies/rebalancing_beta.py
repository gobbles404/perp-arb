"""
Health Factor-Based Rebalancing Strategy

This strategy extends the basic BetaStrategy by adding health factor-based
position rebalancing capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from .beta import BetaStrategy
from ..base.signals import EntrySignal, ExitSignal
from ..base.position_sizer import PositionSizer
from ..base.risk_manager import RiskManager
from ..markets.spot_perp import SpotPerpMarket


class RebalancingBetaStrategy(BetaStrategy):
    """
    Beta strategy with health factor-based position rebalancing.

    Rebalances the position based on health factor thresholds:
    - When health factor < min_threshold: Reduce risk by decreasing position
    - When health factor > max_threshold: Increase risk by increasing position
    - Target health factor is the desired level after rebalancing
    """

    def __init__(
        self,
        entry_signals: Optional[List[EntrySignal]] = None,
        exit_signals: Optional[List[ExitSignal]] = None,
        position_sizer: Optional[PositionSizer] = None,
        risk_manager: Optional[RiskManager] = None,
        name: str = "RebalancingBetaStrategy",
        market: Optional[SpotPerpMarket] = None,
        min_threshold: float = 2.0,  # rebalance if health < this
        max_threshold: float = 6.0,  # rebalance if health > this
        target_threshold: float = 4.0,  # target health after rebalancing
        rebalance_cooldown: int = 1,  # periods between rebalances
    ):
        super().__init__(
            entry_signals=entry_signals,
            exit_signals=exit_signals,
            position_sizer=position_sizer,
            risk_manager=risk_manager,
            name=name,
            market=market,
        )
        # Rebalancing parameters
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.target_threshold = target_threshold
        self.rebalance_cooldown = rebalance_cooldown

        # Rebalancing state tracking
        self.periods_since_rebalance = 0
        self.rebalance_history = []
        self.total_rebalance_fees = 0
        self.total_rebalance_count = 0

        print(f"\n{name} initialized with rebalancing parameters:")
        print(f"  Min Health Threshold: {min_threshold}")
        print(f"  Max Health Threshold: {max_threshold}")
        print(f"  Target Health: {target_threshold}")
        print(f"  Rebalance Cooldown: {rebalance_cooldown} periods")

    def calculate_pnl(self, data_row: pd.Series) -> Dict[str, Any]:
        """
        Calculate PnL and check if we should rebalance the position.

        Extends the parent method to add health factor-based rebalancing.
        """
        # First, calculate basic PnL using parent method
        result = super().calculate_pnl(data_row)

        # If position is closed, nothing to rebalance
        if not self.is_position_open:
            return result

        # Increment periods since last rebalance
        self.periods_since_rebalance += 1

        # Initialize rebalancing fields
        result["rebalanced"] = False

        # Check if we need to rebalance (and cooldown period has passed)
        if self.periods_since_rebalance >= self.rebalance_cooldown:
            # Get current health factor
            health_metrics = self.market.calculate_health_factor()
            current_health = health_metrics["health_factor"]

            # Check if rebalancing is needed
            if current_health < self.min_threshold:
                # Health factor too low - reduce risk
                self._perform_rebalance(data_row, "reduce", current_health, result)

            elif current_health > self.max_threshold:
                # Health factor too high - increase risk
                self._perform_rebalance(data_row, "increase", current_health, result)

        # Add rebalancing status to result
        result["periods_since_rebalance"] = self.periods_since_rebalance
        result["total_rebalance_count"] = self.total_rebalance_count
        result["total_rebalance_fees"] = self.total_rebalance_fees

        return result

    def _perform_rebalance(
        self,
        data_row: pd.Series,
        action_type: str,
        current_health: float,
        result: Dict[str, Any],
    ):
        """
        Execute a rebalancing action.

        Args:
            data_row: Current market data
            action_type: 'reduce' or 'increase' risk
            current_health: Current health factor
            result: Result dict to update with rebalancing info
        """
        # Log rebalancing action
        print(f"\nRebalancing position ({action_type} risk):")
        print(f"  Current Health Factor: {current_health:.2f}")
        print(f"  Target Health Factor: {self.target_threshold:.2f}")

        # Perform rebalancing through market
        rebalance_result = self.market.rebalance_to_target_health(
            data_row, self.target_threshold
        )

        # If rebalancing succeeded, update metrics
        if rebalance_result.get("rebalanced", False):
            # Reset cooldown counter
            self.periods_since_rebalance = 0

            # Update counters
            self.total_rebalance_count += 1
            self.total_rebalance_fees += rebalance_result.get("adjustment_fee", 0)

            # Store rebalance event in history
            rebalance_event = {
                "timestamp": data_row["Timestamp"],
                "action_type": action_type,
                "initial_health": current_health,
                "target_health": self.target_threshold,
                "final_health": rebalance_result.get("final_health_factor", 0),
                "adjustment_fee": rebalance_result.get("adjustment_fee", 0),
                "spot_adjustment_pct": rebalance_result.get("spot_adjustment_pct", 0),
                "perp_adjustment_pct": rebalance_result.get("perp_adjustment_pct", 0),
                "new_spot_qty": rebalance_result.get("new_spot_qty", 0),
                "new_perp_qty": rebalance_result.get("new_perp_qty", 0),
            }

            self.rebalance_history.append(rebalance_event)

            # Update result with rebalancing info
            result["rebalanced"] = True
            result["rebalance_action"] = action_type
            result["initial_health"] = current_health
            result["final_health"] = rebalance_result.get("final_health_factor", 0)
            result["rebalance_fee"] = rebalance_result.get("adjustment_fee", 0)

            # Log rebalance details
            print(
                f"  Final Health Factor: {rebalance_result.get('final_health_factor', 0):.2f}"
            )
            print(f"  Adjustment Fee: ${rebalance_result.get('adjustment_fee', 0):.2f}")

    def get_rebalancing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about rebalancing actions.

        Returns:
            Dict with rebalancing statistics
        """
        if not self.rebalance_history:
            return {
                "total_rebalances": 0,
                "reduce_risk_count": 0,
                "increase_risk_count": 0,
                "total_fees": 0,
                "avg_adjustment_pct": 0,
                "avg_health_improvement": 0,
            }

        # Count actions by type
        reduce_actions = [
            r for r in self.rebalance_history if r["action_type"] == "reduce"
        ]
        increase_actions = [
            r for r in self.rebalance_history if r["action_type"] == "increase"
        ]

        # Calculate averages
        avg_spot_adjustment = np.mean(
            [abs(r["spot_adjustment_pct"]) for r in self.rebalance_history]
        )
        avg_health_improvement = np.mean(
            [
                abs(r["final_health"] - r["initial_health"])
                for r in self.rebalance_history
            ]
        )

        return {
            "total_rebalances": len(self.rebalance_history),
            "reduce_risk_count": len(reduce_actions),
            "increase_risk_count": len(increase_actions),
            "total_fees": self.total_rebalance_fees,
            "avg_adjustment_pct": float(avg_spot_adjustment),
            "avg_health_improvement": float(avg_health_improvement),
        }
