"""
Strategy Builder - Class for composing strategies from modular components.
"""

from typing import List, Dict, Any, Optional, Type

from ..base.base_strategy import BaseStrategy
from ..base.signals import EntrySignal, ExitSignal
from ..base.position_sizer import PositionSizer
from ..base.risk_manager import RiskManager


class StrategyBuilder:
    """
    Builder for composing strategies from modular components.

    This class provides a fluent API for constructing strategies
    from reusable components without complex inheritance hierarchies.
    """

    def __init__(self):
        """Initialize the strategy builder."""
        self.entry_signals = []
        self.exit_signals = []
        self.position_sizer = None
        self.risk_manager = None
        self.strategy_params = {}

    def add_entry_signal(self, signal: EntrySignal) -> "StrategyBuilder":
        """
        Add an entry signal to the strategy.

        Args:
            signal: The entry signal to add

        Returns:
            Self for method chaining
        """
        self.entry_signals.append(signal)
        return self

    def add_exit_signal(self, signal: ExitSignal) -> "StrategyBuilder":
        """
        Add an exit signal to the strategy.

        Args:
            signal: The exit signal to add

        Returns:
            Self for method chaining
        """
        self.exit_signals.append(signal)
        return self

    def set_position_sizer(self, sizer: PositionSizer) -> "StrategyBuilder":
        """
        Set the position sizing component.

        Args:
            sizer: Position sizing component

        Returns:
            Self for method chaining
        """
        self.position_sizer = sizer
        return self

    def set_risk_manager(self, risk_manager: RiskManager) -> "StrategyBuilder":
        """
        Set the risk management component.

        Args:
            risk_manager: Risk management component

        Returns:
            Self for method chaining
        """
        self.risk_manager = risk_manager
        return self

    def set_param(self, name: str, value: Any) -> "StrategyBuilder":
        """
        Set a strategy-specific parameter.

        Args:
            name: Parameter name
            value: Parameter value

        Returns:
            Self for method chaining
        """
        self.strategy_params[name] = value
        return self

    def build(
        self, strategy_class: Optional[Type[BaseStrategy]] = None
    ) -> BaseStrategy:
        """
        Build the strategy instance.

        Args:
            strategy_class: Strategy class to instantiate

        Returns:
            Configured strategy instance
        """
        if strategy_class is None:
            from ..strategies.beta import BetaStrategy

            strategy_class = BetaStrategy

        # Instantiate the strategy with our components
        strategy = strategy_class(
            entry_signals=self.entry_signals,
            exit_signals=self.exit_signals,
            position_sizer=self.position_sizer,
            risk_manager=self.risk_manager,
            **self.strategy_params
        )

        return strategy
