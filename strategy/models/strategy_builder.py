"""
Strategy Builder Framework - Core Components

This module provides a flexible framework for building trading strategies
from modular, reusable components.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable


class StrategyContext:
    """
    Maintains state and provides data access for strategy components.

    This class gives signals and other components access to market data
    and the current state of the strategy.
    """

    def __init__(self, data: pd.DataFrame, current_index: int = 0):
        self.data = data
        self.current_index = current_index
        self.position = None
        self.state = {}  # For storing strategy-specific state

    def get_data_window(self, lookback: int = 10) -> pd.DataFrame:
        """Get a window of data for calculations."""
        start = max(0, self.current_index - lookback)
        return self.data.iloc[start : self.current_index + 1]

    def get_current_row(self) -> pd.Series:
        """Get the current data row."""
        return self.data.iloc[self.current_index]

    def get_asset_data(self, asset: str) -> pd.Series:
        """Get data for a specific asset."""
        if asset in self.data.columns:
            return self.data[asset]
        return None


class Signal(ABC):
    """Base class for all trading signals."""

    @abstractmethod
    def evaluate(self, data_row: pd.Series, context: StrategyContext) -> bool:
        """
        Evaluate the signal based on the current data.

        Args:
            data_row: Current market data point
            context: Strategy context with additional data

        Returns:
            bool: True if signal is active, False otherwise
        """
        pass


class EntrySignal(Signal):
    """Signal that determines when to enter a position."""

    def should_enter(self, data_row: pd.Series, context: StrategyContext) -> bool:
        """
        Check if we should enter a position based on this signal.

        This is a convenient alias for evaluate().
        """
        return self.evaluate(data_row, context)


class ExitSignal(Signal):
    """Signal that determines when to exit a position."""

    def should_exit(self, data_row: pd.Series, context: StrategyContext) -> bool:
        """
        Check if we should exit a position based on this signal.

        This is a convenient alias for evaluate().
        """
        return self.evaluate(data_row, context)


class CompositeSignal(Signal):
    """Combines multiple signals using logical operations."""

    def __init__(self, signals: List[Signal], operator: str = "AND"):
        """
        Initialize a composite signal.

        Args:
            signals: List of signals to combine
            operator: How to combine signals ('AND', 'OR', 'MAJORITY')
        """
        self.signals = signals
        self.operator = operator.upper()

    def evaluate(self, data_row: pd.Series, context: StrategyContext) -> bool:
        if not self.signals:
            return False

        results = [signal.evaluate(data_row, context) for signal in self.signals]

        if self.operator == "AND":
            return all(results)
        elif self.operator == "OR":
            return any(results)
        elif self.operator == "MAJORITY":
            return sum(results) > len(results) / 2
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


class FundingRateSignal(EntrySignal):
    """Signal based on funding rate crossing a threshold."""

    def __init__(
        self, threshold: float = 0.0, column: str = "funding_rate", invert: bool = False
    ):
        """
        Initialize the funding rate signal.

        Args:
            threshold: Minimum funding rate to trigger signal
            column: Column name for funding rate in the dataframe
            invert: If True, trigger when funding is below threshold
        """
        self.threshold = threshold
        self.column = column
        self.invert = invert

    def evaluate(self, data_row: pd.Series, context: StrategyContext) -> bool:
        if self.column not in data_row:
            return False

        funding_rate = data_row[self.column]

        if self.invert:
            return funding_rate < self.threshold
        else:
            return funding_rate >= self.threshold


class VolatilitySignal(EntrySignal):
    """Signal based on price volatility."""

    def __init__(
        self,
        lookback: int = 20,
        threshold: float = 0.5,
        price_column: str = "spot_close",
        annualize: bool = True,
    ):
        """
        Initialize the volatility signal.

        Args:
            lookback: Number of periods to calculate volatility
            threshold: Volatility threshold (annualized)
            price_column: Column name for price data
            annualize: Whether to annualize the volatility
        """
        self.lookback = lookback
        self.threshold = threshold
        self.price_column = price_column
        self.annualize = annualize

    def evaluate(self, data_row: pd.Series, context: StrategyContext) -> bool:
        window = context.get_data_window(self.lookback)

        if self.price_column not in window.columns or len(window) < 2:
            return False

        returns = window[self.price_column].pct_change().dropna()

        if len(returns) < 2:
            return False

        volatility = returns.std()

        if self.annualize:
            # Assuming daily data, adjust as needed
            volatility *= np.sqrt(252)

        return volatility < self.threshold


class PositionSizer(ABC):
    """Base class for position sizing logic."""

    @abstractmethod
    def calculate_position_size(
        self,
        data_row: pd.Series,
        context: StrategyContext,
        capital: float,
        leverage: float,
    ) -> Dict[str, float]:
        """
        Calculate position sizes for all instruments.

        Args:
            data_row: Current market data
            context: Strategy context
            capital: Available capital
            leverage: Leverage multiplier

        Returns:
            Dict mapping instrument names to position sizes
        """
        pass


class EqualNotionalSizer(PositionSizer):
    """Allocates equal notional value to each side of the trade."""

    def calculate_position_size(
        self,
        data_row: pd.Series,
        context: StrategyContext,
        capital: float,
        leverage: float,
    ) -> Dict[str, float]:
        """
        Calculate position sizes with equal notional value.

        For funding arb, this typically means:
        - Long spot position
        - Short perp position of equal notional value
        """
        position_size = capital * leverage / 2  # Split between spot and perp

        spot_price = data_row.get("spot_close", 0)
        perp_price = data_row.get("perp_close", 0)

        if spot_price <= 0 or perp_price <= 0:
            return {"spot_quantity": 0, "perp_quantity": 0}

        spot_quantity = position_size / spot_price
        perp_quantity = position_size / perp_price

        return {
            "spot_quantity": spot_quantity,
            "perp_quantity": perp_quantity,
            "long_notional": spot_quantity * spot_price,
            "short_notional": perp_quantity * perp_price,
        }


class RiskManager(ABC):
    """Base class for risk management components."""

    @abstractmethod
    def check_risk_limits(
        self,
        position_sizing: Dict[str, float],
        data_row: pd.Series,
        context: StrategyContext,
    ) -> Dict[str, float]:
        """
        Check if position sizing meets risk constraints.

        Args:
            position_sizing: Proposed position sizes
            data_row: Current market data
            context: Strategy context

        Returns:
            Adjusted position sizes that meet risk constraints
        """
        pass


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, name: str = "BaseStrategy"):
        """Initialize the strategy."""
        self.name = name
        self.position = None
        self.context = None

    @abstractmethod
    def initialize_position(
        self, data: pd.DataFrame, capital: float, leverage: float, fee_rate: float
    ) -> Dict[str, Any]:
        """
        Initialize a new position based on strategy rules.

        Args:
            data: Market data DataFrame
            capital: Initial capital
            leverage: Leverage multiplier
            fee_rate: Trading fee rate

        Returns:
            Dict with position details
        """
        pass

    @abstractmethod
    def calculate_pnl(self, data_row: pd.Series) -> Dict[str, float]:
        """
        Calculate PnL for current position based on new data.

        Args:
            data_row: New market data point

        Returns:
            Dict with PnL components
        """
        pass

    @abstractmethod
    def close_position(self, data_row: pd.Series, fee_rate: float) -> Dict[str, Any]:
        """
        Close the current position.

        Args:
            data_row: Current market data
            fee_rate: Trading fee rate

        Returns:
            Dict with position exit details
        """
        pass


class BetaStrategy(BaseStrategy):
    """
    Pure funding rate capture strategy (benchmark).

    This strategy:
    1. Enters when funding rate signal is positive
    2. Maintains equal notional value on spot and perp
    3. Exits when exit signal triggers
    """

    def __init__(
        self,
        entry_signals: List[EntrySignal] = None,
        exit_signals: List[ExitSignal] = None,
        position_sizer: PositionSizer = None,
        risk_manager: RiskManager = None,
        name: str = "BetaStrategy",
    ):
        """
        Initialize the Beta Strategy.

        Args:
            entry_signals: Signals that determine entry conditions
            exit_signals: Signals that determine exit conditions
            position_sizer: Component that determines position sizes
            risk_manager: Component that enforces risk limits
            name: Strategy name
        """
        super().__init__(name=name)

        # Use default components if none provided
        self.entry_signals = entry_signals or [FundingRateSignal(threshold=0.0)]
        self.exit_signals = exit_signals or [
            FundingRateSignal(threshold=0.0, invert=True)
        ]
        self.position_sizer = position_sizer or EqualNotionalSizer()
        self.risk_manager = risk_manager

        # Track periods held and trade history
        self.periods_held = 0
        self.trade_history = []
        self.current_trade = None
        self.is_position_open = False

    def initialize_position(
        self, data: pd.DataFrame, capital: float, leverage: float, fee_rate: float
    ) -> Dict[str, Any]:
        """Initialize a new position if entry signals allow."""
        # Initialize context
        self.context = StrategyContext(data, current_index=0)
        first_row = data.iloc[0]

        # Check if we should enter based on entry signals
        should_enter = all(
            signal.evaluate(first_row, self.context) for signal in self.entry_signals
        )

        if not should_enter:
            # Return empty position
            return {
                "entry_date": first_row["Timestamp"],
                "spot_entry": first_row["spot_close"],
                "perp_entry": first_row["perp_close"],
                "spot_quantity": 0,
                "perp_quantity": 0,
                "capital": capital,
                "entry_fee": 0,
                "total_notional": 0,
            }

        # Calculate position sizes
        position_sizes = self.position_sizer.calculate_position_size(
            first_row, self.context, capital, leverage
        )

        # Apply risk management if configured
        if self.risk_manager:
            position_sizes = self.risk_manager.check_risk_limits(
                position_sizes, first_row, self.context
            )

        # Calculate fees
        entry_fee = (
            position_sizes["long_notional"] + position_sizes["short_notional"]
        ) * fee_rate
        total_notional = (
            position_sizes["long_notional"] + position_sizes["short_notional"]
        )

        # Update strategy state
        self.is_position_open = True
        self.periods_held = 1

        # Record trade entry
        self.current_trade = {
            "entry_date": first_row["Timestamp"],
            "spot_entry": first_row["spot_close"],
            "perp_entry": first_row["perp_close"],
            "spot_quantity": position_sizes["spot_quantity"],
            "perp_quantity": position_sizes["perp_quantity"],
            "entry_funding": first_row["funding_rate"],
            "entry_capital": capital,
            "fees": entry_fee,
        }

        # Update position data
        self.position = {
            "entry_date": first_row["Timestamp"],
            "spot_entry": first_row["spot_close"],
            "perp_entry": first_row["perp_close"],
            "spot_quantity": position_sizes["spot_quantity"],
            "perp_quantity": position_sizes["perp_quantity"],
            "capital": capital - entry_fee,
            "entry_fee": entry_fee,
            "total_notional": total_notional,
        }

        return self.position

    def calculate_pnl(self, data_row: pd.Series) -> Dict[str, Any]:
        """Calculate PnL and check if we should exit the position."""
        # Update context
        if self.context:
            self.context.current_index += 1

        # Initialize result
        result = {
            "date": data_row["Timestamp"],
            "spot_pnl": 0,
            "perp_pnl": 0,
            "funding_payment": 0,
            "funding_rate": data_row["funding_rate"],
            "should_exit": False,
            "total_notional": 0,
        }

        # If no position is open, return empty result
        if not self.is_position_open or self.position is None:
            return result

        # Get current prices
        spot_price = data_row["spot_close"]
        perp_price = data_row["perp_close"]
        funding_rate = data_row["funding_rate"]

        # Calculate PnL components
        spot_pnl = self.position["spot_quantity"] * (
            spot_price - self.position["spot_entry"]
        )
        perp_pnl = self.position["perp_quantity"] * (
            self.position["perp_entry"] - perp_price
        )

        # Calculate funding payment
        # Adjust based on funding periods per day if needed
        funding_multiplier = 1
        if hasattr(self, "funding_periods_multiplier"):
            funding_multiplier = self.funding_periods_multiplier

        funding_payment = (
            self.position["perp_quantity"]
            * perp_price
            * funding_rate
            * funding_multiplier
        )

        # Calculate total notional value
        total_notional = (self.position["spot_quantity"] * spot_price) + (
            self.position["perp_quantity"] * perp_price
        )

        # Update periods held
        self.periods_held += 1

        # Check if we should exit based on exit signals
        should_exit = any(
            signal.evaluate(data_row, self.context) for signal in self.exit_signals
        )

        # Update result
        result.update(
            {
                "spot_pnl": spot_pnl,
                "perp_pnl": perp_pnl,
                "funding_payment": funding_payment,
                "funding_rate": funding_rate,
                "should_exit": should_exit,
                "total_notional": total_notional,
            }
        )

        return result

    def close_position(self, data_row: pd.Series, fee_rate: float) -> Dict[str, Any]:
        """Close the current position and record the trade."""
        # If no position is open, return empty result
        if not self.is_position_open or self.position is None:
            return {
                "exit_date": data_row["Timestamp"],
                "exit_spot": data_row["spot_close"],
                "exit_perp": data_row["perp_close"],
                "exit_fee": 0,
                "final_total_notional": 0,
            }

        # Get exit prices
        exit_spot = data_row["spot_close"]
        exit_perp = data_row["perp_close"]

        # Calculate final notional and fees
        final_spot_notional = self.position["spot_quantity"] * exit_spot
        final_perp_notional = self.position["perp_quantity"] * exit_perp
        final_total_notional = final_spot_notional + final_perp_notional
        exit_fee = final_total_notional * fee_rate

        # Calculate PnLs for the trade
        spot_pnl = self.position["spot_quantity"] * (
            exit_spot - self.position["spot_entry"]
        )
        perp_pnl = self.position["perp_quantity"] * (
            self.position["perp_entry"] - exit_perp
        )
        net_pnl = spot_pnl + perp_pnl

        # Complete the current trade record
        if self.current_trade is not None:
            self.current_trade.update(
                {
                    "exit_date": data_row["Timestamp"],
                    "spot_exit": exit_spot,
                    "perp_exit": exit_perp,
                    "exit_funding": data_row["funding_rate"],
                    "duration": (
                        (data_row["Timestamp"] - self.current_trade["entry_date"]).days
                        if isinstance(data_row["Timestamp"], pd.Timestamp)
                        else 0
                    ),
                    "spot_pnl": spot_pnl,
                    "perp_pnl": perp_pnl,
                    "net_pnl": net_pnl,
                    "fees": self.current_trade["fees"] + exit_fee,
                }
            )

            # Add to trade history
            self.trade_history.append(self.current_trade)
            self.current_trade = None

        # Reset position state
        self.is_position_open = False
        self.periods_held = 0
        old_position = self.position
        self.position = None

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
            "position": old_position,
        }

    def get_trade_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from trade history."""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "average_profit": 0,
                "average_loss": 0,
                "profit_factor": 0,
                "average_duration": 0,
            }

        winning_trades = [t for t in self.trade_history if t["net_pnl"] > 0]
        losing_trades = [t for t in self.trade_history if t["net_pnl"] <= 0]

        win_rate = (
            len(winning_trades) / len(self.trade_history) if self.trade_history else 0
        )

        avg_profit = (
            np.mean([t["net_pnl"] for t in winning_trades]) if winning_trades else 0
        )
        avg_loss = (
            np.mean([t["net_pnl"] for t in losing_trades]) if losing_trades else 0
        )

        total_profit = (
            sum(t["net_pnl"] for t in winning_trades) if winning_trades else 0
        )
        total_loss = sum(t["net_pnl"] for t in losing_trades) if losing_trades else 0

        profit_factor = -total_profit / total_loss if total_loss != 0 else float("inf")

        avg_duration = (
            np.mean([t["duration"] for t in self.trade_history])
            if self.trade_history
            else 0
        )

        return {
            "total_trades": len(self.trade_history),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "average_profit": float(avg_profit),
            "average_loss": float(avg_loss),
            "profit_factor": (
                float(profit_factor) if not np.isinf(profit_factor) else float(0)
            ),
            "average_duration": float(avg_duration),
        }


class EnhancedBetaStrategy(BetaStrategy):
    """
    Enhanced funding rate strategy that adds futures contracts.

    This strategy:
    1. Extends the basic Beta strategy
    2. Adds the ability to use term futures for more capital-efficient positioning
    3. Optimizes the allocation between spot, perp, and futures
    """

    def __init__(
        self,
        entry_signals: List[EntrySignal] = None,
        exit_signals: List[ExitSignal] = None,
        position_sizer: PositionSizer = None,
        risk_manager: RiskManager = None,
        use_futures: bool = True,
        name: str = "EnhancedBetaStrategy",
    ):
        """
        Initialize the Enhanced Beta Strategy.

        Args:
            entry_signals: Signals that determine entry conditions
            exit_signals: Signals that determine exit conditions
            position_sizer: Component that determines position sizes
            risk_manager: Component that enforces risk limits
            use_futures: Whether to use futures contracts
            name: Strategy name
        """
        super().__init__(
            entry_signals=entry_signals,
            exit_signals=exit_signals,
            position_sizer=position_sizer,
            risk_manager=risk_manager,
            name=name,
        )
        self.use_futures = use_futures

    # Override methods to implement futures contract handling
    # For now, we'll inherit from BetaStrategy with minimal changes
    # The actual implementation would add logic for futures positioning


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

    def build(self, strategy_class: type = None) -> BaseStrategy:
        """
        Build the strategy instance.

        Args:
            strategy_class: Strategy class to instantiate (default: BetaStrategy)

        Returns:
            Configured strategy instance
        """
        strategy_class = strategy_class or BetaStrategy

        # Instantiate the strategy with our components
        strategy = strategy_class(
            entry_signals=self.entry_signals,
            exit_signals=self.exit_signals,
            position_sizer=self.position_sizer,
            risk_manager=self.risk_manager,
            **self.strategy_params,
        )

        return strategy
