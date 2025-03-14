"""
Signal Components - Classes for generating trading signals.
"""

from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np

from .base_strategy import StrategyContext


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
