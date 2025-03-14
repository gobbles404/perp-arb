"""
Base Strategy Definition - Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import pandas as pd


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
    def calculate_pnl(self, data_row: pd.Series) -> Dict[str, Any]:
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
