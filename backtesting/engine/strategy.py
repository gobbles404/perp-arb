# backtesting/engine/strategy.py
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from core.logger import get_logger
from engine.events import SignalEvent, OrderDirection, EventQueue, MarketEvent
from engine.portfolio import Portfolio

# Set up logger
logger = get_logger(__name__)


class Strategy:
    """
    Base strategy class that all trading strategies must inherit from.
    Defines the interface for how strategies interact with the backtesting engine.
    """

    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize the strategy.

        Args:
            name: Strategy name for identification
        """
        self.name = name
        self.id = f"{name}_{id(self)}"

        # Recent market data cache
        self.market_data_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.max_cache_size = 100  # Maximum number of data points to cache

        logger.info(f"Initialized strategy: {name}")

    def initialize(self, portfolio: Portfolio) -> None:
        """
        Initialize the strategy with the portfolio.
        Called once at the start of the backtest.

        Args:
            portfolio: Portfolio instance
        """
        logger.debug(f"Strategy {self.name} initialized with portfolio")

    def calculate_signals(
        self,
        timestamp: datetime,
        market_data: Dict[str, Dict[str, Any]],
        portfolio: Portfolio,
    ) -> List[SignalEvent]:
        """
        Calculate trading signals based on market data and portfolio state.
        Must be implemented by subclasses.

        Args:
            timestamp: Current timestamp
            market_data: Dictionary of market data by symbol
            portfolio: Current portfolio state

        Returns:
            List of signal events or empty list
        """
        raise NotImplementedError("Subclass must implement calculate_signals")

    def update_market_data(self, market_event: MarketEvent) -> None:
        """
        Update internal market data cache with new market event.

        Args:
            market_event: New market data event
        """
        symbol = market_event.symbol

        # Initialize cache for this symbol if not exists
        if symbol not in self.market_data_cache:
            self.market_data_cache[symbol] = []

        # Add new data point
        self.market_data_cache[symbol].append(
            {"timestamp": market_event.timestamp, **market_event.data}
        )

        # Limit cache size
        if len(self.market_data_cache[symbol]) > self.max_cache_size:
            self.market_data_cache[symbol].pop(0)

        logger.debug(
            f"Updated market data cache for {symbol} ({len(self.market_data_cache[symbol])} points)"
        )

    def get_historical_data(self, symbol: str, lookback: int = 20) -> pd.DataFrame:
        """
        Get historical data from cache as DataFrame.

        Args:
            symbol: Market symbol
            lookback: Number of data points to retrieve

        Returns:
            DataFrame with historical data
        """
        if symbol not in self.market_data_cache:
            logger.warning(f"No market data cache for {symbol}")
            return pd.DataFrame()

        data = self.market_data_cache[symbol][-lookback:]
        return pd.DataFrame(data)

    def calculate_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Z-score for a time series using rolling window.

        Args:
            series: Time series data
            window: Rolling window size

        Returns:
            Series of Z-scores
        """
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        # Calculate Z-score
        z_score = (series - rolling_mean) / rolling_std

        return z_score

    def position_sizing(
        self,
        symbol: str,
        direction: OrderDirection,
        portfolio: Portfolio,
        confidence: float = 1.0,
        max_risk_pct: float = 0.02,
    ) -> float:
        """
        Calculate appropriate position size based on portfolio and risk parameters.

        Args:
            symbol: Market symbol
            direction: Trade direction
            portfolio: Current portfolio
            confidence: Signal confidence (0.0 to 1.0)
            max_risk_pct: Maximum risk per trade as percentage of portfolio

        Returns:
            Position size
        """
        # Get portfolio value
        portfolio_value = portfolio.get_total_value()

        # Calculate risk amount
        risk_amount = portfolio_value * max_risk_pct * confidence

        # Get position for this symbol
        is_derivative = False  # Assume spot for simplicity
        position = portfolio.get_position(symbol, is_derivative)

        # Base position size calculation (assuming 1:1 leverage for simplicity)
        position_size = risk_amount / 100.0  # Simplified for illustration

        logger.debug(
            f"Calculated position size for {symbol}: {position_size:.4f} "
            f"(portfolio: {portfolio_value:.2f}, risk: {max_risk_pct:.2%})"
        )

        return position_size

    def on_fill(self, fill_data: Dict[str, Any]) -> None:
        """
        Callback when an order from this strategy is filled.

        Args:
            fill_data: Fill data
        """
        logger.debug(f"Strategy {self.name} notified of fill: {fill_data}")

    def teardown(self) -> None:
        """
        Clean up resources when backtest is complete.
        """
        self.market_data_cache.clear()
        logger.debug(f"Strategy {self.name} teardown complete")


class ZScoreStrategy(Strategy):
    """
    Strategy that trades based on Z-score of basis or spread.
    Entry when Z-score exceeds threshold, exit when it reverts to mean.
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        lookback_window: int = 20,
        max_positions: int = 3,
        risk_per_trade: float = 0.02,
        max_leverage: float = 3.0,
    ):
        """
        Initialize the Z-score strategy.

        Args:
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            lookback_window: Window for Z-score calculation
            max_positions: Maximum number of open positions
            risk_per_trade: Risk per trade as fraction of portfolio
            max_leverage: Maximum leverage to use for derivatives
        """
        super().__init__(name="ZScoreStrategy")
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.lookback_window = lookback_window
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage

        # Track active positions
        self.active_positions: Dict[str, Tuple[OrderDirection, float]] = {}

        logger.info(
            f"Initialized {self.name} with entry threshold {entry_threshold}, "
            f"exit threshold {exit_threshold}, lookback {lookback_window}"
        )

    def calculate_signals(
        self,
        timestamp: datetime,
        market_data: Dict[str, Dict[str, Any]],
        portfolio: Portfolio,
    ) -> List[SignalEvent]:
        """
        Calculate trading signals based on Z-scores.

        Args:
            timestamp: Current timestamp
            market_data: Dictionary of market data by symbol
            portfolio: Current portfolio state

        Returns:
            List of signal events
        """
        signals = []

        # Process each symbol
        for symbol, data in market_data.items():
            # Get historical data
            hist_data = self.get_historical_data(symbol, self.lookback_window * 2)

            # Skip if not enough data
            if len(hist_data) < self.lookback_window:
                logger.debug(f"Not enough data for {symbol}, skipping")
                continue

            # Check if basis data is available
            if "basis" not in data:
                logger.debug(f"No basis data for {symbol}, skipping")
                continue

            # Calculate Z-score if we have enough data
            basis_series = hist_data["basis"].astype(float)
            if len(basis_series) >= self.lookback_window:
                # Calculate Z-score
                z_score = self.calculate_zscore(
                    basis_series, self.lookback_window
                ).iloc[-1]

                # Check for entry signals
                if symbol not in self.active_positions:
                    # No active position, check for entry
                    if z_score > self.entry_threshold:
                        # Negative Z-score means basis is below average, expect it to increase
                        # Short spot, long perp
                        signals.append(
                            self._create_signal(
                                timestamp, symbol, OrderDirection.SELL, z_score
                            )
                        )
                        self.active_positions[symbol] = (OrderDirection.SELL, z_score)
                        logger.info(
                            f"Entry signal for {symbol}: SHORT spot/LONG perp (Z-score: {z_score:.2f})"
                        )
                    elif z_score < -self.entry_threshold:
                        # Positive Z-score means basis is above average, expect it to decrease
                        # Long spot, short perp
                        signals.append(
                            self._create_signal(
                                timestamp, symbol, OrderDirection.BUY, abs(z_score)
                            )
                        )
                        self.active_positions[symbol] = (
                            OrderDirection.BUY,
                            abs(z_score),
                        )
                        logger.info(
                            f"Entry signal for {symbol}: LONG spot/SHORT perp (Z-score: {z_score:.2f})"
                        )
                else:
                    # Active position, check for exit
                    direction, entry_score = self.active_positions[symbol]

                    # Check if Z-score reverted enough to exit
                    if (
                        direction == OrderDirection.SELL
                        and z_score < self.exit_threshold
                    ) or (
                        direction == OrderDirection.BUY
                        and z_score > -self.exit_threshold
                    ):
                        # Exit position - opposite direction of entry
                        exit_direction = (
                            OrderDirection.BUY
                            if direction == OrderDirection.SELL
                            else OrderDirection.SELL
                        )
                        signals.append(
                            self._create_signal(timestamp, symbol, exit_direction, 1.0)
                        )
                        del self.active_positions[symbol]
                        logger.info(
                            f"Exit signal for {symbol}: {exit_direction.value} (Z-score: {z_score:.2f})"
                        )

        return signals

    def _create_signal(
        self,
        timestamp: datetime,
        symbol: str,
        direction: OrderDirection,
        strength: float,
    ) -> SignalEvent:
        """
        Create a signal event.

        Args:
            timestamp: Current timestamp
            symbol: Market symbol
            direction: Trade direction
            strength: Signal strength

        Returns:
            Signal event
        """
        return SignalEvent(
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            strength=min(1.0, strength / self.entry_threshold),
            strategy_id=self.id,
        )


class BasisArbitrageStrategy(Strategy):
    """
    Simple basis arbitrage strategy that trades when basis exceeds a threshold.
    """

    def __init__(
        self,
        basis_threshold: float = 0.5,
        max_positions: int = 3,
        risk_per_trade: float = 0.02,
        max_leverage: float = 3.0,
    ):
        """
        Initialize the basis arbitrage strategy.

        Args:
            basis_threshold: Basis threshold for entry (percentage)
            max_positions: Maximum number of open positions
            risk_per_trade: Risk per trade as fraction of portfolio
            max_leverage: Maximum leverage to use for derivatives
        """
        super().__init__(name="BasisArbitrageStrategy")
        self.basis_threshold = basis_threshold
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage

        # Track active positions
        self.active_positions: Dict[str, OrderDirection] = {}

        logger.info(
            f"Initialized {self.name} with basis threshold {basis_threshold}%, "
            f"max positions {max_positions}"
        )

    def calculate_signals(
        self,
        timestamp: datetime,
        market_data: Dict[str, Dict[str, Any]],
        portfolio: Portfolio,
    ) -> List[SignalEvent]:
        """
        Calculate trading signals based on basis.

        Args:
            timestamp: Current timestamp
            market_data: Dictionary of market data by symbol
            portfolio: Current portfolio state

        Returns:
            List of signal events
        """
        signals = []

        # Limit to max positions
        if len(self.active_positions) >= self.max_positions:
            return signals

        # Process each symbol
        for symbol, data in market_data.items():
            # Skip if already have an active position for this symbol
            if symbol in self.active_positions:
                # Check for exit opportunity
                current_basis = data.get("basis", 0.0)
                direction = self.active_positions[symbol]

                # Exit if basis has converged
                if (direction == OrderDirection.SELL and current_basis < 0.2) or (
                    direction == OrderDirection.BUY and current_basis > -0.2
                ):
                    # Exit position - opposite direction of entry
                    exit_direction = (
                        OrderDirection.BUY
                        if direction == OrderDirection.SELL
                        else OrderDirection.SELL
                    )
                    signals.append(
                        SignalEvent(
                            timestamp=timestamp,
                            symbol=symbol,
                            direction=exit_direction,
                            strength=1.0,
                            strategy_id=self.id,
                        )
                    )
                    del self.active_positions[symbol]
                    logger.info(
                        f"Exit signal for {symbol}: {exit_direction.value} (basis: {current_basis:.2f}%)"
                    )

                continue

            # Check if basis data is available
            if "basis" not in data:
                continue

            # Get current basis
            basis = data["basis"]

            # Check for entry signals
            if basis > self.basis_threshold:
                # Positive basis means perp > spot
                # Go short perp, long spot
                signals.append(
                    SignalEvent(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=OrderDirection.BUY,  # Buy spot
                        strength=min(1.0, basis / (self.basis_threshold * 2)),
                        strategy_id=self.id,
                    )
                )
                self.active_positions[symbol] = OrderDirection.BUY
                logger.info(
                    f"Entry signal for {symbol}: LONG spot/SHORT perp (basis: {basis:.2f}%)"
                )
            elif basis < -self.basis_threshold:
                # Negative basis means perp < spot
                # Go long perp, short spot
                signals.append(
                    SignalEvent(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=OrderDirection.SELL,  # Sell spot
                        strength=min(1.0, abs(basis) / (self.basis_threshold * 2)),
                        strategy_id=self.id,
                    )
                )
                self.active_positions[symbol] = OrderDirection.SELL
                logger.info(
                    f"Entry signal for {symbol}: SHORT spot/LONG perp (basis: {basis:.2f}%)"
                )

        return signals


class FundingArbitrageStrategy(Strategy):
    """
    Strategy that captures funding payments from perpetual markets.
    """

    def __init__(
        self,
        funding_threshold: float = 0.01,
        max_positions: int = 3,
        risk_per_trade: float = 0.02,
        max_leverage: float = 3.0,
    ):
        """
        Initialize the funding arbitrage strategy.

        Args:
            funding_threshold: Funding rate threshold for entry (percentage)
            max_positions: Maximum number of open positions
            risk_per_trade: Risk per trade as fraction of portfolio
            max_leverage: Maximum leverage to use for derivatives
        """
        super().__init__(name="FundingArbitrageStrategy")
        self.funding_threshold = funding_threshold
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage

        # Track active positions
        self.active_positions: Dict[str, OrderDirection] = {}

        logger.info(
            f"Initialized {self.name} with funding threshold {funding_threshold}%, "
            f"max positions {max_positions}"
        )

    def calculate_signals(
        self,
        timestamp: datetime,
        market_data: Dict[str, Dict[str, Any]],
        portfolio: Portfolio,
    ) -> List[SignalEvent]:
        """
        Calculate trading signals based on funding rates.

        Args:
            timestamp: Current timestamp
            market_data: Dictionary of market data by symbol
            portfolio: Current portfolio state

        Returns:
            List of signal events
        """
        signals = []

        # Limit to max positions
        if len(self.active_positions) >= self.max_positions:
            return signals

        # Process each symbol
        for symbol, data in market_data.items():
            # Skip if already have an active position for this symbol
            if symbol in self.active_positions:
                # Check for basis convergence to exit
                current_basis = data.get("basis", 0.0)
                direction = self.active_positions[symbol]

                # Exit if basis has flipped sign
                if (direction == OrderDirection.SELL and current_basis < 0) or (
                    direction == OrderDirection.BUY and current_basis > 0
                ):
                    # Exit position - opposite direction of entry
                    exit_direction = (
                        OrderDirection.BUY
                        if direction == OrderDirection.SELL
                        else OrderDirection.SELL
                    )
                    signals.append(
                        SignalEvent(
                            timestamp=timestamp,
                            symbol=symbol,
                            direction=exit_direction,
                            strength=1.0,
                            strategy_id=self.id,
                        )
                    )
                    del self.active_positions[symbol]
                    logger.info(
                        f"Exit signal for {symbol}: {exit_direction.value} (basis: {current_basis:.2f}%)"
                    )

                continue

            # Check if funding rate data is available
            if "funding_rate" not in data:
                continue

            # Get current funding rate and basis
            funding_rate = data["funding_rate"]
            basis = data.get("basis", 0.0)

            # Check for entry signals
            if funding_rate > self.funding_threshold:
                # Positive funding means longs pay shorts
                # Go short perp, long spot to capture funding
                signals.append(
                    SignalEvent(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=OrderDirection.BUY,  # Buy spot
                        strength=min(1.0, funding_rate / (self.funding_threshold * 2)),
                        strategy_id=self.id,
                    )
                )
                self.active_positions[symbol] = OrderDirection.BUY
                logger.info(
                    f"Entry signal for {symbol}: LONG spot/SHORT perp (funding: {funding_rate:.4f}%)"
                )
            elif funding_rate < -self.funding_threshold:
                # Negative funding means shorts pay longs
                # Go long perp, short spot to capture funding
                signals.append(
                    SignalEvent(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=OrderDirection.SELL,  # Sell spot
                        strength=min(
                            1.0, abs(funding_rate) / (self.funding_threshold * 2)
                        ),
                        strategy_id=self.id,
                    )
                )
                self.active_positions[symbol] = OrderDirection.SELL
                logger.info(
                    f"Entry signal for {symbol}: SHORT spot/LONG perp (funding: {funding_rate:.4f}%)"
                )

        return signals
