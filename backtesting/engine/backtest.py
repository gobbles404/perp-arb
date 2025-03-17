# backtesting/engine/backtest.py
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import time
import os
import json

from core.logger import get_logger
from data.market_data import MarketData, MultiMarketData
from engine.events import (
    EventQueue,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    FundingEvent,
    OrderDirection,
    OrderType,
    EventType,
)
from engine.portfolio import Portfolio
from engine.broker import Broker
from engine.strategy import Strategy

# Set up logger
logger = get_logger(__name__)


class Backtest:
    """
    Main backtesting engine that coordinates the event-driven simulation.
    Manages market data, portfolio, strategy, and event processing.
    """

    def __init__(
        self,
        market_data: Union[MarketData, MultiMarketData, Dict[str, MarketData]],
        strategy: Strategy,
        initial_capital: float = 100000.0,
        cash_position_pct: float = 0.3,
        fee_rate: float = 0.0004,
        slippage: float = 0.0001,
        leverage: float = 1.0,
        freq: str = "1d",
    ):
        """
        Initialize the backtest.

        Args:
            market_data: Market data for one or more symbols
            strategy: Trading strategy
            initial_capital: Initial portfolio capital
            cash_position_pct: Percentage of capital to keep as cash reserve
            fee_rate: Trading fee rate
            slippage: Slippage rate for market orders
            leverage: Default leverage for derivative trades
            freq: Data frequency ('1d', '1h', etc.)
        """
        self.market_data = self._normalize_market_data(market_data)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cash_position_pct = cash_position_pct
        self.leverage = leverage
        self.freq = freq

        # Initialize components
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.broker = Broker(fee_rate=fee_rate, slippage_std=slippage)
        self.event_queue = EventQueue()

        # Initialize tracking vars
        self.current_datetime = None
        self.symbols = list(self.market_data.keys())
        self.results = None
        self.is_running = False

        # Initialize statistics
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_fees": 0.0,
            "total_slippage": 0.0,
            "total_funding": 0.0,
        }

        # Data for debugging and analysis
        self.all_market_events = []
        self.all_signal_events = []
        self.all_order_events = []
        self.all_fill_events = []
        self.all_funding_events = []

        logger.info(
            f"Initialized backtest with {len(self.symbols)} symbols, "
            f"initial capital: {initial_capital}, "
            f"leverage: {leverage}x, frequency: {freq}"
        )

    def _normalize_market_data(
        self, market_data: Union[MarketData, MultiMarketData, Dict[str, MarketData]]
    ) -> Dict[str, MarketData]:
        """
        Normalize market data to a dictionary of MarketData objects by symbol.

        Args:
            market_data: Market data in various formats

        Returns:
            Dictionary of MarketData objects by symbol
        """
        if isinstance(market_data, MarketData):
            # Single MarketData object
            return {market_data.symbol: market_data}
        elif isinstance(market_data, MultiMarketData):
            # MultiMarketData object
            return market_data.markets
        elif isinstance(market_data, dict):
            # Dictionary of MarketData objects
            return market_data
        else:
            raise ValueError(f"Unsupported market data type: {type(market_data)}")

    def _get_data_timerange(self) -> Tuple[datetime, datetime]:
        """
        Get common time range across all market data.

        Returns:
            Tuple of (start_time, end_time)
        """
        # Get min/max timestamps for each symbol
        start_dates = []
        end_dates = []

        for symbol, market in self.market_data.items():
            data = market.data
            start_dates.append(data.index.min())
            end_dates.append(data.index.max())

        # Get common range
        start_time = max(start_dates)
        end_time = min(end_dates)

        logger.info(f"Data time range: {start_time} to {end_time}")
        return start_time, end_time

    def _get_all_timestamps(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> List[datetime]:
        """
        Get list of all unique timestamps across all market data.

        Args:
            start_time: Start time filter
            end_time: End time filter

        Returns:
            Sorted list of timestamps
        """
        if not start_time or not end_time:
            data_start, data_end = self._get_data_timerange()
            start_time = start_time or data_start
            end_time = end_time or data_end

        # Collect all timestamps
        all_timestamps = set()
        for symbol, market in self.market_data.items():
            # Filter by date range
            filtered_data = market.data[
                (market.data.index >= start_time) & (market.data.index <= end_time)
            ]
            all_timestamps.update(filtered_data.index)

        # Sort timestamps
        sorted_timestamps = sorted(all_timestamps)

        logger.debug(f"Found {len(sorted_timestamps)} unique timestamps in data")
        return sorted_timestamps

    def _create_market_events(self, timestamp: datetime) -> Dict[str, Dict[str, Any]]:
        """
        Create market events for the given timestamp.

        Args:
            timestamp: Current timestamp

        Returns:
            Dictionary of market data by symbol
        """
        market_data_by_symbol = {}

        for symbol, market in self.market_data.items():
            # Get data for this timestamp
            try:
                data_at_time = market.data.loc[timestamp]

                # Convert to dict if Series
                if isinstance(data_at_time, pd.Series):
                    data_at_time = data_at_time.to_dict()

                # Create market event
                market_event = MarketEvent(
                    timestamp=timestamp, symbol=symbol, data=data_at_time
                )

                # Add to event queue
                self.event_queue.add(market_event)
                self.all_market_events.append(market_event)

                # Update strategy's market data cache
                self.strategy.update_market_data(market_event)

                # Add market data to result dictionary
                market_data_by_symbol[symbol] = data_at_time

            except KeyError:
                # No data for this timestamp for this symbol
                continue

        return market_data_by_symbol

    def _create_funding_events(
        self, timestamp: datetime, market_data: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Create funding events if applicable (for perpetual contracts).

        Args:
            timestamp: Current timestamp
            market_data: Current market data by symbol
        """
        # Check if it's a funding time
        funding_hours = [0, 8, 16]  # Hours when funding occurs
        is_funding_time = False

        # For daily data, assume funding is applied once per day
        if self.freq == "1d":
            is_funding_time = True
        # For hourly data, check if current hour is a funding hour
        elif self.freq == "1h" and timestamp.hour in funding_hours:
            is_funding_time = True

        if not is_funding_time:
            return

        # Create funding events for each symbol
        for symbol, data in market_data.items():
            if "funding_rate" in data and data["funding_rate"] != 0:
                # Create funding event
                funding_event = FundingEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    funding_rate=data["funding_rate"],
                    mark_price=data.get("perp_close", data.get("close", 0.0)),
                )

                # Add to event queue
                self.event_queue.add(funding_event)
                self.all_funding_events.append(funding_event)

                logger.debug(
                    f"Created funding event for {symbol}: "
                    f"{data['funding_rate']:.6f} at {timestamp}"
                )

    def _process_signals(self, signals: List[SignalEvent]) -> List[OrderEvent]:
        """
        Convert signals to orders based on portfolio state.

        Args:
            signals: List of signal events

        Returns:
            List of order events
        """
        orders = []

        for signal in signals:
            # Get symbol, direction, and current timestamp
            symbol = signal.symbol
            direction = signal.direction
            timestamp = signal.timestamp

            # Basic position sizing - fixed size for now
            position_size = 1.0

            # Create market order
            order = OrderEvent(
                timestamp=timestamp,
                symbol=symbol,
                order_type=OrderType.MARKET,
                direction=direction,
                quantity=position_size,
                leverage=self.leverage,
                signal_id=signal.event_id,
            )

            # Add to event queue
            self.event_queue.add(order)
            self.all_order_events.append(order)
            orders.append(order)

            logger.info(
                f"Created order from signal: {direction.value} {position_size} {symbol}"
            )

        return orders

    def run(
        self,
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the backtest.

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            verbose: Whether to print progress

        Returns:
            Dictionary of backtest results
        """
        # Parse dates if string
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Get all timestamps
        timestamps = self._get_all_timestamps(start_date, end_date)
        if not timestamps:
            logger.error("No data found for specified date range")
            return {"error": "No data found for specified date range"}

        # Initialize strategy
        self.strategy.initialize(self.portfolio)

        # Start timing
        start_time = time.time()
        self.is_running = True

        # Track progress
        total_timestamps = len(timestamps)
        last_progress = -1
        last_timestamp = None

        try:
            # Main event loop
            for i, timestamp in enumerate(timestamps):
                # Update current datetime
                self.current_datetime = timestamp

                # Print progress
                if verbose:
                    progress = int(i / total_timestamps * 100)
                    if progress > last_progress:
                        logger.info(f"Progress: {progress}% ({timestamp})")
                        last_progress = progress

                # Create market events
                market_data = self._create_market_events(timestamp)
                if not market_data:
                    continue

                # Create funding events if applicable
                self._create_funding_events(timestamp, market_data)

                # Process events in queue
                while not self.event_queue.is_empty():
                    event = self.event_queue.get()

                    if event.event_type == EventType.MARKET:
                        # Market event - already processed
                        pass

                    elif event.event_type == EventType.SIGNAL:
                        # Signal event - convert to orders
                        self._process_signals([event])

                    elif event.event_type == EventType.ORDER:
                        # Order event - execute through broker
                        self.broker.execute_order(
                            order=event,
                            market_data=market_data.get(event.symbol, {}),
                            timestamp=timestamp,
                            event_queue=self.event_queue,
                        )

                    elif event.event_type == EventType.FILL:
                        # Fill event - update portfolio
                        self.portfolio.update(event)
                        self.stats["total_trades"] += 1
                        self.stats["total_fees"] += event.fees
                        self.stats["total_slippage"] += event.slippage

                    elif event.event_type == EventType.FUNDING:
                        # Funding event - apply to portfolio
                        self.portfolio.apply_funding(event)
                        self.stats["total_funding"] += abs(event.funding_rate)

                # Process any open orders with current market data
                self.broker.process_open_orders(
                    market_data, timestamp, self.event_queue
                )

                # Generate new signals
                signals = self.strategy.calculate_signals(
                    timestamp=timestamp,
                    market_data=market_data,
                    portfolio=self.portfolio,
                )

                # Add signals to event queue
                for signal in signals:
                    self.event_queue.add(signal)
                    self.all_signal_events.append(signal)

                # Update portfolio values
                self.portfolio.update_market_value(timestamp, market_data)

                # Remember this timestamp
                last_timestamp = timestamp

            # End timing
            end_time = time.time()
            elapsed = end_time - start_time

            # Calculate results
            self.results = self._calculate_results(elapsed)

            # Cleanup
            self.strategy.teardown()
            self.is_running = False

            # Log summary
            logger.info(f"Backtest complete. Elapsed time: {elapsed:.2f} seconds")
            logger.info(f"Initial capital: {self.initial_capital:.2f}")
            logger.info(f"Final equity: {self.portfolio.get_total_value():.2f}")
            logger.info(
                f"Total return: {self.results['performance']['total_return_pct']:.2f}%"
            )

            return self.results

        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            self.is_running = False
            raise

    def _calculate_results(self, elapsed_time: float) -> Dict[str, Any]:
        """
        Calculate final backtest results.

        Args:
            elapsed_time: Elapsed time in seconds

        Returns:
            Dictionary of results
        """
        # Get portfolio statistics
        portfolio_stats = self.portfolio.get_statistics()

        # Get equity curve
        equity_curve = self.portfolio.get_equity_curve_df()

        # Get returns
        returns = self.portfolio.get_returns_df()

        # Calculate additional metrics
        win_rate = 0
        if self.stats["total_trades"] > 0:
            win_rate = self.stats["winning_trades"] / self.stats["total_trades"]

        results = {
            "performance": {
                "initial_capital": self.initial_capital,
                "final_equity": portfolio_stats["final_equity"],
                "total_return": portfolio_stats["total_return"],
                "total_return_pct": portfolio_stats["total_return_pct"],
                "annualized_return": self._calculate_annualized_return(
                    portfolio_stats["total_return"], equity_curve
                ),
                "sharpe_ratio": portfolio_stats["sharpe_ratio"],
                "max_drawdown": portfolio_stats["max_drawdown"],
                "max_drawdown_pct": portfolio_stats["max_drawdown_pct"],
                "max_drawdown_duration": portfolio_stats["max_drawdown_duration"],
            },
            "trading": {
                "total_trades": self.stats["total_trades"],
                "winning_trades": self.stats["winning_trades"],
                "losing_trades": self.stats["losing_trades"],
                "win_rate": win_rate,
                "total_fees": self.stats["total_fees"],
                "total_slippage": self.stats["total_slippage"],
                "total_funding": self.stats["total_funding"],
            },
            "equity_curve": equity_curve.to_dict(orient="records"),
            "returns": returns.to_dict(orient="records"),
            "metadata": {
                "symbols": self.symbols,
                "start_date": (
                    equity_curve.index[0].strftime("%Y-%m-%d")
                    if not equity_curve.empty
                    else None
                ),
                "end_date": (
                    equity_curve.index[-1].strftime("%Y-%m-%d")
                    if not equity_curve.empty
                    else None
                ),
                "strategy": self.strategy.name,
                "elapsed_time": elapsed_time,
                "events_processed": sum(self.event_queue.get_stats().values()),
            },
            "positions": self.portfolio.get_positions_summary(),
        }

        return results

    def _calculate_annualized_return(
        self, total_return: float, equity_curve: pd.DataFrame
    ) -> float:
        """
        Calculate annualized return.

        Args:
            total_return: Total return as fraction
            equity_curve: Equity curve dataframe

        Returns:
            Annualized return
        """
        if equity_curve.empty:
            return 0.0

        # Calculate years
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        days = (end_date - start_date).days
        years = days / 365.0

        if years < 0.01:  # Less than ~3-4 days
            return 0.0

        # Calculate annualized return
        annualized = (1 + total_return) ** (1 / years) - 1

        return annualized

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve as DataFrame.

        Returns:
            Equity curve DataFrame
        """
        return self.portfolio.get_equity_curve_df()

    def get_returns(self) -> pd.DataFrame:
        """
        Get returns as DataFrame.

        Returns:
            Returns DataFrame
        """
        return self.portfolio.get_returns_df()

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.

        Returns:
            List of position dictionaries
        """
        return self.portfolio.get_positions_summary()

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get trade history.

        Returns:
            List of trade dictionaries
        """
        trades = []
        for fill_event in self.all_fill_events:
            trades.append(
                {
                    "timestamp": fill_event.timestamp,
                    "symbol": fill_event.symbol,
                    "direction": fill_event.direction.value,
                    "quantity": fill_event.quantity,
                    "price": fill_event.fill_price,
                    "fees": fill_event.fees,
                    "slippage": fill_event.slippage,
                    "order_id": fill_event.order_id,
                }
            )
        return trades

    def save_results(self, filename: str, output_dir: str = "results") -> str:
        """
        Save backtest results to file.

        Args:
            filename: Output filename
            output_dir: Output directory

        Returns:
            Full path to saved file
        """
        if self.results is None:
            logger.error("No results to save. Run backtest first.")
            return ""

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Append .json extension if not present
        if not filename.endswith(".json"):
            filename += ".json"

        # Full file path
        file_path = os.path.join(output_dir, filename)

        # Save results to JSON
        with open(file_path, "w") as f:
            # Need custom serializer for datetime
            def json_serial(obj):
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")

            json.dump(self.results, f, default=json_serial, indent=2)

        logger.info(f"Saved backtest results to {file_path}")
        return file_path
