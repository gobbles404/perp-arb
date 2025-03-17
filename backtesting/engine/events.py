# backtesting/engine/events.py
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from collections import deque

from core.logger import get_logger

# Set up logger
logger = get_logger(__name__)


class EventType(Enum):
    """Enumeration of event types in the backtesting system."""

    MARKET = "MARKET"  # New market data
    SIGNAL = "SIGNAL"  # Strategy signal
    ORDER = "ORDER"  # Order request
    FILL = "FILL"  # Order fill
    FUNDING = "FUNDING"  # Funding payment event


class OrderDirection(Enum):
    """Order direction types."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order execution types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class Event:
    """
    Base event class that all specific event types inherit from.
    """

    def __init__(self, event_type: EventType, timestamp: Optional[datetime] = None):
        """
        Initialize base event.

        Args:
            event_type: Type of event
            timestamp: Event timestamp (defaults to current time if None)
        """
        self.event_type = event_type
        self.timestamp = timestamp or datetime.now()
        self.event_id = str(uuid.uuid4())

    def __str__(self) -> str:
        """String representation of the event."""
        return f"{self.event_type.value} Event (ID: {self.event_id[:8]}, Time: {self.timestamp})"


class MarketEvent(Event):
    """
    Event for new market data updates.
    Triggered when new price data becomes available for processing.
    """

    def __init__(self, timestamp: datetime, symbol: str, data: Dict[str, Any]):
        """
        Initialize market event.

        Args:
            timestamp: Event timestamp
            symbol: Market symbol
            data: Dictionary containing market data (prices, volumes, etc.)
        """
        super().__init__(EventType.MARKET, timestamp)
        self.symbol = symbol
        self.data = data
        logger.debug(f"Created MarketEvent for {symbol} at {timestamp}")

    def __str__(self) -> str:
        """String representation of the market event."""
        return f"MARKET Event: {self.symbol} at {self.timestamp}"


class SignalEvent(Event):
    """
    Event generated when a strategy creates a trading signal.
    Signals indicate trade direction but not specific order parameters.
    """

    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        direction: OrderDirection,
        strength: float = 1.0,
        strategy_id: str = "default",
    ):
        """
        Initialize signal event.

        Args:
            timestamp: Event timestamp
            symbol: Market symbol
            direction: Trade direction (BUY/SELL)
            strength: Signal strength (0.0 to 1.0, where 1.0 is strongest)
            strategy_id: Identifier for the generating strategy
        """
        super().__init__(EventType.SIGNAL, timestamp)
        self.symbol = symbol
        self.direction = direction
        self.strength = max(0.0, min(1.0, strength))  # Clamp to [0, 1]
        self.strategy_id = strategy_id
        logger.debug(
            f"Created SignalEvent: {direction.value} {symbol} "
            f"(strength: {strength:.2f}) from strategy {strategy_id}"
        )

    def __str__(self) -> str:
        """String representation of the signal event."""
        return (
            f"SIGNAL Event: {self.direction.value} {self.symbol} "
            f"(strength: {self.strength:.2f}, strategy: {self.strategy_id})"
        )


class OrderEvent(Event):
    """
    Event for order requests to be sent to the broker.
    Contains specific order parameters including quantity and order type.
    """

    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        order_type: OrderType,
        direction: OrderDirection,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        leverage: float = 1.0,
        signal_id: Optional[str] = None,
    ):
        """
        Initialize order event.

        Args:
            timestamp: Event timestamp
            symbol: Market symbol
            order_type: Type of order (MARKET, LIMIT, etc.)
            direction: Order direction (BUY/SELL)
            quantity: Order quantity
            price: Limit price (required for LIMIT and STOP_LIMIT orders)
            stop_price: Stop price (required for STOP and STOP_LIMIT orders)
            leverage: Order leverage (for margin trading)
            signal_id: Reference to originating signal event ID (if applicable)
        """
        super().__init__(EventType.ORDER, timestamp)
        self.symbol = symbol
        self.order_type = order_type
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.leverage = leverage
        self.signal_id = signal_id

        # Validate order parameters
        self._validate()

        logger.debug(
            f"Created OrderEvent: {direction.value} {quantity} {symbol} "
            f"at {price if price else 'MARKET'} (type: {order_type.value})"
        )

    def _validate(self) -> None:
        """Validate order parameters."""
        if (
            self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]
            and self.price is None
        ):
            raise ValueError(f"{self.order_type.value} orders require a price")

        if (
            self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]
            and self.stop_price is None
        ):
            raise ValueError(f"{self.order_type.value} orders require a stop_price")

        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")

    def __str__(self) -> str:
        """String representation of the order event."""
        price_str = f" @ {self.price}" if self.price else ""
        stop_str = f" (stop: {self.stop_price})" if self.stop_price else ""
        return (
            f"ORDER Event: {self.direction.value} {self.quantity} {self.symbol}"
            f"{price_str}{stop_str} (type: {self.order_type.value})"
        )


class FillEvent(Event):
    """
    Event generated when an order is filled by the broker.
    Contains execution details including fill price, fees, and slippage.
    """

    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        direction: OrderDirection,
        quantity: float,
        fill_price: float,
        fees: float,
        slippage: float,
        order_id: str,
        leverage: float = 1.0,
    ):
        """
        Initialize fill event.

        Args:
            timestamp: Event timestamp
            symbol: Market symbol
            direction: Order direction (BUY/SELL)
            quantity: Filled quantity
            fill_price: Execution price
            fees: Transaction fees
            slippage: Price slippage
            order_id: Reference to originating order event ID
            leverage: Order leverage used
        """
        super().__init__(EventType.FILL, timestamp)
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.fill_price = fill_price
        self.fees = fees
        self.slippage = slippage
        self.order_id = order_id
        self.leverage = leverage

        logger.debug(
            f"Created FillEvent: {direction.value} {quantity} {symbol} "
            f"at {fill_price} (fees: {fees:.4f}, slippage: {slippage:.4f})"
        )

    def __str__(self) -> str:
        """String representation of the fill event."""
        return (
            f"FILL Event: {self.direction.value} {self.quantity} {self.symbol} "
            f"@ {self.fill_price} (fees: {self.fees:.4f})"
        )


class FundingEvent(Event):
    """
    Event for perpetual swap funding payments.
    """

    def __init__(
        self, timestamp: datetime, symbol: str, funding_rate: float, mark_price: float
    ):
        """
        Initialize funding event.

        Args:
            timestamp: Event timestamp
            symbol: Market symbol
            funding_rate: Funding rate (positive means longs pay shorts)
            mark_price: Mark price used for funding calculation
        """
        super().__init__(EventType.FUNDING, timestamp)
        self.symbol = symbol
        self.funding_rate = funding_rate
        self.mark_price = mark_price

        logger.debug(
            f"Created FundingEvent for {symbol}: rate {funding_rate:.6f} at price {mark_price}"
        )

    def __str__(self) -> str:
        """String representation of the funding event."""
        return (
            f"FUNDING Event: {self.symbol} rate {self.funding_rate:.6f} "
            f"at {self.mark_price}"
        )


class EventQueue:
    """
    Queue for managing and processing events in the backtesting system.
    Events are processed in the order they are added to the queue.
    """

    def __init__(self):
        """Initialize an empty event queue."""
        self.queue = deque()
        self.event_counts = {event_type: 0 for event_type in EventType}
        logger.debug("Initialized EventQueue")

    def add(self, event: Event) -> None:
        """
        Add an event to the queue.

        Args:
            event: Event to add
        """
        self.queue.append(event)
        self.event_counts[event.event_type] += 1
        logger.debug(f"Added {event.event_type.value} event to queue")

    def get(self) -> Optional[Event]:
        """
        Get the next event from the queue.

        Returns:
            Next event or None if queue is empty
        """
        if not self.queue:
            return None

        event = self.queue.popleft()
        logger.debug(f"Retrieved {event.event_type.value} event from queue")
        return event

    def peek(self) -> Optional[Event]:
        """
        Look at the next event without removing it.

        Returns:
            Next event or None if queue is empty
        """
        if not self.queue:
            return None
        return self.queue[0]

    def clear(self) -> None:
        """Clear all events from the queue."""
        queue_size = len(self.queue)
        self.queue.clear()
        self.event_counts = {event_type: 0 for event_type in EventType}
        logger.debug(f"Cleared {queue_size} events from queue")

    def __len__(self) -> int:
        """Get the number of events in the queue."""
        return len(self.queue)

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self.queue) == 0

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the events processed.

        Returns:
            Dictionary with counts by event type
        """
        return {
            event_type.value: count for event_type, count in self.event_counts.items()
        }
