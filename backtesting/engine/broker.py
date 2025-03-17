# backtesting/engine/broker.py
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
import random

from core.logger import get_logger
from engine.events import OrderEvent, FillEvent, OrderDirection, OrderType, EventQueue

# Set up logger
logger = get_logger(__name__)


class Broker:
    """
    Simulates a broker for executing orders with realistic fees and slippage.
    Converts OrderEvents to FillEvents based on market conditions.
    """

    def __init__(
        self,
        fee_rate: float = 0.0004,
        slippage_model: str = "basic",
        slippage_std: float = 0.0001,
        default_slippage: float = 0.0005,
        market_impact_factor: float = 0.1,
        min_fee: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the broker.

        Args:
            fee_rate: Default fee rate as a decimal (e.g., 0.0004 for 0.04%)
            slippage_model: Slippage model to use ('basic', 'random', or 'impact')
            slippage_std: Standard deviation for random slippage model
            default_slippage: Default slippage for basic model
            market_impact_factor: Factor for market impact model
            min_fee: Minimum fee per transaction
            seed: Random seed for reproducibility
        """
        self.fee_rate = fee_rate
        self.slippage_model = slippage_model
        self.slippage_std = slippage_std
        self.default_slippage = default_slippage
        self.market_impact_factor = market_impact_factor
        self.min_fee = min_fee

        # Symbol-specific fee rates
        self.fee_rates: Dict[str, float] = {}

        # For limit orders that haven't been filled yet
        self.open_orders: List[OrderEvent] = []

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        logger.info(
            f"Initialized broker with fee rate {fee_rate:.4%}, "
            f"slippage model '{slippage_model}'"
        )

    def set_fee_rate(self, symbol: str, fee_rate: float) -> None:
        """
        Set custom fee rate for a specific symbol.

        Args:
            symbol: Market symbol
            fee_rate: Fee rate as a decimal
        """
        self.fee_rates[symbol] = fee_rate
        logger.debug(f"Set custom fee rate for {symbol}: {fee_rate:.4%}")

    def _get_fee_rate(self, symbol: str) -> float:
        """
        Get fee rate for a symbol.

        Args:
            symbol: Market symbol

        Returns:
            Fee rate as a decimal
        """
        return self.fee_rates.get(symbol, self.fee_rate)

    def _calculate_slippage(
        self, order: OrderEvent, market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate slippage based on the selected model.

        Args:
            order: Order event
            market_data: Current market data

        Returns:
            Slippage amount as a decimal
        """
        # For market orders, apply slippage
        if order.order_type != OrderType.MARKET:
            return 0.0

        symbol = order.symbol
        direction = order.direction
        quantity = order.quantity

        # Get volume if available
        volume = market_data.get("volume", 1000.0)  # Default value if not available

        # Basic slippage model
        if self.slippage_model == "basic":
            slippage = self.default_slippage

        # Random slippage model - normal distribution around 0
        elif self.slippage_model == "random":
            slippage = random.gauss(0, self.slippage_std)
            if direction == OrderDirection.BUY:
                slippage = abs(slippage)  # Buy orders slip up
            else:
                slippage = -abs(slippage)  # Sell orders slip down

        # Market impact model - slippage increases with order size relative to volume
        elif self.slippage_model == "impact":
            # Calculate order's percentage of volume
            volume_pct = quantity / volume if volume > 0 else 0.01

            # Slippage increases with square root of volume percentage
            base_slippage = self.market_impact_factor * (volume_pct**0.5)

            # Adjust direction
            if direction == OrderDirection.BUY:
                slippage = base_slippage  # Buy orders slip up
            else:
                slippage = -base_slippage  # Sell orders slip down

        else:
            # Unknown model, use basic
            logger.warning(
                f"Unknown slippage model '{self.slippage_model}', using basic"
            )
            slippage = self.default_slippage

        logger.debug(
            f"Calculated slippage for {order.direction.value} {order.quantity} {symbol}: "
            f"{slippage:.6f} ({self.slippage_model} model)"
        )

        return slippage

    def _calculate_fill_price(
        self, order: OrderEvent, market_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Calculate fill price with slippage.

        Args:
            order: Order event
            market_data: Current market data

        Returns:
            Tuple of (fill_price, slippage_amount)
        """
        symbol = order.symbol
        direction = order.direction

        # Get reference price based on order type
        if order.order_type == OrderType.MARKET:
            # For market orders, use current market price
            if direction == OrderDirection.BUY:
                # For buys, use ask price if available, otherwise close price
                ref_price = market_data.get("ask", market_data.get("close", 0.0))
            else:
                # For sells, use bid price if available, otherwise close price
                ref_price = market_data.get("bid", market_data.get("close", 0.0))
        else:
            # For limit orders, use the limit price
            ref_price = order.price

        # Calculate slippage
        slippage_pct = self._calculate_slippage(order, market_data)

        # Apply slippage to price
        slippage_amount = ref_price * slippage_pct
        fill_price = ref_price + slippage_amount

        logger.debug(
            f"Calculated fill price for {order.direction.value} {order.quantity} {symbol}: "
            f"{fill_price:.4f} (reference: {ref_price:.4f}, slippage: {slippage_amount:.4f})"
        )

        return fill_price, slippage_amount

    def _calculate_fees(self, order: OrderEvent, fill_price: float) -> float:
        """
        Calculate transaction fees.

        Args:
            order: Order event
            fill_price: Execution price

        Returns:
            Fee amount
        """
        symbol = order.symbol
        quantity = order.quantity

        # Get fee rate for this symbol
        fee_rate = self._get_fee_rate(symbol)

        # Calculate fee amount
        fee_amount = quantity * fill_price * fee_rate

        # Apply minimum fee if needed
        fee_amount = max(fee_amount, self.min_fee)

        logger.debug(
            f"Calculated fees for {order.direction.value} {quantity} {symbol}: "
            f"{fee_amount:.4f} (rate: {fee_rate:.4%})"
        )

        return fee_amount

    def _check_limit_order_execution(
        self, order: OrderEvent, market_data: Dict[str, Any]
    ) -> bool:
        """
        Check if a limit order would be executed given current market data.

        Args:
            order: Limit order event
            market_data: Current market data

        Returns:
            True if the order would be executed, False otherwise
        """
        if order.order_type == OrderType.MARKET:
            return True

        # Get high and low prices
        high_price = market_data.get("high", market_data.get("close", 0.0))
        low_price = market_data.get("low", market_data.get("close", 0.0))

        # For limit orders
        if order.order_type == OrderType.LIMIT:
            if order.direction == OrderDirection.BUY:
                # Buy limit executes if price drops to or below limit price
                return low_price <= order.price
            else:
                # Sell limit executes if price rises to or above limit price
                return high_price >= order.price

        # For stop orders
        elif order.order_type == OrderType.STOP:
            if order.direction == OrderDirection.BUY:
                # Buy stop executes if price rises to or above stop price
                return high_price >= order.stop_price
            else:
                # Sell stop executes if price drops to or below stop price
                return low_price <= order.stop_price

        # For stop-limit orders
        elif order.order_type == OrderType.STOP_LIMIT:
            # First check if stop is triggered
            stop_triggered = False
            if order.direction == OrderDirection.BUY:
                stop_triggered = high_price >= order.stop_price
            else:
                stop_triggered = low_price <= order.stop_price

            # If stop triggered, check limit price
            if stop_triggered:
                if order.direction == OrderDirection.BUY:
                    return low_price <= order.price
                else:
                    return high_price >= order.price

            return False

        # Unknown order type
        logger.warning(f"Unknown order type: {order.order_type.value}")
        return False

    def execute_order(
        self,
        order: OrderEvent,
        market_data: Dict[str, Any],
        timestamp: datetime,
        event_queue: EventQueue,
    ) -> Optional[FillEvent]:
        """
        Execute an order and create a fill event.

        Args:
            order: Order to execute
            market_data: Current market data
            timestamp: Current timestamp
            event_queue: Event queue for adding the fill event

        Returns:
            Fill event if order was executed, None otherwise
        """
        symbol = order.symbol

        # Check if order can be executed
        if not self._check_limit_order_execution(order, market_data):
            # Add to open orders if not executed
            if order.order_type != OrderType.MARKET:
                self.open_orders.append(order)
                logger.debug(
                    f"Order not executed, added to open orders: "
                    f"{order.direction.value} {order.quantity} {symbol} {order.order_type.value}"
                )
            else:
                logger.warning(
                    f"Market order not executed: {order.direction.value} {order.quantity} {symbol}"
                )
            return None

        # Calculate fill price and slippage
        fill_price, slippage_amount = self._calculate_fill_price(order, market_data)

        # Calculate fees
        fees = self._calculate_fees(order, fill_price)

        # Create fill event
        fill_event = FillEvent(
            timestamp=timestamp,
            symbol=symbol,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            fees=fees,
            slippage=abs(slippage_amount),
            order_id=order.event_id,
            leverage=order.leverage,
        )

        # Add to event queue
        event_queue.add(fill_event)

        logger.info(
            f"Executed order: {order.direction.value} {order.quantity} {symbol} "
            f"at {fill_price:.4f} (fees: {fees:.4f})"
        )

        return fill_event

    def process_open_orders(
        self,
        market_data: Dict[str, Dict[str, Any]],
        timestamp: datetime,
        event_queue: EventQueue,
    ) -> List[FillEvent]:
        """
        Process all open orders with current market data.

        Args:
            market_data: Dictionary mapping symbols to market data
            timestamp: Current timestamp
            event_queue: Event queue for adding fill events

        Returns:
            List of fill events for executed orders
        """
        if not self.open_orders:
            return []

        executed_fills = []
        remaining_orders = []

        for order in self.open_orders:
            symbol = order.symbol

            # Skip if no market data for this symbol
            if symbol not in market_data:
                remaining_orders.append(order)
                continue

            # Try to execute the order
            fill_event = self.execute_order(
                order, market_data[symbol], timestamp, event_queue
            )

            if fill_event:
                executed_fills.append(fill_event)
            else:
                # Keep in open orders if not executed
                remaining_orders.append(order)

        # Update open orders list
        self.open_orders = remaining_orders

        if executed_fills:
            logger.info(f"Processed open orders: {len(executed_fills)} orders filled")

        return executed_fills

    def clear_open_orders(self) -> None:
        """Clear all open orders."""
        num_orders = len(self.open_orders)
        self.open_orders = []
        logger.debug(f"Cleared {num_orders} open orders")

    def get_open_orders_count(self) -> int:
        """Get the number of open orders."""
        return len(self.open_orders)

    def get_order_book_summary(self) -> Dict[str, int]:
        """
        Get summary of open orders by symbol.

        Returns:
            Dictionary mapping symbols to order counts
        """
        summary = {}
        for order in self.open_orders:
            symbol = order.symbol
            if symbol in summary:
                summary[symbol] += 1
            else:
                summary[symbol] = 1

        return summary


class SlippageModel:
    """
    Base class for slippage models that can be used by the broker.
    Allows for customization of slippage calculation.
    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the slippage model.

        Args:
            params: Dictionary of model parameters
        """
        self.params = params or {}

    def calculate_slippage(
        self, order: OrderEvent, market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate slippage for an order.

        Args:
            order: Order event
            market_data: Current market data

        Returns:
            Slippage as a decimal
        """
        raise NotImplementedError("Subclass must implement calculate_slippage")


class FixedSlippageModel(SlippageModel):
    """Applies a fixed slippage rate."""

    def __init__(self, slippage_rate: float = 0.0005):
        """
        Initialize with fixed slippage rate.

        Args:
            slippage_rate: Fixed slippage rate
        """
        super().__init__({"slippage_rate": slippage_rate})
        self.slippage_rate = slippage_rate

    def calculate_slippage(
        self, order: OrderEvent, market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate fixed slippage.

        Args:
            order: Order event
            market_data: Current market data

        Returns:
            Fixed slippage rate with direction adjustment
        """
        # Adjust for direction
        if order.direction == OrderDirection.BUY:
            return self.slippage_rate  # Buy orders slip up
        else:
            return -self.slippage_rate  # Sell orders slip down


class RandomSlippageModel(SlippageModel):
    """Applies random slippage based on normal distribution."""

    def __init__(self, std_dev: float = 0.0002, mean: float = 0.0):
        """
        Initialize with standard deviation and mean.

        Args:
            std_dev: Standard deviation of slippage
            mean: Mean of slippage distribution
        """
        super().__init__({"std_dev": std_dev, "mean": mean})
        self.std_dev = std_dev
        self.mean = mean

    def calculate_slippage(
        self, order: OrderEvent, market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate random slippage.

        Args:
            order: Order event
            market_data: Current market data

        Returns:
            Random slippage value
        """
        # Generate random slippage
        slippage = random.gauss(self.mean, self.std_dev)

        # Adjust for direction
        if order.direction == OrderDirection.BUY:
            return abs(slippage)  # Buy orders slip up
        else:
            return -abs(slippage)  # Sell orders slip down


class VolumeSlippageModel(SlippageModel):
    """Slippage model based on order size relative to volume."""

    def __init__(self, impact_factor: float = 0.1):
        """
        Initialize with impact factor.

        Args:
            impact_factor: Factor controlling impact of order size
        """
        super().__init__({"impact_factor": impact_factor})
        self.impact_factor = impact_factor

    def calculate_slippage(
        self, order: OrderEvent, market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate volume-based slippage.

        Args:
            order: Order event
            market_data: Current market data

        Returns:
            Slippage based on order volume
        """
        # Get volume
        volume = market_data.get("volume", 1000.0)  # Default if not available

        # Calculate volume ratio
        volume_ratio = order.quantity / volume if volume > 0 else 0.01

        # Calculate slippage - square root model
        base_slippage = self.impact_factor * (volume_ratio**0.5)

        # Adjust for direction
        if order.direction == OrderDirection.BUY:
            return base_slippage  # Buy orders slip up
        else:
            return -base_slippage  # Sell orders slip down
