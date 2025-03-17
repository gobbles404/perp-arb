# backtesting/engine/portfolio.py
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np

from core.logger import get_logger
from engine.events import FillEvent, OrderDirection, FundingEvent

# Set up logger
logger = get_logger(__name__)


class Position:
    """
    Class representing a single trading position for a specific symbol.
    Tracks quantity, entry price, and P&L for both spot and derivative positions.
    """

    def __init__(self, symbol: str, is_derivative: bool = False):
        """
        Initialize a new position.

        Args:
            symbol: Market symbol
            is_derivative: Whether this is a derivative (future/perp) position
        """
        self.symbol = symbol
        self.is_derivative = is_derivative

        # Position details
        self.quantity = 0.0
        self.direction = None  # Will be set based on quantity sign
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.fees_paid = 0.0

        # For derivatives
        self.leverage = 1.0
        self.liquidation_price = None
        self.margin_used = 0.0
        self.funding_paid = 0.0
        self.funding_received = 0.0

        # Trade history
        self.trades = []

        logger.debug(f"Initialized position for {symbol}")

    def update(self, fill_event: FillEvent) -> float:
        """
        Update position based on a fill event.

        Args:
            fill_event: Fill event with execution details

        Returns:
            Cash flow from the trade (negative for buys, positive for sells)
        """
        # Extract details from fill event
        price = fill_event.fill_price
        quantity = fill_event.quantity
        direction = fill_event.direction
        fees = fill_event.fees

        # Calculate cash flow for this trade
        if direction == OrderDirection.BUY:
            cash_flow = -(price * quantity + fees)
            new_quantity = self.quantity + quantity
        else:  # SELL
            cash_flow = price * quantity - fees
            new_quantity = self.quantity - quantity

        # Calculate P&L for partial close
        realized_pnl = 0.0
        if (self.quantity > 0 and direction == OrderDirection.SELL) or (
            self.quantity < 0 and direction == OrderDirection.BUY
        ):
            # Closing position partially or fully
            close_quantity = min(abs(self.quantity), quantity)
            price_diff = (
                price - self.avg_entry_price
                if self.quantity > 0
                else self.avg_entry_price - price
            )
            realized_pnl = close_quantity * price_diff

            logger.debug(
                f"Realized P&L for {self.symbol}: {realized_pnl:.2f} "
                f"(closed {close_quantity} at {price:.4f}, entry: {self.avg_entry_price:.4f})"
            )

        # Update position
        old_quantity = self.quantity

        # Handle entry into a new position
        if old_quantity == 0:
            self.avg_entry_price = price
            self.quantity = new_quantity
        # Handle adding to existing position
        elif (old_quantity > 0 and new_quantity > old_quantity) or (
            old_quantity < 0 and new_quantity < old_quantity
        ):
            # Adding to position - update average entry price
            if old_quantity > 0:
                self.avg_entry_price = (
                    old_quantity * self.avg_entry_price + quantity * price
                ) / new_quantity
            else:
                self.avg_entry_price = (
                    abs(old_quantity) * self.avg_entry_price + quantity * price
                ) / abs(new_quantity)
            self.quantity = new_quantity
        # Handle reducing position
        elif (old_quantity > 0 and new_quantity < old_quantity) or (
            old_quantity < 0 and new_quantity > old_quantity
        ):
            # Reducing position - keep same average entry price
            self.quantity = new_quantity
        # Handle closing and reversing
        elif (old_quantity > 0 and new_quantity < 0) or (
            old_quantity < 0 and new_quantity > 0
        ):
            # Position reversed - set new entry price
            self.avg_entry_price = price
            self.quantity = new_quantity

        # Update realized P&L
        self.realized_pnl += realized_pnl

        # Update fees paid
        self.fees_paid += fees

        # For derivatives, update leverage and margin
        if self.is_derivative:
            self.leverage = fill_event.leverage
            self.margin_used = abs(self.quantity * self.avg_entry_price / self.leverage)

        # Record trade
        self.trades.append(
            {
                "timestamp": fill_event.timestamp,
                "direction": direction.value,
                "quantity": quantity,
                "price": price,
                "fees": fees,
                "realized_pnl": realized_pnl,
            }
        )

        logger.info(
            f"Updated position for {self.symbol}: {self.quantity} @ {self.avg_entry_price:.4f} "
            f"(realized P&L: {self.realized_pnl:.2f})"
        )

        return cash_flow

    def apply_funding(self, funding_event: FundingEvent) -> float:
        """
        Apply funding payment for perpetual positions.

        Args:
            funding_event: Funding event with rate and mark price

        Returns:
            Cash flow from funding (negative for payments, positive for receipts)
        """
        if not self.is_derivative or self.quantity == 0:
            return 0.0

        funding_rate = funding_event.funding_rate
        mark_price = funding_event.mark_price

        # Calculate funding payment
        # Positive funding rate: longs pay shorts
        # Negative funding rate: shorts pay longs
        funding_amount = -self.quantity * mark_price * funding_rate

        # Update funding totals
        if funding_amount < 0:
            self.funding_paid += abs(funding_amount)
        else:
            self.funding_received += funding_amount

        logger.debug(
            f"Applied funding for {self.symbol}: {funding_amount:.4f} "
            f"(rate: {funding_rate:.6f}, quantity: {self.quantity})"
        )

        return funding_amount

    def update_unrealized_pnl(self, current_price: float) -> float:
        """
        Update unrealized P&L based on current market price.

        Args:
            current_price: Current market price

        Returns:
            Updated unrealized P&L
        """
        if self.quantity == 0:
            self.unrealized_pnl = 0.0
            return 0.0

        if self.quantity > 0:
            # Long position
            self.unrealized_pnl = self.quantity * (current_price - self.avg_entry_price)
        else:
            # Short position
            self.unrealized_pnl = abs(self.quantity) * (
                self.avg_entry_price - current_price
            )

        return self.unrealized_pnl

    def calculate_liquidation_price(self, maintenance_margin: float) -> Optional[float]:
        """
        Calculate the liquidation price for a derivative position.

        Args:
            maintenance_margin: Maintenance margin requirement (as a fraction)

        Returns:
            Liquidation price or None if not applicable
        """
        if not self.is_derivative or self.quantity == 0:
            self.liquidation_price = None
            return None

        # Calculate liquidation price
        # For longs: entry_price * (1 - (1 / leverage) + maintenance_margin)
        # For shorts: entry_price * (1 + (1 / leverage) - maintenance_margin)
        if self.quantity > 0:
            # Long position
            self.liquidation_price = self.avg_entry_price * (
                1 - (1 / self.leverage) + maintenance_margin
            )
        else:
            # Short position
            self.liquidation_price = self.avg_entry_price * (
                1 + (1 / self.leverage) - maintenance_margin
            )

        logger.debug(
            f"Calculated liquidation price for {self.symbol}: {self.liquidation_price:.4f} "
            f"(entry: {self.avg_entry_price:.4f}, leverage: {self.leverage}x)"
        )

        return self.liquidation_price

    def is_liquidated(self, current_price: float) -> bool:
        """
        Check if position would be liquidated at the current price.

        Args:
            current_price: Current market price

        Returns:
            True if position would be liquidated, False otherwise
        """
        if (
            not self.is_derivative
            or self.quantity == 0
            or self.liquidation_price is None
        ):
            return False

        if self.quantity > 0:
            # Long position - liquidated if price falls below liquidation price
            return current_price <= self.liquidation_price
        else:
            # Short position - liquidated if price rises above liquidation price
            return current_price >= self.liquidation_price

    def get_notional_value(self, current_price: float) -> float:
        """
        Get the notional value of the position.

        Args:
            current_price: Current market price

        Returns:
            Notional value of the position
        """
        return abs(self.quantity * current_price)

    def get_margin_ratio(self, current_price: float) -> Optional[float]:
        """
        Calculate current margin ratio for the position.

        Args:
            current_price: Current market price

        Returns:
            Margin ratio as a fraction or None if not applicable
        """
        if not self.is_derivative or self.quantity == 0:
            return None

        notional_value = self.get_notional_value(current_price)
        return self.margin_used / notional_value if notional_value > 0 else None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert position to dictionary for reporting.

        Returns:
            Dictionary representation of the position
        """
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fees_paid": self.fees_paid,
            "is_derivative": self.is_derivative,
            "leverage": self.leverage,
            "liquidation_price": self.liquidation_price,
            "margin_used": self.margin_used,
            "funding_paid": self.funding_paid,
            "funding_received": self.funding_received,
            "total_pnl": self.realized_pnl
            + self.unrealized_pnl
            - self.fees_paid
            + self.funding_received
            - self.funding_paid,
        }

    def __str__(self) -> str:
        """String representation of the position."""
        direction = (
            "LONG" if self.quantity > 0 else "SHORT" if self.quantity < 0 else "FLAT"
        )
        leverage_str = (
            f" {self.leverage}x" if self.is_derivative and self.quantity != 0 else ""
        )

        return (
            f"{self.symbol} {direction}{leverage_str}: {abs(self.quantity):.4f} @ {self.avg_entry_price:.4f} "
            f"(P&L: {self.realized_pnl + self.unrealized_pnl - self.fees_paid:.2f})"
        )


class Portfolio:
    """
    Portfolio class that manages positions and cash balance.
    Tracks overall P&L and provides risk metrics.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        default_leverage: float = 1.0,
        max_leverage: float = 10.0,
    ):
        """
        Initialize the portfolio.

        Args:
            initial_capital: Initial cash balance
            default_leverage: Default leverage for derivative positions
            max_leverage: Maximum allowed leverage
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.default_leverage = default_leverage
        self.max_leverage = max_leverage

        # Position management
        self.positions: Dict[str, Position] = {}

        # P&L tracking
        self.equity_curve = []
        self.daily_returns = []
        self.total_fees = 0.0
        self.total_funding = 0.0
        self.high_water_mark = initial_capital

        # Risk metrics
        self.max_drawdown = 0.0
        self.max_drawdown_duration = 0
        self.current_drawdown_start = None

        logger.info(
            f"Initialized portfolio with {initial_capital:.2f} initial capital "
            f"(default leverage: {default_leverage}x, max: {max_leverage}x)"
        )

    def get_position(self, symbol: str, is_derivative: bool = False) -> Position:
        """
        Get a position by symbol, creating it if it doesn't exist.

        Args:
            symbol: Market symbol
            is_derivative: Whether this is a derivative position

        Returns:
            Position object
        """
        position_key = f"{symbol}_{'derivative' if is_derivative else 'spot'}"

        if position_key not in self.positions:
            self.positions[position_key] = Position(symbol, is_derivative)
            logger.debug(f"Created new position for {position_key}")

        return self.positions[position_key]

    def update(self, fill_event: FillEvent) -> None:
        """
        Update portfolio based on a fill event.

        Args:
            fill_event: Fill event with execution details
        """
        symbol = fill_event.symbol
        leverage = fill_event.leverage

        # Determine if this is a derivative
        is_derivative = leverage > 1.0

        # Get position
        position = self.get_position(symbol, is_derivative)

        # Update position and get cash flow
        cash_flow = position.update(fill_event)

        # Update cash balance
        self.cash += cash_flow

        # Update fees
        self.total_fees += fill_event.fees

        logger.info(
            f"Updated portfolio after fill for {symbol}: "
            f"cash flow {cash_flow:.2f}, new cash balance {self.cash:.2f}"
        )

    def apply_funding(self, funding_event: FundingEvent) -> None:
        """
        Apply funding payment to relevant position.

        Args:
            funding_event: Funding event details
        """
        symbol = funding_event.symbol

        # Get derivative position
        position = self.get_position(symbol, is_derivative=True)

        # Apply funding and get cash flow
        cash_flow = position.apply_funding(funding_event)

        # Update cash balance
        self.cash += cash_flow

        # Update total funding
        if cash_flow > 0:
            self.total_funding += cash_flow
        else:
            self.total_funding -= abs(cash_flow)

        logger.debug(
            f"Applied funding for {symbol}: cash flow {cash_flow:.4f}, "
            f"new cash balance {self.cash:.2f}"
        )

    def update_market_value(
        self, timestamp: datetime, market_data: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Update portfolio market value and unrealized P&L based on current market prices.

        Args:
            timestamp: Current timestamp
            market_data: Dictionary of market data by symbol with current prices
        """
        total_unrealized_pnl = 0.0

        # Update each position
        for position_key, position in self.positions.items():
            symbol = position.symbol

            # Skip if no market data for this symbol
            if symbol not in market_data:
                logger.warning(
                    f"No market data found for {symbol}, skipping P&L update"
                )
                continue

            # Get current price
            if position.is_derivative:
                current_price = market_data[symbol].get("perp_close") or market_data[
                    symbol
                ].get("futures_close")
            else:
                current_price = market_data[symbol].get("spot_close")

            if current_price is None:
                logger.warning(
                    f"No price data found for {position_key}, skipping P&L update"
                )
                continue

            # Update unrealized P&L
            unrealized_pnl = position.update_unrealized_pnl(current_price)
            total_unrealized_pnl += unrealized_pnl

            # Check for liquidation
            if position.is_derivative:
                maintenance_margin = market_data[symbol].get(
                    "maintenance_margin", 0.025
                )  # Default 2.5%
                position.calculate_liquidation_price(maintenance_margin)

                if position.is_liquidated(current_price):
                    logger.warning(
                        f"Position {position_key} would be liquidated at {current_price:.4f} "
                        f"(liquidation price: {position.liquidation_price:.4f})"
                    )

        # Calculate total portfolio value
        total_equity = self.cash + total_unrealized_pnl

        # Update equity curve
        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "cash": self.cash,
                "unrealized_pnl": total_unrealized_pnl,
                "equity": total_equity,
            }
        )

        # Calculate daily return if we have at least two data points
        if len(self.equity_curve) >= 2:
            prev_equity = self.equity_curve[-2]["equity"]
            daily_return = (total_equity / prev_equity) - 1 if prev_equity > 0 else 0
            self.daily_returns.append({"timestamp": timestamp, "return": daily_return})

        # Update drawdown metrics
        self._update_drawdown_metrics(timestamp, total_equity)

        logger.debug(
            f"Updated portfolio market value: equity {total_equity:.2f} "
            f"(cash: {self.cash:.2f}, unrealized P&L: {total_unrealized_pnl:.2f})"
        )

    def _update_drawdown_metrics(self, timestamp: datetime, equity: float) -> None:
        """
        Update drawdown metrics based on current equity.

        Args:
            timestamp: Current timestamp
            equity: Current portfolio equity
        """
        # Update high water mark
        if equity > self.high_water_mark:
            self.high_water_mark = equity
            self.current_drawdown_start = None

        # Calculate current drawdown
        current_drawdown = (
            1 - (equity / self.high_water_mark) if self.high_water_mark > 0 else 0
        )

        # Update max drawdown if applicable
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

            # Start tracking drawdown duration if not already
            if self.current_drawdown_start is None:
                self.current_drawdown_start = timestamp

        # Update drawdown duration
        if self.current_drawdown_start is not None:
            current_duration = (timestamp - self.current_drawdown_start).days
            if current_duration > self.max_drawdown_duration:
                self.max_drawdown_duration = current_duration

    def get_total_value(self) -> float:
        """
        Get total portfolio value including cash and unrealized P&L.

        Returns:
            Total portfolio value
        """
        if not self.equity_curve:
            return self.cash

        return self.equity_curve[-1]["equity"]

    def get_margin_used(self) -> float:
        """
        Get total margin used by all positions.

        Returns:
            Total margin used
        """
        return sum(
            position.margin_used
            for position in self.positions.values()
            if position.is_derivative
        )

    def get_margin_ratio(self) -> float:
        """
        Get overall portfolio margin ratio.

        Returns:
            Margin ratio as a percentage of total capital
        """
        margin_used = self.get_margin_used()
        total_value = self.get_total_value()

        return margin_used / total_value if total_value > 0 else 0

    def get_position_exposure(self) -> Dict[str, float]:
        """
        Get exposure by position as percentage of portfolio.

        Returns:
            Dictionary mapping position keys to exposure percentage
        """
        total_value = self.get_total_value()

        if total_value == 0:
            return {}

        exposures = {}
        for position_key, position in self.positions.items():
            if position.quantity == 0:
                continue

            # For derivatives, use notional value
            if position.is_derivative:
                if len(self.equity_curve) == 0:
                    continue

                # Use last known price
                symbol = position.symbol
                market_data = self.equity_curve[-1].get("market_data", {})
                if symbol not in market_data:
                    continue

                price = market_data[symbol].get("perp_close") or market_data[
                    symbol
                ].get("futures_close")
                if price is None:
                    continue

                exposure = position.get_notional_value(price) / total_value
            else:
                # For spot, use position value
                symbol = position.symbol
                market_data = self.equity_curve[-1].get("market_data", {})
                if symbol not in market_data:
                    continue

                price = market_data[symbol].get("spot_close")
                if price is None:
                    continue

                exposure = (position.quantity * price) / total_value

            exposures[position_key] = exposure

        return exposures

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get portfolio performance statistics.

        Returns:
            Dictionary of performance statistics
        """
        if not self.equity_curve:
            return {
                "initial_capital": self.initial_capital,
                "final_equity": self.cash,
                "total_return": 0,
                "max_drawdown": 0,
                "max_drawdown_duration": 0,
                "sharpe_ratio": 0,
                "total_fees": 0,
                "total_funding": 0,
            }

        # Calculate statistics
        initial_equity = self.initial_capital
        final_equity = self.equity_curve[-1]["equity"]

        # Calculate total return
        total_return = (final_equity / initial_equity) - 1 if initial_equity > 0 else 0

        # Calculate Sharpe ratio if we have daily returns
        sharpe_ratio = 0
        if self.daily_returns:
            returns = [r["return"] for r in self.daily_returns]
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            # Annualized Sharpe ratio (assuming daily returns)
            risk_free_rate = 0.0  # Could make this configurable
            sharpe_ratio = (
                (mean_return - risk_free_rate) / std_return * (252**0.5)
                if std_return > 0
                else 0
            )

        return {
            "initial_capital": initial_equity,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown * 100,
            "max_drawdown_duration": self.max_drawdown_duration,
            "sharpe_ratio": sharpe_ratio,
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "current_cash": self.cash,
            "current_positions": len(
                [p for p in self.positions.values() if p.quantity != 0]
            ),
        }

    def get_equity_curve_df(self) -> pd.DataFrame:
        """
        Get equity curve as a pandas DataFrame.

        Returns:
            DataFrame with equity curve data
        """
        if not self.equity_curve:
            return pd.DataFrame(
                columns=["timestamp", "cash", "unrealized_pnl", "equity"]
            )

        return pd.DataFrame(self.equity_curve).set_index("timestamp")

    def get_returns_df(self) -> pd.DataFrame:
        """
        Get daily returns as a pandas DataFrame.

        Returns:
            DataFrame with daily returns data
        """
        if not self.daily_returns:
            return pd.DataFrame(columns=["timestamp", "return"])

        return pd.DataFrame(self.daily_returns).set_index("timestamp")

    def get_positions_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of current positions.

        Returns:
            List of position dictionaries
        """
        return [
            position.to_dict()
            for position in self.positions.values()
            if position.quantity != 0
        ]

    def __str__(self) -> str:
        """String representation of the portfolio."""
        stats = self.get_statistics()
        return (
            f"Portfolio: {stats['final_equity']:.2f} ({stats['total_return_pct']:+.2f}%) | "
            f"Max DD: {stats['max_drawdown_pct']:.2f}% | "
            f"Sharpe: {stats['sharpe_ratio']:.2f} | "
            f"Active positions: {stats['current_positions']}"
        )
