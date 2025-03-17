# backtesting/engine/__init__.py

# Import events components
from .events import (
    EventType,
    EventQueue,
    Event,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    FundingEvent,
    OrderDirection,
    OrderType,
)

# Import portfolio components
from .portfolio import Portfolio, Position

# Import broker components
from .broker import (
    Broker,
    SlippageModel,
    FixedSlippageModel,
    RandomSlippageModel,
    VolumeSlippageModel,
)

# Import strategy components
from .strategy import (
    Strategy,
    ZScoreStrategy,
    BasisArbitrageStrategy,
    FundingArbitrageStrategy,
)

# Import backtest components
from .backtest import Backtest

# Define exported modules
__all__ = [
    # Event types
    "EventType",
    "EventQueue",
    "Event",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "FundingEvent",
    "OrderDirection",
    "OrderType",
    # Portfolio
    "Portfolio",
    "Position",
    # Broker
    "Broker",
    "SlippageModel",
    "FixedSlippageModel",
    "RandomSlippageModel",
    "VolumeSlippageModel",
    # Strategy
    "Strategy",
    "ZScoreStrategy",
    "BasisArbitrageStrategy",
    "FundingArbitrageStrategy",
    # Backtest
    "Backtest",
]
