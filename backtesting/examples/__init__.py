# backtesting/examples/__init__.py

# Import example functions
from .basic_arb import run_basic_arbitrage
from .zscore_strategy import run_zscore_strategy
from .funding_arbitrage import run_funding_arbitrage

# Define exported modules
__all__ = [
    "run_basic_arbitrage",
    "run_zscore_strategy",
    "run_funding_arbitrage",
]
