# backtesting/examples/__init__.py

# Import example functions
from .basic_arb import run_basic_arbitrage
from .zscore_strategy import run_zscore_strategy

# Define exported modules
__all__ = [
    "run_basic_arbitrage",
    "run_zscore_strategy",
]
