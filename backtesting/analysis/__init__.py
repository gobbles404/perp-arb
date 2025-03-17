# backtesting/analysis/__init__.py

# Import metrics components
from .metrics import (
    calculate_returns_metrics,
    calculate_drawdown_metrics,
    calculate_trade_metrics,
    calculate_basis_metrics,
    calculate_funding_metrics,
    generate_performance_summary,
)

# Import visualization components
from .visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_histogram,
    plot_basis_and_position,
    generate_backtest_report,
)

# Define exported modules
__all__ = [
    # Metrics
    "calculate_returns_metrics",
    "calculate_drawdown_metrics",
    "calculate_trade_metrics",
    "calculate_basis_metrics",
    "calculate_funding_metrics",
    "generate_performance_summary",
    # Visualization
    "plot_equity_curve",
    "plot_drawdown",
    "plot_returns_histogram",
    "plot_basis_and_position",
    "generate_backtest_report",
]
