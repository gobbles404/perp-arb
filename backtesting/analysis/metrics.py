# backtesting/analysis/metrics.py
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from core.logger import get_logger

# Set up logger
logger = get_logger(__name__)


def calculate_returns_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate return-based performance metrics.

    Args:
        returns: Series of period returns (not cumulative)

    Returns:
        Dictionary of return metrics
    """
    if returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
        }

    # Total return
    cumulative_return = (1 + returns).prod() - 1

    # Annualize metrics based on frequency
    periods_per_year = _get_periods_per_year(returns.index)

    # Annualized return
    annualized_return = (1 + cumulative_return) ** (periods_per_year / len(returns)) - 1

    # Volatility
    volatility = returns.std()
    annualized_volatility = volatility * np.sqrt(periods_per_year)

    # Sharpe ratio (assuming 0 risk-free rate for simplicity)
    risk_free_rate = 0.0
    mean_return = returns.mean()
    sharpe_ratio = 0.0
    if volatility > 0:
        sharpe_ratio = ((mean_return - risk_free_rate) / volatility) * np.sqrt(
            periods_per_year
        )

    # Sortino ratio (using only negative returns for downside risk)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() if not downside_returns.empty else 0
    sortino_ratio = 0.0
    if downside_deviation > 0:
        sortino_ratio = ((mean_return - risk_free_rate) / downside_deviation) * np.sqrt(
            periods_per_year
        )

    return {
        "total_return": cumulative_return,
        "total_return_pct": cumulative_return * 100,
        "annualized_return": annualized_return,
        "annualized_return_pct": annualized_return * 100,
        "volatility": volatility,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
    }


def calculate_drawdown_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate drawdown metrics from an equity curve.

    Args:
        equity_curve: Series of equity values

    Returns:
        Dictionary of drawdown metrics
    """
    if equity_curve.empty:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_duration": 0,
            "drawdowns": [],
        }

    # Calculate running maximum
    running_max = equity_curve.cummax()

    # Calculate drawdown in percentage terms
    drawdown = equity_curve / running_max - 1

    # Find maximum drawdown
    max_drawdown = drawdown.min()

    # Calculate drawdown duration
    is_in_drawdown = drawdown < 0

    # Find drawdown periods
    drawdown_periods = []
    current_drawdown_start = None

    for date, in_dd in is_in_drawdown.items():
        if in_dd and current_drawdown_start is None:
            # Start of a drawdown period
            current_drawdown_start = date
        elif not in_dd and current_drawdown_start is not None:
            # End of a drawdown period
            drawdown_periods.append((current_drawdown_start, date))
            current_drawdown_start = None

    # Add final drawdown period if still in drawdown
    if current_drawdown_start is not None:
        drawdown_periods.append((current_drawdown_start, equity_curve.index[-1]))

    # Calculate drawdown durations in days
    drawdown_durations = []
    for start, end in drawdown_periods:
        duration = (end - start).days
        if duration > 0:  # Ignore very short drawdowns
            dd_value = drawdown.loc[start:end].min()
            drawdown_durations.append(
                {
                    "start": start,
                    "end": end,
                    "duration_days": duration,
                    "max_drawdown": dd_value,
                    "max_drawdown_pct": dd_value * 100,
                }
            )

    # Find maximum drawdown duration
    max_duration = 0
    if drawdown_durations:
        max_duration = max(d["duration_days"] for d in drawdown_durations)

    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100,
        "max_drawdown_duration": max_duration,
        "drawdowns": drawdown_durations,
    }


def calculate_trade_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate trade-related metrics.

    Args:
        trades: List of trade dictionaries

    Returns:
        Dictionary of trade metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "largest_profit": 0.0,
            "largest_loss": 0.0,
        }

    # Count trades
    total_trades = len(trades)

    # Separate winning and losing trades
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

    # Calculate metrics
    num_winning = len(winning_trades)
    num_losing = len(losing_trades)

    # Win rate
    win_rate = num_winning / total_trades if total_trades > 0 else 0

    # Profit factor
    total_profit = sum(t.get("pnl", 0) for t in winning_trades)
    total_loss = abs(sum(t.get("pnl", 0) for t in losing_trades))
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    # Average profit/loss
    avg_profit = total_profit / num_winning if num_winning > 0 else 0
    avg_loss = total_loss / num_losing if num_losing > 0 else 0

    # Largest profit/loss
    largest_profit = max([t.get("pnl", 0) for t in trades]) if trades else 0
    largest_loss = min([t.get("pnl", 0) for t in trades]) if trades else 0

    return {
        "total_trades": total_trades,
        "winning_trades": num_winning,
        "losing_trades": num_losing,
        "win_rate": win_rate,
        "win_rate_pct": win_rate * 100,
        "profit_factor": profit_factor,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "largest_profit": largest_profit,
        "largest_loss": largest_loss,
    }


def calculate_basis_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate metrics related to basis trading.

    Args:
        data: DataFrame with 'basis' column

    Returns:
        Dictionary of basis metrics
    """
    if data.empty or "basis" not in data.columns:
        return {
            "avg_basis": 0.0,
            "max_basis": 0.0,
            "min_basis": 0.0,
            "basis_volatility": 0.0,
        }

    basis = data["basis"]

    return {
        "avg_basis": basis.mean(),
        "max_basis": basis.max(),
        "min_basis": basis.min(),
        "basis_volatility": basis.std(),
    }


def calculate_funding_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate metrics related to funding rates.

    Args:
        data: DataFrame with 'funding_rate' column

    Returns:
        Dictionary of funding metrics
    """
    if data.empty or "funding_rate" not in data.columns:
        return {
            "avg_funding_rate": 0.0,
            "annualized_funding": 0.0,
            "funding_volatility": 0.0,
        }

    funding_rate = data["funding_rate"]

    # Estimate annual funding based on frequency
    periods_per_year = _get_periods_per_year(data.index)
    annualized_funding = funding_rate.mean() * periods_per_year

    return {
        "avg_funding_rate": funding_rate.mean(),
        "annualized_funding": annualized_funding,
        "annualized_funding_pct": annualized_funding * 100,
        "funding_volatility": funding_rate.std(),
    }


def generate_performance_summary(
    equity_curve: pd.DataFrame,
    trades: Optional[List[Dict[str, Any]]] = None,
    market_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive performance summary.

    Args:
        equity_curve: DataFrame with 'equity' column
        trades: List of trade dictionaries (optional)
        market_data: Market data DataFrame (optional)

    Returns:
        Dictionary with performance metrics
    """
    if equity_curve.empty:
        logger.warning("Empty equity curve, returning default metrics")
        return {
            "returns": {
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "annualized_return": 0.0,
                "annualized_return_pct": 0.0,
            },
            "risk": {
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
            },
            "trades": {"total_trades": 0, "win_rate": 0.0},
        }

    # Extract equity values
    if isinstance(equity_curve, pd.Series):
        equity_values = equity_curve
    else:
        equity_values = (
            equity_curve["equity"]
            if "equity" in equity_curve.columns
            else equity_curve.iloc[:, 0]
        )

    # Calculate period returns
    returns = equity_values.pct_change().dropna()

    # Calculate metrics
    returns_metrics = calculate_returns_metrics(returns)
    drawdown_metrics = calculate_drawdown_metrics(equity_values)

    # Trade metrics if available
    trade_metrics = {}
    if trades:
        trade_metrics = calculate_trade_metrics(trades)

    # Market-specific metrics if available
    market_metrics = {}
    if market_data is not None:
        # Basis metrics
        if "basis" in market_data.columns:
            market_metrics.update(calculate_basis_metrics(market_data))

        # Funding metrics
        if "funding_rate" in market_data.columns:
            market_metrics.update(calculate_funding_metrics(market_data))

    # Combine metrics into summary
    summary = {
        "returns": {
            "total_return": returns_metrics["total_return"],
            "total_return_pct": returns_metrics["total_return_pct"],
            "annualized_return": returns_metrics["annualized_return"],
            "annualized_return_pct": returns_metrics["annualized_return_pct"],
        },
        "risk": {
            "volatility": returns_metrics["volatility"],
            "annualized_volatility": returns_metrics["annualized_volatility"],
            "sharpe_ratio": returns_metrics["sharpe_ratio"],
            "sortino_ratio": returns_metrics["sortino_ratio"],
            "max_drawdown": drawdown_metrics["max_drawdown"],
            "max_drawdown_pct": drawdown_metrics["max_drawdown_pct"],
            "max_drawdown_duration": drawdown_metrics["max_drawdown_duration"],
        },
    }

    # Add trade metrics if available
    if trade_metrics:
        summary["trades"] = {
            "total_trades": trade_metrics["total_trades"],
            "winning_trades": trade_metrics["winning_trades"],
            "losing_trades": trade_metrics["losing_trades"],
            "win_rate": trade_metrics["win_rate"],
            "win_rate_pct": trade_metrics["win_rate_pct"],
            "profit_factor": trade_metrics["profit_factor"],
        }

    # Add market metrics if available
    if market_metrics:
        summary["market"] = market_metrics

    return summary


def _get_periods_per_year(index: pd.DatetimeIndex) -> int:
    """
    Estimate number of periods per year based on index frequency.

    Args:
        index: DatetimeIndex

    Returns:
        Estimated number of periods per year
    """
    if len(index) <= 1:
        return 252  # Default to daily

    # Calculate average time delta
    deltas = []
    for i in range(1, min(len(index), 20)):  # Use up to 20 samples
        delta = (index[i] - index[i - 1]).total_seconds()
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return 252  # Default to daily

    avg_seconds = sum(deltas) / len(deltas)

    # Determine frequency
    if avg_seconds <= 60:  # Minute or less
        return 252 * 6.5 * 60  # Approx trading minutes per year
    elif avg_seconds <= 3600:  # Hour or less
        return 252 * 6.5  # Approx trading hours per year
    elif avg_seconds <= 86400:  # Day or less
        return 252  # Trading days per year
    elif avg_seconds <= 604800:  # Week or less
        return 52  # Weeks per year
    else:  # Month or more
        return 12  # Months per year
