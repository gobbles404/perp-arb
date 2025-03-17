# backtesting/analysis/visualization.py
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import os
from datetime import datetime

from core.logger import get_logger

# Set up logger
logger = get_logger(__name__)


def plot_equity_curve(
    equity_data: Union[pd.DataFrame, pd.Series],
    benchmark_data: Optional[Union[pd.DataFrame, pd.Series]] = None,
    title: str = "Equity Curve",
    ylabel: str = "Portfolio Value",
    figsize: Tuple[int, int] = (10, 6),
    filename: Optional[str] = None,
    output_dir: str = "results",
) -> plt.Figure:
    """
    Plot equity curve with optional benchmark comparison.

    Args:
        equity_data: DataFrame or Series with equity values
        benchmark_data: DataFrame or Series with benchmark values (optional)
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size as (width, height)
        filename: Output filename (if saving)
        output_dir: Output directory

    Returns:
        Figure object
    """
    # Convert to Series if DataFrame
    if isinstance(equity_data, pd.DataFrame):
        if "equity" in equity_data.columns:
            equity_series = equity_data["equity"]
        else:
            equity_series = equity_data.iloc[:, 0]  # Use first column
            logger.warning(f"No 'equity' column found, using {equity_data.columns[0]}")
    else:
        equity_series = equity_data

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot equity curve
    ax.plot(
        equity_series.index, equity_series.values, "b-", linewidth=2, label="Strategy"
    )

    # Plot benchmark if provided
    if benchmark_data is not None:
        if isinstance(benchmark_data, pd.DataFrame):
            benchmark_series = benchmark_data.iloc[:, 0]
        else:
            benchmark_series = benchmark_data

        # Normalize benchmark to same starting value
        start_ratio = equity_series.iloc[0] / benchmark_series.iloc[0]
        normalized_benchmark = benchmark_series * start_ratio

        ax.plot(
            normalized_benchmark.index,
            normalized_benchmark.values,
            "r--",
            linewidth=1.5,
            label="Benchmark",
        )

        # Add legend
        ax.legend()

    # Format x-axis with dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    # Format y-axis with comma separator
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Add grid, title and labels
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")

    # Adjust layout
    plt.tight_layout()

    # Save figure if filename provided
    if filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Append .png extension if not present
        if not filename.endswith(".png"):
            filename += ".png"

        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150)
        logger.info(f"Saved equity curve plot to {filepath}")

    return fig


def plot_drawdown(
    equity_data: Union[pd.DataFrame, pd.Series],
    title: str = "Drawdown",
    figsize: Tuple[int, int] = (10, 6),
    filename: Optional[str] = None,
    output_dir: str = "results",
) -> plt.Figure:
    """
    Plot drawdown chart.

    Args:
        equity_data: DataFrame or Series with equity values
        title: Plot title
        figsize: Figure size as (width, height)
        filename: Output filename (if saving)
        output_dir: Output directory

    Returns:
        Figure object
    """
    # Convert to Series if DataFrame
    if isinstance(equity_data, pd.DataFrame):
        if "equity" in equity_data.columns:
            equity_series = equity_data["equity"]
        else:
            equity_series = equity_data.iloc[:, 0]  # Use first column
    else:
        equity_series = equity_data

    # Calculate drawdown series
    running_max = equity_series.cummax()
    drawdown = (equity_series / running_max - 1) * 100  # Convert to percentage

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot drawdown
    ax.fill_between(
        drawdown.index, drawdown.values, 0, where=(drawdown < 0), color="red", alpha=0.3
    )
    ax.plot(drawdown.index, drawdown.values, "r-", linewidth=1)

    # Format x-axis with dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # Add grid, title and labels
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")

    # Set y-axis limits
    min_dd = drawdown.min()
    ax.set_ylim([min(min_dd * 1.1, -1), 1])  # Add some margin below

    # Add zero line
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.2)

    # Add max drawdown annotation
    max_dd_value = drawdown.min()
    max_dd_date = drawdown.idxmin()

    ax.annotate(
        f"Max DD: {max_dd_value:.1f}%",
        xy=(max_dd_date, max_dd_value),
        xytext=(max_dd_date, max_dd_value * 0.8),
        arrowprops=dict(arrowstyle="->", color="black", alpha=0.7),
        ha="center",
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure if filename provided
    if filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Append .png extension if not present
        if not filename.endswith(".png"):
            filename += ".png"

        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150)
        logger.info(f"Saved drawdown plot to {filepath}")

    return fig


def plot_returns_histogram(
    returns_data: Union[pd.DataFrame, pd.Series],
    title: str = "Returns Distribution",
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 6),
    filename: Optional[str] = None,
    output_dir: str = "results",
) -> plt.Figure:
    """
    Plot histogram of returns.

    Args:
        returns_data: DataFrame or Series with returns
        title: Plot title
        bins: Number of histogram bins
        figsize: Figure size as (width, height)
        filename: Output filename (if saving)
        output_dir: Output directory

    Returns:
        Figure object
    """
    # Convert to Series if DataFrame
    if isinstance(returns_data, pd.DataFrame):
        if "return" in returns_data.columns:
            returns_series = returns_data["return"]
        else:
            returns_series = returns_data.iloc[:, 0]  # Use first column
            logger.warning(f"No 'return' column found, using {returns_data.columns[0]}")
    else:
        returns_series = returns_data

    # Convert to percentage
    returns_pct = returns_series * 100

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    n, bins, patches = ax.hist(
        returns_pct, bins=bins, alpha=0.7, color="blue", edgecolor="black"
    )

    # Color negative returns red and positive returns green
    for i in range(len(patches)):
        if bins[i] < 0:
            patches[i].set_facecolor("red")
        else:
            patches[i].set_facecolor("green")

    # Add mean line
    mean_return = returns_pct.mean()
    ax.axvline(
        mean_return,
        color="black",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean: {mean_return:.2f}%",
    )

    # Add standard deviation lines
    std_return = returns_pct.std()
    ax.axvline(
        mean_return + std_return,
        color="gray",
        linestyle="dotted",
        linewidth=1,
        label=f"+1 Std: {mean_return + std_return:.2f}%",
    )
    ax.axvline(
        mean_return - std_return,
        color="gray",
        linestyle="dotted",
        linewidth=1,
        label=f"-1 Std: {mean_return - std_return:.2f}%",
    )

    # Add legend
    ax.legend()

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # Add grid, title and labels
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")

    # Adjust layout
    plt.tight_layout()

    # Save figure if filename provided
    if filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Append .png extension if not present
        if not filename.endswith(".png"):
            filename += ".png"

        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150)
        logger.info(f"Saved returns histogram plot to {filepath}")

    return fig


def plot_basis_and_position(
    market_data: pd.DataFrame,
    trades: Optional[List[Dict[str, Any]]] = None,
    title: str = "Basis and Position",
    figsize: Tuple[int, int] = (12, 8),
    filename: Optional[str] = None,
    output_dir: str = "results",
) -> plt.Figure:
    """
    Plot basis and position size over time.

    Args:
        market_data: DataFrame with 'basis' column
        trades: List of trade dictionaries (optional)
        title: Plot title
        figsize: Figure size as (width, height)
        filename: Output filename (if saving)
        output_dir: Output directory

    Returns:
        Figure object
    """
    if "basis" not in market_data.columns:
        logger.error("No 'basis' column found in market data")
        return None

    # Create figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Plot basis on top subplot
    ax1.plot(market_data.index, market_data["basis"], "b-", linewidth=1.5)

    # Add horizontal lines at 0 and +/- thresholds
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.2)
    ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax1.axhline(y=-1.0, color="red", linestyle="--", alpha=0.5)

    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # Format subplot 1
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)
    ax1.set_ylabel("Basis (%)")

    # Plot positions on bottom subplot if trades provided
    if trades:
        # Extract timestamps and positions from trades
        positions = []
        current_position = 0

        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda x: x["timestamp"])

        for trade in sorted_trades:
            timestamp = trade["timestamp"]
            direction = trade["direction"]
            quantity = trade["quantity"]

            # Update position
            if direction == "BUY":
                current_position += quantity
            else:  # SELL
                current_position -= quantity

            positions.append({"timestamp": timestamp, "position": current_position})

        # Convert to DataFrame
        positions_df = pd.DataFrame(positions)
        positions_df.set_index("timestamp", inplace=True)

        # Resample to match market data frequency
        if not positions_df.empty:
            # Fill in positions between trades
            positions_df = positions_df.reindex(market_data.index, method="ffill")
            positions_df.fillna(0, inplace=True)

            # Plot positions
            ax2.fill_between(
                positions_df.index,
                positions_df["position"],
                0,
                where=(positions_df["position"] > 0),
                color="green",
                alpha=0.3,
                step="post",
            )
            ax2.fill_between(
                positions_df.index,
                positions_df["position"],
                0,
                where=(positions_df["position"] < 0),
                color="red",
                alpha=0.3,
                step="post",
            )
            ax2.plot(
                positions_df.index,
                positions_df["position"],
                "k-",
                linewidth=1,
                drawstyle="steps-post",
            )

    # Format x-axis with dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    # Format subplot 2
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Position Size")
    ax2.set_xlabel("Date")

    # Add zero line for position
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.2)

    # Adjust layout
    plt.tight_layout()

    # Save figure if filename provided
    if filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Append .png extension if not present
        if not filename.endswith(".png"):
            filename += ".png"

        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150)
        logger.info(f"Saved basis and position plot to {filepath}")

    return fig


def generate_backtest_report(
    backtest_results: Dict[str, Any],
    output_dir: str = "results",
    prefix: str = "backtest",
    plot_types: List[str] = ["equity", "drawdown", "returns", "basis"],
) -> Dict[str, str]:
    """
    Generate a set of plots for a backtest report.

    Args:
        backtest_results: Dictionary of backtest results
        output_dir: Output directory
        prefix: Filename prefix
        plot_types: List of plot types to generate

    Returns:
        Dictionary mapping plot types to file paths
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract data from results
    equity_data = None
    returns_data = None
    market_data = None
    trades = None

    if "equity_curve" in backtest_results:
        equity_curve = backtest_results["equity_curve"]
        if isinstance(equity_curve, list):
            equity_data = pd.DataFrame(equity_curve)
            if "timestamp" in equity_data.columns:
                equity_data.set_index("timestamp", inplace=True)

    if "returns" in backtest_results:
        returns = backtest_results["returns"]
        if isinstance(returns, list):
            returns_data = pd.DataFrame(returns)
            if "timestamp" in returns_data.columns:
                returns_data.set_index("timestamp", inplace=True)

    # Extract market data if available
    # This would need to be added to the backtest results

    # Extract trades if available
    if "trades" in backtest_results:
        trades = backtest_results["trades"]

    # Generate plots
    output_files = {}

    if equity_data is not None:
        if "equity" in plot_types:
            # Equity curve
            fig = plot_equity_curve(
                equity_data=equity_data,
                title="Portfolio Equity Curve",
                filename=f"{prefix}_equity",
                output_dir=output_dir,
            )
            plt.close(fig)
            output_files["equity"] = os.path.join(output_dir, f"{prefix}_equity.png")

        if "drawdown" in plot_types:
            # Drawdown
            fig = plot_drawdown(
                equity_data=equity_data,
                title="Portfolio Drawdown",
                filename=f"{prefix}_drawdown",
                output_dir=output_dir,
            )
            plt.close(fig)
            output_files["drawdown"] = os.path.join(
                output_dir, f"{prefix}_drawdown.png"
            )

    if returns_data is not None and "returns" in plot_types:
        # Returns histogram
        fig = plot_returns_histogram(
            returns_data=returns_data,
            title="Returns Distribution",
            filename=f"{prefix}_returns_hist",
            output_dir=output_dir,
        )
        plt.close(fig)
        output_files["returns"] = os.path.join(output_dir, f"{prefix}_returns_hist.png")

    if market_data is not None and "basis" in plot_types:
        # Basis and position
        fig = plot_basis_and_position(
            market_data=market_data,
            trades=trades,
            title="Basis and Position Size",
            filename=f"{prefix}_basis_position",
            output_dir=output_dir,
        )
        if fig:
            plt.close(fig)
            output_files["basis"] = os.path.join(
                output_dir, f"{prefix}_basis_position.png"
            )

    logger.info(f"Generated {len(output_files)} plots for backtest report")
    return output_files
