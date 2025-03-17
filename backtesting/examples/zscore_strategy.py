# backtesting/examples/zscore_strategy.py
"""
Z-Score Mean Reversion Strategy Example

This example demonstrates a z-score based mean reversion strategy for basis arbitrage:
1. Calculates z-score of basis using a rolling window
2. Enters trades when z-score exceeds thresholds (indicating statistical extremes)
3. Exits when z-score reverts toward mean

Usage:
    python -m examples.zscore_strategy
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse

# Import from our backtesting modules
from data.loaders import DataManager
from data.market_data import MarketData
from engine.backtest import Backtest
from engine.strategy import ZScoreStrategy
from analysis.metrics import generate_performance_summary
from analysis.visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_histogram,
)
from core.logger import get_logger

# Set up logger
logger = get_logger(__name__)


def plot_zscore_signals(
    market_data: pd.DataFrame, lookback: int = 20, entry_threshold: float = 2.0
):
    """
    Plot z-score and signal thresholds.

    Args:
        market_data: DataFrame with basis data
        lookback: Lookback window for z-score calculation
        entry_threshold: Z-score threshold for entry
    """
    if "basis" not in market_data.columns:
        logger.error("No basis column in market data")
        return

    # Calculate z-score
    basis = market_data["basis"]
    rolling_mean = basis.rolling(window=lookback).mean()
    rolling_std = basis.rolling(window=lookback).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    zscore = (basis - rolling_mean) / rolling_std

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot z-score
    ax.plot(zscore.index, zscore.values, "b-", linewidth=1.5, label="Z-Score")

    # Add threshold lines
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.2)
    ax.axhline(
        y=entry_threshold,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"+{entry_threshold} Threshold",
    )
    ax.axhline(
        y=-entry_threshold,
        color="green",
        linestyle="--",
        alpha=0.5,
        label=f"-{entry_threshold} Threshold",
    )

    # Add grid, title and labels
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Basis Z-Score ({lookback}-day window)")
    ax.set_ylabel("Z-Score")
    ax.set_xlabel("Date")

    # Add legend
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    return fig


def run_zscore_strategy(
    market_data_dir: str,
    contract_specs_file: str,
    symbols: list = None,
    zscore_lookback: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = 100000.0,
    leverage: float = 3.0,
    fee_rate: float = 0.0004,
    output_dir: str = "results/zscore_strategy",
):
    """
    Run the z-score strategy backtest.

    Args:
        market_data_dir: Directory containing market data CSV files
        contract_specs_file: Path to contract specifications CSV file
        symbols: List of symbols to trade (if None, uses all available)
        zscore_lookback: Lookback window for z-score calculation
        entry_threshold: Z-score threshold for trade entry
        exit_threshold: Z-score threshold for trade exit
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        initial_capital: Initial capital for backtest
        leverage: Leverage for derivative trades
        fee_rate: Trading fee rate
        output_dir: Directory for output files
    """
    # Initialize data manager
    logger.info(f"Initializing DataManager with data from {market_data_dir}")
    data_manager = DataManager(market_data_dir, contract_specs_file)

    # Get available symbols if none specified
    if symbols is None:
        available_symbols = data_manager.get_available_symbols()
        symbols = available_symbols[:3]  # Use first 3 symbols for demo
        logger.info(f"No symbols specified, using: {symbols}")

    # Load market data for specified symbols
    market_data_dict = {}
    raw_data_dict = {}

    for symbol in symbols:
        try:
            # Load data with contract specs
            data, specs = data_manager.load_data_with_specs(symbol)

            # Store raw data for visualization
            raw_data_dict[symbol] = data

            # Create MarketData object
            market = MarketData(symbol, data, specs)
            market_data_dict[symbol] = market

            logger.info(f"Loaded market data for {symbol} with {len(data)} rows")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")

    if not market_data_dict:
        logger.error("No market data loaded, exiting")
        return

    # Create strategy
    strategy = ZScoreStrategy(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        lookback_window=zscore_lookback,
        max_positions=len(symbols),  # Allow one position per symbol
        risk_per_trade=0.05,  # 5% risk per trade
        max_leverage=leverage,
    )

    # Set up backtest
    backtest = Backtest(
        market_data=market_data_dict,
        strategy=strategy,
        initial_capital=initial_capital,
        cash_position_pct=0.3,  # Keep 30% as cash reserve
        fee_rate=fee_rate,
        leverage=leverage,
        freq="1d",  # Daily data
    )

    # Parse dates
    start = pd.to_datetime(start_date) if start_date else None
    end = pd.to_datetime(end_date) if end_date else None

    # Run backtest
    logger.info(f"Running backtest from {start_date or 'start'} to {end_date or 'end'}")
    results = backtest.run(start_date=start, end_date=end, verbose=True)

    # Generate performance summary
    equity_curve = backtest.get_equity_curve()
    trade_history = backtest.get_trade_history()

    summary = generate_performance_summary(
        equity_curve=equity_curve, trades=trade_history
    )

    # Print summary
    logger.info("\n=== Performance Summary ===")
    logger.info(f"Total Return: {summary['returns']['total_return_pct']:.2f}%")
    logger.info(
        f"Annualized Return: {summary['returns']['annualized_return_pct']:.2f}%"
    )
    logger.info(f"Sharpe Ratio: {summary['risk']['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {summary['risk']['max_drawdown_pct']:.2f}%")
    if "trades" in summary:
        logger.info(f"Win Rate: {summary['trades']['win_rate_pct']:.2f}%")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results
    results_file = backtest.save_results(
        filename=f"zscore_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        output_dir=output_dir,
    )

    # Plot equity curve
    equity_fig = plot_equity_curve(
        equity_data=equity_curve,
        title=f"Z-Score Strategy - Equity Curve",
        filename=f"zscore_equity_{datetime.now().strftime('%Y%m%d')}",
        output_dir=output_dir,
    )

    # Plot drawdown
    drawdown_fig = plot_drawdown(
        equity_data=equity_curve,
        title=f"Z-Score Strategy - Drawdown",
        filename=f"zscore_drawdown_{datetime.now().strftime('%Y%m%d')}",
        output_dir=output_dir,
    )

    # Plot returns histogram
    returns = equity_curve["equity"].pct_change().dropna()
    returns_fig = plot_returns_histogram(
        returns_data=returns,
        title=f"Z-Score Strategy - Returns Distribution",
        filename=f"zscore_returns_{datetime.now().strftime('%Y%m%d')}",
        output_dir=output_dir,
    )

    # Plot z-score for first symbol (for visualization)
    if raw_data_dict and symbols:
        first_symbol = symbols[0]
        if first_symbol in raw_data_dict:
            zscore_fig = plot_zscore_signals(
                market_data=raw_data_dict[first_symbol],
                lookback=zscore_lookback,
                entry_threshold=entry_threshold,
            )
            zscore_fig.savefig(
                os.path.join(
                    output_dir,
                    f"zscore_signals_{first_symbol}_{datetime.now().strftime('%Y%m%d')}.png",
                ),
                dpi=150,
            )
            plt.show()

    logger.info(f"Backtest complete. Results saved to {results_file}")
    logger.info(f"Plots saved to {output_dir}")

    return results


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Run Z-Score strategy backtest")

    # Data parameters
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../binance_data_pipeline/data/markets",
        help="Directory containing market data CSV files",
    )
    parser.add_argument(
        "--specs-file",
        type=str,
        default="../binance_data_pipeline/data/contracts/fut_specs.csv",
        help="Path to contract specifications CSV file",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to trade (if not specified, uses first 3 available)",
    )

    # Strategy parameters
    parser.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Lookback window for z-score calculation",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for trade entry",
    )
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=0.0,
        help="Z-score threshold for trade exit",
    )
    parser.add_argument(
        "--leverage", type=float, default=3.0, help="Leverage for derivative trades"
    )

    # Backtest parameters
    parser.add_argument(
        "--start-date", type=str, help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital for backtest",
    )
    parser.add_argument(
        "--fee-rate", type=float, default=0.0004, help="Trading fee rate"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/zscore_strategy",
        help="Directory for output files",
    )

    args = parser.parse_args()

    # Run backtest
    run_zscore_strategy(
        market_data_dir=args.data_dir,
        contract_specs_file=args.specs_file,
        symbols=args.symbols,
        zscore_lookback=args.lookback,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
