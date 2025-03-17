# backtesting/examples/basic_arb.py
"""
Basic Basis Arbitrage Strategy Example

This example demonstrates a simple basis arbitrage strategy that:
1. Goes long spot and short perp when basis is positive above threshold
2. Goes short spot and long perp when basis is negative below threshold
3. Exits when basis converges toward zero

Usage:
    python -m examples.basic_arb
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Import from our backtesting modules
from data.loaders import DataManager
from data.market_data import MarketData
from engine.backtest import Backtest
from engine.strategy import BasisArbitrageStrategy
from analysis.metrics import generate_performance_summary
from analysis.visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_histogram,
    generate_backtest_report,
)
from core.logger import get_logger

# Set up logger
logger = get_logger(__name__)


def run_basic_arbitrage(
    market_data_dir: str,
    contract_specs_file: str,
    symbols: list = None,
    basis_threshold: float = 0.5,
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = 100000.0,
    leverage: float = 3.0,
    fee_rate: float = 0.0004,
    output_dir: str = "results/basic_arb",
):
    """
    Run the basic arbitrage strategy backtest.

    Args:
        market_data_dir: Directory containing market data CSV files
        contract_specs_file: Path to contract specifications CSV file
        symbols: List of symbols to trade (if None, uses all available)
        basis_threshold: Basis threshold for trade entry (percentage)
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
    for symbol in symbols:
        try:
            # Load data with contract specs
            data, specs = data_manager.load_data_with_specs(symbol)

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
    strategy = BasisArbitrageStrategy(
        basis_threshold=basis_threshold,
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
        filename=f"basic_arb_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        output_dir=output_dir,
    )

    # Generate plots
    strategy_name = strategy.name.lower().replace(" ", "_")
    prefix = f"{strategy_name}_{datetime.now().strftime('%Y%m%d')}"

    # Generate and save plots
    logger.info("Generating plots...")
    plot_files = generate_backtest_report(
        backtest_results=results,
        output_dir=output_dir,
        prefix=prefix,
        plot_types=["equity", "drawdown", "returns"],
    )

    # Display equity curve plot
    fig = plot_equity_curve(
        equity_data=equity_curve,
        title=f"Basis Arbitrage Strategy - Equity Curve",
        filename=None,  # Don't save again
    )
    plt.show()

    # Display drawdown plot
    fig = plot_drawdown(
        equity_data=equity_curve,
        title=f"Basis Arbitrage Strategy - Drawdown",
        filename=None,  # Don't save again
    )
    plt.show()

    logger.info(f"Backtest complete. Results saved to {results_file}")
    logger.info(f"Plots saved to {output_dir}")

    return results


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(
        description="Run basic arbitrage strategy backtest"
    )

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
        "--basis-threshold",
        type=float,
        default=0.5,
        help="Basis threshold for trade entry (percentage)",
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
        default="results/basic_arb",
        help="Directory for output files",
    )

    args = parser.parse_args()

    # Run backtest
    run_basic_arbitrage(
        market_data_dir=args.data_dir,
        contract_specs_file=args.specs_file,
        symbols=args.symbols,
        basis_threshold=args.basis_threshold,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
