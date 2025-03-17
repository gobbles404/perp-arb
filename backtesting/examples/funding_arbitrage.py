# backtesting/examples/funding_arbitrage.py
"""
Funding Arbitrage Strategy Example

This example demonstrates a funding arbitrage strategy that:
1. Enters positions when funding rates exceed a threshold
2. Takes long positions in markets with negative funding (shorts pay longs)
3. Takes short positions in markets with positive funding (longs pay shorts)
4. Exits when basis flips sign

Usage:
    python -m examples.funding_arbitrage
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import from our backtesting modules
from data.loaders import DataManager
from data.market_data import MarketData
from engine.backtest import Backtest
from engine.strategy import FundingArbitrageStrategy
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

# Default configuration dictionary
DEFAULT_CONFIG = {
    "market_data_dir": "../binance_data_pipeline/data/markets",
    "contract_specs_file": "../binance_data_pipeline/data/contracts/fut_specs.csv",
    "symbols": None,  # None means use first 3 available symbols
    "funding_threshold": 0.01,  # 0.01% funding rate threshold
    "start_date": None,
    "end_date": None,
    "initial_capital": 100000.0,
    "leverage": 3.0,
    "fee_rate": 0.0004,
    "output_dir": "results/funding_arb",
}


def run_funding_arbitrage(config=None):
    """
    Run the funding arbitrage strategy backtest.

    Args:
        config: Configuration dictionary, uses DEFAULT_CONFIG if None

    Returns:
        Dictionary of backtest results
    """
    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Initialize data manager
    logger.info(f"Initializing DataManager with data from {config['market_data_dir']}")
    data_manager = DataManager(config["market_data_dir"], config["contract_specs_file"])

    # Get available symbols if none specified
    symbols = config["symbols"]
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
        return None

    # Create strategy
    strategy = FundingArbitrageStrategy(
        funding_threshold=config["funding_threshold"],
        max_positions=len(symbols),  # Allow one position per symbol
        risk_per_trade=0.05,  # 5% risk per trade
        max_leverage=config["leverage"],
    )

    # Set up backtest
    backtest = Backtest(
        market_data=market_data_dict,
        strategy=strategy,
        initial_capital=config["initial_capital"],
        cash_position_pct=0.3,  # Keep 30% as cash reserve
        fee_rate=config["fee_rate"],
        leverage=config["leverage"],
        freq="1d",  # Daily data
    )

    # Parse dates
    start = pd.to_datetime(config["start_date"]) if config["start_date"] else None
    end = pd.to_datetime(config["end_date"]) if config["end_date"] else None

    # Run backtest
    logger.info(
        f"Running backtest from {config['start_date'] or 'start'} to {config['end_date'] or 'end'}"
    )
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
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    # Save results
    results_file = backtest.save_results(
        filename=f"funding_arb_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        output_dir=config["output_dir"],
    )

    # Generate plots
    strategy_name = strategy.name.lower().replace(" ", "_")
    prefix = f"{strategy_name}_{datetime.now().strftime('%Y%m%d')}"

    # Generate and save plots
    logger.info("Generating plots...")
    plot_files = generate_backtest_report(
        backtest_results=results,
        output_dir=config["output_dir"],
        prefix=prefix,
        plot_types=["equity", "drawdown", "returns"],
    )

    # Display equity curve plot
    fig = plot_equity_curve(
        equity_data=equity_curve,
        title=f"Funding Arbitrage Strategy - Equity Curve",
        filename=None,  # Don't save again
    )
    plt.show()

    # Display drawdown plot
    fig = plot_drawdown(
        equity_data=equity_curve,
        title=f"Funding Arbitrage Strategy - Drawdown",
        filename=None,  # Don't save again
    )
    plt.show()

    logger.info(f"Backtest complete. Results saved to {results_file}")
    logger.info(f"Plots saved to {config['output_dir']}")

    return results


def main():
    """Main function to run the example."""
    import sys

    # Simple command line argument handling
    config = DEFAULT_CONFIG.copy()

    # Check if any arguments were passed
    if len(sys.argv) > 1:
        try:
            # Allow setting funding threshold from command line
            config["funding_threshold"] = float(sys.argv[1])
            logger.info(f"Using funding threshold: {config['funding_threshold']}")

            # Allow setting symbols if provided
            if len(sys.argv) > 2:
                config["symbols"] = sys.argv[2].split(",")
                logger.info(f"Using symbols: {config['symbols']}")
        except ValueError:
            logger.error(
                f"Invalid funding threshold: {sys.argv[1]}, using default: {config['funding_threshold']}"
            )

    # Run with the config
    run_funding_arbitrage(config)


if __name__ == "__main__":
    main()
