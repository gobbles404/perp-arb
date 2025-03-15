#!/usr/bin/env python3
"""
Unified script for running strategy backtests using the models framework.
"""

import sys
import os
import argparse
import numpy as np
from datetime import datetime
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data.data_loader import load_data, filter_date_range, calculate_metrics
from models.strategies.beta import BetaStrategy
from models.strategies.enhanced_beta import EnhancedBetaStrategy
from models.base.signals import FundingRateSignal
from models.base.position_sizer import EqualNotionalSizer
from backtesting.vectorbt_backtester import VectorbtBacktester
from analytics.metrics import calculate_performance_metrics, print_performance_summary
from analytics.visualizations import create_performance_charts


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run strategy backtest with the models framework"
    )

    # Common parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="beta",
        choices=["beta", "enhanced_beta"],
        help="Strategy type to run",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/markets/BTCUSDT_1d.csv",
        help="Path to data file",
    )
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--leverage", type=float, default=1.0, help="Trading leverage")
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0004,
        help="Trading fee rate (e.g., 0.0004 for 0.04%)",
    )
    parser.add_argument(
        "--start-date", type=str, help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, default=None, help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--funding-threshold",
        type=float,
        default=0.0,
        help="Minimum funding rate to enter position",
    )
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=0.0,
        help="Funding rate threshold to exit position",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="strategy/results",
        help="Directory for output files",
    )

    # Enhanced Beta specific parameters
    enhanced_beta_group = parser.add_argument_group("Enhanced Beta Parameters")
    enhanced_beta_group.add_argument(
        "--use-futures",
        action="store_true",
        help="Use futures contracts in addition to perp",
    )
    enhanced_beta_group.add_argument(
        "--futures-allocation",
        type=float,
        default=0.5,
        help="Portion of short side allocated to futures (vs perp)",
    )

    return parser.parse_args()


def main():
    """Main function to run the strategy backtest."""
    # Parse arguments
    args = parse_arguments()

    print(
        f"Running {args.strategy.replace('_', ' ').title()} strategy backtest from "
        f"{args.start_date or 'earliest available'} to {args.end_date or 'latest available'}"
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading market data...")
    data = load_data(args.data_path)
    if data is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)

    # Filter by date range
    print("Filtering data by date range...")
    filtered_data = filter_date_range(data, args.start_date, args.end_date)
    if filtered_data is None:
        print("Failed to filter data. Exiting.")
        sys.exit(1)

    # Prepare data
    print("Calculating additional metrics...")
    prepared_data = calculate_metrics(filtered_data)

    # Create signals for the strategy
    print("Setting up strategy components...")
    entry_signal = FundingRateSignal(threshold=args.funding_threshold)
    exit_signal = FundingRateSignal(threshold=args.exit_threshold, invert=True)
    position_sizer = EqualNotionalSizer()

    # Initialize the appropriate strategy
    print(f"Initializing {args.strategy.replace('_', ' ').title()} strategy...")
    if args.strategy == "beta":
        strategy = BetaStrategy(
            entry_signals=[entry_signal],
            exit_signals=[exit_signal],
            position_sizer=position_sizer,
            name="Beta Strategy",
        )
    elif args.strategy == "enhanced_beta":
        strategy = EnhancedBetaStrategy(
            entry_signals=[entry_signal],
            exit_signals=[exit_signal],
            position_sizer=position_sizer,
            use_futures=args.use_futures,
            futures_allocation=args.futures_allocation,
            name="Enhanced Beta Strategy",
        )
    else:
        print(f"Unknown strategy type: {args.strategy}")
        sys.exit(1)

    # Set funding periods multiplier based on data timeframe
    if "1d" in args.data_path:
        strategy.funding_periods_multiplier = 3  # 3 funding periods per day
    elif "8h" in args.data_path:
        strategy.funding_periods_multiplier = 1  # 1 funding period per 8h
    else:
        strategy.funding_periods_multiplier = 1

    print(f"Using funding multiplier: {strategy.funding_periods_multiplier}x")

    # Print strategy-specific info
    if args.strategy == "enhanced_beta":
        print(f"Using futures: {'Yes' if args.use_futures else 'No'}")
        if args.use_futures:
            print(f"Futures allocation: {args.futures_allocation * 100:.1f}%")

    # Run backtest
    print("Running backtest...")
    backtester = VectorbtBacktester(
        strategy=strategy,
        data=prepared_data,
        initial_capital=args.capital,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
    )

    results = backtester.run()
    if results is None:
        print("Backtest execution failed. Exiting.")
        sys.exit(1)

    # Calculate performance metrics
    print("Calculating performance metrics...")
    metrics = calculate_performance_metrics(results)

    # Print performance summary
    print_performance_summary(results, metrics)

    # Print trade statistics if available
    if hasattr(strategy, "trade_history") and strategy.trade_history:
        trade_stats = strategy.get_trade_statistics()
        print("\n=== TRADE STATISTICS ===")
        print(f"Total Trades: {trade_stats['total_trades']}")
        print(
            f"Winning Trades: {trade_stats['winning_trades']} ({trade_stats['win_rate']*100:.2f}%)"
        )
        print(f"Losing Trades: {trade_stats['losing_trades']}")
        print(f"Average Profit on Winners: ${trade_stats['average_profit']:.2f}")
        print(f"Average Loss on Losers: ${trade_stats['average_loss']:.2f}")
        print(f"Average Trade Duration: {trade_stats['average_duration']:.2f} days")
        print(f"Profit Factor: {trade_stats['profit_factor']:.2f}")

    # Generate visualizations
    if not args.no_plot:
        print("Generating visualization charts...")
        try:
            create_performance_charts(results, metrics, args.output_dir)
        except Exception as e:
            print(f"Error creating charts: {str(e)}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_path = f"{args.output_dir}/{args.strategy}_results_{timestamp}.json"

    # Gather parameters for the results file
    parameters = {
        "strategy": args.strategy,
        "funding_threshold": args.funding_threshold,
        "exit_threshold": args.exit_threshold,
        "capital": args.capital,
        "leverage": args.leverage,
        "fee_rate": args.fee_rate,
    }

    # Add enhanced beta specific parameters if needed
    if args.strategy == "enhanced_beta":
        parameters.update(
            {
                "use_futures": args.use_futures,
                "futures_allocation": args.futures_allocation,
            }
        )

    # Convert metrics for JSON serialization
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (float, np.floating)):
            serializable_metrics[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            serializable_metrics[k] = int(v)
        else:
            serializable_metrics[k] = v

    with open(result_path, "w") as f:
        json.dump(
            {
                "metrics": serializable_metrics,
                "parameters": parameters,
            },
            f,
            indent=4,
        )

    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
