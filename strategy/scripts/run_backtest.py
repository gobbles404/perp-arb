#!/usr/bin/env python3
"""
Main script for running the funding arbitrage strategy backtest.
"""

import sys
import os
import argparse
import numpy as np  # Added missing numpy import
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data.data_loader import load_data, filter_date_range, calculate_metrics
from models.funding_arb import FundingArbStrategy
from backtesting.backtester import Backtester
from analytics.metrics import calculate_performance_metrics, print_performance_summary
from analytics.visualizations import create_performance_charts


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run funding arbitrage strategy backtest"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/markets/BTCUSDT_8h.csv",
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
        "--end-date", type=str, help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="strategy/results",
        help="Directory for output files",
    )

    return parser.parse_args()


def main():
    """Main function to run the backtest."""
    # Parse arguments
    args = parse_arguments()

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

    # Initialize strategy
    print("Initializing strategy...")
    strategy = FundingArbStrategy()

    # Run backtest
    print("Running backtest...")
    backtester = Backtester(
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

    # Generate visualizations
    if not args.no_plot:
        print("Generating visualization charts...")
        create_performance_charts(results, metrics, args.output_dir)

    # Save results
    print("Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Save metrics to JSON
    import json

    # Convert numpy values to native Python types for JSON serialization
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (float, np.floating)):
            serializable_metrics[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            serializable_metrics[k] = int(v)
        else:
            serializable_metrics[k] = v

    with open(f"{args.output_dir}/backtest_metrics_{timestamp}.json", "w") as f:
        json.dump(serializable_metrics, f, indent=4)

    print(f"\nBacktest complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
