#!/usr/bin/env python3
"""
Main script for running the funding arbitrage strategy backtest.
"""

import sys
import os
import argparse
import numpy as np
from datetime import datetime
import re
import json
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data.data_loader import load_data, filter_date_range, calculate_metrics
from models.funding_arb import FundingArbStrategy
from backtesting.vectorbt_backtester import VectorbtBacktester as Backtester
from analytics.metrics import calculate_performance_metrics, print_performance_summary
from analytics.visualizations import create_performance_charts

# Import vectorbt utilities
from backtesting.vectorbt_utils import (
    calculate_vectorbt_metrics,
    create_vectorbt_performance_plots,
    analyze_funding_rate_impact,
    analyze_market_conditions,
)


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
        "--end-date",
        type=str,
        default="2025-03-01",
        help="End date for backtest (YYYY-MM-DD) (default: 2025-03-01)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="strategy/results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        help="Timeframe of the data (autodetected from filename if not specified)",
    )
    parser.add_argument(
        "--advanced-metrics",
        action="store_true",
        help="Calculate advanced metrics using vectorbt",
    )

    return parser.parse_args()


def detect_timeframe_from_filename(filename):
    """
    Detect the timeframe from the filename pattern.

    Looks for patterns like BTCUSDT_1d.csv, BTCUSDT_8h.csv, etc.
    Returns the detected timeframe or '8h' as default.
    """
    # Extract the timeframe using regex
    match = re.search(r"_(\d+[dhm])\.", filename)
    if match:
        return match.group(1)
    return "8h"  # Default timeframe


def main():
    """Main function to run the backtest."""
    # Parse arguments
    args = parse_arguments()

    print(
        f"Running backtest from start date: {args.start_date or 'earliest available'} to end date: {args.end_date}"
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Detect timeframe from filename if not specified
    timeframe = args.timeframe
    if not timeframe:
        timeframe = detect_timeframe_from_filename(args.data_path)
        print(f"Detected timeframe: {timeframe}")

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

    # Initialize strategy with the correct timeframe
    print(f"Initializing strategy with timeframe: {timeframe}...")
    strategy = FundingArbStrategy(timeframe=timeframe)

    # Run backtest
    print("Running backtest using vectorbt...")
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

    # Calculate advanced metrics if requested
    if (
        args.advanced_metrics
        and "vbt_portfolio" in results
        and results["vbt_portfolio"] is not None
    ):
        print("\n=== ADVANCED METRICS ===")
        vectorbt_metrics = calculate_vectorbt_metrics(results["vbt_portfolio"])
        for key, value in vectorbt_metrics.items():
            print(f"{key}: {value:.4f}")

        # Calculate funding rate impact
        print("\n=== FUNDING RATE ANALYSIS ===")
        funding_analysis = analyze_funding_rate_impact(results)
        print(
            f"Funding contribution to total profit: {funding_analysis['funding_contribution_pct']:.2f}%"
        )
        print(
            f"Correlation between funding rate and returns: {funding_analysis['funding_return_correlation']:.4f}"
        )
        print(
            f"Average funding rate on profitable days: {funding_analysis['avg_funding_profitable_ann']:.4f}% APR"
        )
        print(
            f"Average funding rate on unprofitable days: {funding_analysis['avg_funding_unprofitable_ann']:.4f}% APR"
        )

        # Calculate market condition analysis
        print("\n=== MARKET CONDITION ANALYSIS ===")
        market_analysis = analyze_market_conditions(results)
        for condition, stats in market_analysis.items():
            print(f"\n{condition.capitalize()} Market ({stats['days']} days):")
            print(f"  Average Daily Return: {stats['avg_return']:.4f}%")
            print(f"  Total Return: {stats['total_return']:.4f}%")
            print(f"  Average Funding Rate: {stats['avg_funding_ann']:.4f}% APR")

    # Generate visualizations
    if not args.no_plot:
        print("Generating visualization charts...")
        create_performance_charts(results, metrics, args.output_dir)

        # Generate vectorbt plots if available
        if (
            args.advanced_metrics
            and "vbt_portfolio" in results
            and results["vbt_portfolio"] is not None
        ):
            print("Generating advanced vectorbt charts...")
            filename = f"{args.output_dir}/vectorbt_plots_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            create_vectorbt_performance_plots(
                results["vbt_portfolio"],
                title="Funding Arbitrage Strategy",
                filename=filename,
            )
            print(f"VectorBT plots saved to {filename}")

    # Save results
    print("Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Save metrics to JSON
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

    # Save advanced metrics if available
    if (
        args.advanced_metrics
        and "vbt_portfolio" in results
        and results["vbt_portfolio"] is not None
    ):
        # Save vectorbt metrics
        with open(f"{args.output_dir}/vectorbt_metrics_{timestamp}.json", "w") as f:
            json.dump(vectorbt_metrics, f, indent=4)

        # Save funding analysis
        with open(f"{args.output_dir}/funding_analysis_{timestamp}.json", "w") as f:
            # Convert numpy types to Python native types
            funding_analysis_serializable = {}
            for k, v in funding_analysis.items():
                if isinstance(v, (float, np.floating)):
                    funding_analysis_serializable[k] = float(v)
                elif isinstance(v, (int, np.integer)):
                    funding_analysis_serializable[k] = int(v)
                else:
                    funding_analysis_serializable[k] = v
            json.dump(funding_analysis_serializable, f, indent=4)

    print(f"\nBacktest complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
