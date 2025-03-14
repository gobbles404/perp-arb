#!/usr/bin/env python3
"""
Main script for running the adaptive funding arbitrage strategy backtest.
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
from models.adaptive_funding_arb import AdaptiveFundingArbStrategy
from backtesting.vectorbt_adaptive_backtester import VectorbtAdaptiveBacktester
from analytics.metrics import calculate_performance_metrics, print_performance_summary

# Import the fixed visualization function instead
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
        description="Run adaptive funding arbitrage strategy backtest"
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
        "--funding-threshold",
        type=float,
        default=0.0,
        help="Funding rate threshold for position entry (default: 0.0)",
    )
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=0.0,
        help="Funding rate threshold for position exit (default: 0.0)",
    )
    parser.add_argument(
        "--min-holding-periods",
        type=int,
        default=1,
        help="Minimum number of periods to hold a position (default: 1)",
    )
    parser.add_argument(
        "--advanced-metrics",
        action="store_true",
        help="Calculate advanced metrics using vectorbt",
    )
    parser.add_argument(
        "--compare-standard",
        action="store_true",
        help="Compare with standard strategy",
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


def calculate_trade_statistics(trades):
    """Calculate statistics from a list of trades."""
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "average_profit": 0,
            "average_loss": 0,
            "total_profit": 0,
            "average_duration": 0,
        }

    winning_trades = [t for t in trades if t["net_pnl"] > 0]
    losing_trades = [t for t in trades if t["net_pnl"] <= 0]

    win_rate = len(winning_trades) / len(trades) if trades else 0
    avg_profit = (
        np.mean([t["net_pnl"] for t in winning_trades]) if winning_trades else 0
    )
    avg_loss = np.mean([t["net_pnl"] for t in losing_trades]) if losing_trades else 0
    total_profit = sum(t["net_pnl"] for t in trades)

    # Calculate average duration in days
    durations = []
    for trade in trades:
        if isinstance(trade["entry_date"], str):
            entry_date = datetime.strptime(trade["entry_date"], "%Y-%m-%d %H:%M:%S")
        else:
            entry_date = trade["entry_date"]

        if isinstance(trade["exit_date"], str):
            exit_date = datetime.strptime(trade["exit_date"], "%Y-%m-%d %H:%M:%S")
        else:
            exit_date = trade["exit_date"]

        duration = (exit_date - entry_date).total_seconds() / (
            24 * 60 * 60
        )  # Convert to days
        durations.append(duration)

    avg_duration = np.mean(durations) if durations else 0

    return {
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": win_rate,
        "average_profit": float(avg_profit),
        "average_loss": float(avg_loss),
        "total_profit": float(total_profit),
        "average_duration": float(avg_duration),
    }


def main():
    """Main function to run the adaptive backtest."""
    # Parse arguments
    args = parse_arguments()

    print(
        f"Running adaptive strategy backtest from {args.start_date or 'earliest available'} to {args.end_date}"
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
    print(f"Initializing adaptive strategy with timeframe: {timeframe}...")
    strategy = AdaptiveFundingArbStrategy(
        timeframe=timeframe,
        funding_threshold=args.funding_threshold,
        exit_threshold=args.exit_threshold,
        min_holding_periods=args.min_holding_periods,
    )

    # Run adaptive backtest
    print("Running adaptive backtest using vectorbt...")
    try:
        backtester = VectorbtAdaptiveBacktester(
            strategy=strategy,
            data=prepared_data,
            initial_capital=args.capital,
            leverage=args.leverage,
            fee_rate=args.fee_rate,
        )

        results = backtester.run()
        if results is None:
            print("Adaptive backtest execution failed. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"Error running adaptive backtest: {e}")
        sys.exit(1)

    # Calculate performance metrics
    print("Calculating performance metrics...")
    metrics = calculate_performance_metrics(results)

    # Print performance summary
    print("\n=== ADAPTIVE STRATEGY PERFORMANCE ===")
    print_performance_summary(results, metrics)

    # Print trade statistics
    if "trades" in results and results["trades"]:
        print("\n=== TRADE STATISTICS ===")
        trade_stats = (
            results["trade_stats"]
            if "trade_stats" in results
            else calculate_trade_statistics(results["trades"])
        )

        print(f"Total Trades: {trade_stats['total_trades']}")
        print(
            f"Winning Trades: {trade_stats['winning_trades']} ({trade_stats['win_rate']*100:.2f}%)"
        )
        print(f"Losing Trades: {trade_stats['losing_trades']}")
        print(f"Average Profit on Winners: ${trade_stats['average_profit']:.2f}")
        print(f"Average Loss on Losers: ${trade_stats['average_loss']:.2f}")
        print(f"Average Trade Duration: {trade_stats['average_duration']:.2f} days")

    # Calculate additional metrics if requested
    if args.advanced_metrics:
        print("\n=== ADVANCED METRICS ===")

        # Try to calculate vectorbt metrics if available
        try:
            if (
                "vbt_portfolio_spot" in results
                and results["vbt_portfolio_spot"] is not None
            ):
                # Combine metrics from both portfolios
                vectorbt_metrics_spot = calculate_vectorbt_metrics(
                    results["vbt_portfolio_spot"]
                )
                vectorbt_metrics_perp = calculate_vectorbt_metrics(
                    results["vbt_portfolio_perp"]
                )

                # Print key metrics
                print("Spot Portfolio:")
                for key in [
                    "total_return_pct",
                    "annualized_return",
                    "sharpe_ratio",
                    "max_drawdown",
                ]:
                    if key in vectorbt_metrics_spot:
                        print(f"  {key}: {vectorbt_metrics_spot[key]:.4f}")

                print("\nPerp Portfolio:")
                for key in [
                    "total_return_pct",
                    "annualized_return",
                    "sharpe_ratio",
                    "max_drawdown",
                ]:
                    if key in vectorbt_metrics_perp:
                        print(f"  {key}: {vectorbt_metrics_perp[key]:.4f}")

        except Exception as e:
            print(f"Warning: Could not calculate advanced metrics: {e}")

        # Calculate funding rate impact
        try:
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
        except Exception as e:
            print(f"Warning: Could not analyze funding rate impact: {e}")

        # Calculate market condition analysis
        try:
            print("\n=== MARKET CONDITION ANALYSIS ===")
            market_analysis = analyze_market_conditions(results)
            for condition, stats in market_analysis.items():
                print(f"\n{condition.capitalize()} Market ({stats['days']} days):")
                print(f"  Average Daily Return: {stats['avg_return']:.4f}%")
                print(f"  Total Return: {stats['total_return']:.4f}%")
                print(f"  Average Funding Rate: {stats['avg_funding_ann']:.4f}% APR")
        except Exception as e:
            print(f"Warning: Could not analyze market conditions: {e}")

    # Generate visualizations
    if not args.no_plot:
        try:
            print("Generating visualization charts...")
            create_performance_charts(results, metrics, args.output_dir)
        except Exception as e:
            print(f"Error creating performance charts: {str(e)}")
            print("Continuing with backtest process despite chart error.")

    # Save results
    print("Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Create results directory if it doesn't exist
    result_dir = f"{args.output_dir}/adaptive"
    os.makedirs(result_dir, exist_ok=True)

    # Save metrics to JSON
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (float, np.floating)):
            serializable_metrics[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            serializable_metrics[k] = int(v)
        else:
            serializable_metrics[k] = v

    with open(f"{result_dir}/adaptive_metrics_{timestamp}.json", "w") as f:
        json.dump(serializable_metrics, f, indent=4)

    # Save trade statistics if available
    if "trades" in results and results["trades"]:
        trade_data = []
        for trade in results["trades"]:
            # Create a serializable version of the trade data
            serializable_trade = {}
            for k, v in trade.items():
                if isinstance(v, (datetime, np.datetime64)):
                    serializable_trade[k] = v.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(v, (float, np.floating)):
                    serializable_trade[k] = float(v)
                elif isinstance(v, (int, np.integer)):
                    serializable_trade[k] = int(v)
                else:
                    serializable_trade[k] = v
            trade_data.append(serializable_trade)

        with open(f"{result_dir}/adaptive_trades_{timestamp}.json", "w") as f:
            json.dump(trade_data, f, indent=4)

    print(f"\nAdaptive backtest complete. Results saved to {result_dir}")

    # Compare with standard strategy if requested
    if args.compare_standard:
        # Import and run standard strategy for comparison
        try:
            from models.funding_arb import FundingArbStrategy
            from backtesting.vectorbt_backtester import VectorbtBacktester

            print("\n=== COMPARING WITH STANDARD STRATEGY ===")

            standard_strategy = FundingArbStrategy(timeframe=timeframe)

            # Run standard backtest
            standard_backtester = VectorbtBacktester(
                strategy=standard_strategy,
                data=prepared_data,
                initial_capital=args.capital,
                leverage=args.leverage,
                fee_rate=args.fee_rate,
            )

            standard_results = standard_backtester.run()
            standard_metrics = calculate_performance_metrics(standard_results)

            # Print comparison
            print("\nStrategy Comparison:")
            print(
                f"{'Metric':<25} {'Adaptive':<15} {'Standard':<15} {'Difference':<15}"
            )
            print("-" * 70)

            for key in [
                "total_return_pct",
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown",
                "funding_apr",
            ]:
                if key in metrics and key in standard_metrics:
                    adaptive_val = metrics[key]
                    standard_val = standard_metrics[key]
                    diff = adaptive_val - standard_val
                    diff_str = (
                        f"{diff:+.2f}" if key != "sharpe_ratio" else f"{diff:+.2f}"
                    )

                    print(
                        f"{key:<25} {adaptive_val:<15.2f} {standard_val:<15.2f} {diff_str:<15}"
                    )

            # Save comparison results
            comparison = {
                "adaptive": serializable_metrics,
                "standard": {
                    k: (
                        float(v)
                        if isinstance(v, (float, np.floating))
                        else int(v) if isinstance(v, (int, np.integer)) else v
                    )
                    for k, v in standard_metrics.items()
                },
            }

            with open(f"{result_dir}/strategy_comparison_{timestamp}.json", "w") as f:
                json.dump(comparison, f, indent=4)

            print(
                f"\nComparison results saved to {result_dir}/strategy_comparison_{timestamp}.json"
            )

        except Exception as e:
            print(f"Error comparing with standard strategy: {e}")


if __name__ == "__main__":
    main()
