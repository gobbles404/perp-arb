#!/usr/bin/env python3
"""
Script for running the Rebalancing Beta strategy backtest.
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
from data.data_loader import load_data, filter_date_range
from models.strategies.rebalancing_beta import RebalancingBetaStrategy
from models.base.signals import FundingRateSignal
from models.base.position_sizer import EqualNotionalSizer
from backtesting.vectorbt_backtester import VectorbtBacktester
from analytics.metrics import calculate_performance_metrics, print_performance_summary
from analytics.visualizations import (
    create_performance_charts,
    create_risk_dashboard,
    print_risk_summary,
)
from models.markets.spot_perp import SpotPerpMarket


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Rebalancing Beta strategy backtest"
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
    parser.add_argument(
        "--min-holding-periods",
        type=int,
        default=1,
        help="Minimum periods to hold a position",
    )

    # Rebalancing-specific parameters
    parser.add_argument(
        "--health-min",
        type=float,
        default=2.0,
        help="Minimum health factor before rebalancing",
    )
    parser.add_argument(
        "--health-max",
        type=float,
        default=6.0,
        help="Maximum health factor before rebalancing",
    )
    parser.add_argument(
        "--health-target",
        type=float,
        default=4.0,
        help="Target health factor after rebalancing",
    )
    parser.add_argument(
        "--rebalance-cooldown",
        type=int,
        default=1,
        help="Minimum periods between rebalances",
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
    """Main function to run the Rebalancing Beta strategy backtest."""
    # Parse arguments
    args = parse_arguments()

    print(
        f"Running Rebalancing Beta strategy backtest from {args.start_date or 'earliest available'} to {args.end_date or 'latest available'}"
    )
    print(
        f"Health factor rebalancing thresholds: {args.health_min} - {args.health_max} → {args.health_target}"
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

    # VALIDATE LEVERAGE FIRST
    try:
        # Create a temporary market instance just to validate the leverage
        temp_market = SpotPerpMarket(
            data=filtered_data,
            capital=args.capital,
            leverage=args.leverage,
            fee_rate=args.fee_rate,
            enforce_margin_limits=True,
        )
        print(
            f"Leverage validation successful: {args.leverage}x is within exchange limits."
        )
    except ValueError as e:
        print(f"Leverage validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during leverage validation: {e}")
        sys.exit(1)

    # Create signals for the strategy
    print("Setting up strategy components...")
    entry_signal = FundingRateSignal(threshold=args.funding_threshold)
    exit_signal = FundingRateSignal(threshold=args.exit_threshold, invert=True)
    position_sizer = EqualNotionalSizer()

    # Initialize Beta strategy with rebalancing
    print("Initializing Rebalancing Beta strategy...")
    strategy = RebalancingBetaStrategy(
        entry_signals=[entry_signal],
        exit_signals=[exit_signal],
        position_sizer=position_sizer,
        name="Rebalancing Beta Strategy",
        min_threshold=args.health_min,
        max_threshold=args.health_max,
        target_threshold=args.health_target,
        rebalance_cooldown=args.rebalance_cooldown,
    )

    # Set funding periods multiplier based on data timeframe
    if "1d" in args.data_path:
        strategy.funding_periods_multiplier = 3  # 3 funding periods per day
    elif "8h" in args.data_path:
        strategy.funding_periods_multiplier = 1  # 1 funding period per 8h
    else:
        # Default to 1
        strategy.funding_periods_multiplier = 1

    print(f"Using funding multiplier: {strategy.funding_periods_multiplier}x")

    # Run backtest
    print("Running backtest...")
    try:
        backtester = VectorbtBacktester(
            strategy=strategy,
            data=filtered_data,
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

        # Print rebalancing statistics if available
        if hasattr(strategy, "get_rebalancing_statistics"):
            rebalance_stats = strategy.get_rebalancing_statistics()
            print("\n=== REBALANCING STATISTICS ===")
            print(f"Total Rebalances: {rebalance_stats['total_rebalances']}")
            print(f"  Reduce Risk Actions: {rebalance_stats['reduce_risk_count']}")
            print(f"  Increase Risk Actions: {rebalance_stats['increase_risk_count']}")
            print(f"Total Rebalancing Fees: ${rebalance_stats['total_fees']:.2f}")
            print(
                f"Average Adjustment Size: {rebalance_stats['avg_adjustment_pct']*100:.2f}%"
            )
            print(
                f"Average Health Improvement: {rebalance_stats['avg_health_improvement']:.2f}"
            )

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
                create_risk_dashboard(results, metrics, args.output_dir)
                print_risk_summary(results)
            except Exception as e:
                print(f"Error creating charts: {str(e)}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_path = f"{args.output_dir}/rebalancing_strategy_results_{timestamp}.json"

        # Prepare rebalancing data for serialization
        if hasattr(strategy, "rebalance_history"):
            rebalance_history_serializable = []
            for event in strategy.rebalance_history:
                # Convert datetime to string if present
                event_copy = event.copy()
                if isinstance(event_copy.get("timestamp"), datetime):
                    event_copy["timestamp"] = event_copy["timestamp"].strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                rebalance_history_serializable.append(event_copy)
        else:
            rebalance_history_serializable = []

        # Convert any non-serializable objects to serializable formats
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
                    "parameters": {
                        "funding_threshold": args.funding_threshold,
                        "exit_threshold": args.exit_threshold,
                        "min_holding_periods": args.min_holding_periods,
                        "capital": args.capital,
                        "leverage": args.leverage,
                        "fee_rate": args.fee_rate,
                        "health_min": args.health_min,
                        "health_max": args.health_max,
                        "health_target": args.health_target,
                        "rebalance_cooldown": args.rebalance_cooldown,
                    },
                    "rebalancing": {
                        "total_rebalances": rebalance_stats.get("total_rebalances", 0),
                        "reduce_risk_count": rebalance_stats.get(
                            "reduce_risk_count", 0
                        ),
                        "increase_risk_count": rebalance_stats.get(
                            "increase_risk_count", 0
                        ),
                        "total_fees": rebalance_stats.get("total_fees", 0),
                        "avg_adjustment_pct": rebalance_stats.get(
                            "avg_adjustment_pct", 0
                        ),
                        "avg_health_improvement": rebalance_stats.get(
                            "avg_health_improvement", 0
                        ),
                        "history": rebalance_history_serializable,
                    },
                },
                f,
                indent=4,
            )

        print(f"Results saved to {result_path}")

    except Exception as e:
        print(f"Error during backtest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
