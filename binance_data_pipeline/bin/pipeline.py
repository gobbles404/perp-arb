#!/usr/bin/env python3
"""
Data processing pipeline for Binance market data.

This script runs the complete data processing pipeline:
1. Build futures curve term structure
2. Build consolidated market data

Usage:
  python bin/pipeline.py BTCUSDT "1d 8h 1h"
  python bin/pipeline.py <symbol> <intervals>
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to Python's path to enable imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import core components and processors
from binance_data_pipeline.core.logger import get_logger
from binance_data_pipeline.core.config import config
from binance_data_pipeline.processors import FuturesCurveProcessor, MarketBuilder
from binance_data_pipeline.exceptions import ProcessorError

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the data processing pipeline.")

    parser.add_argument(
        "symbol",
        type=str,
        nargs="?",  # Make optional for backward compatibility
        default=config.default_symbol,
        help=f"Trading pair symbol (e.g., {config.default_symbol})",
    )

    parser.add_argument(
        "intervals",
        type=str,
        nargs="?",  # Make optional for backward compatibility
        default=" ".join(config.default_intervals["futures"]),
        help="Time intervals, space-separated (e.g., '1d 8h 1h')",
    )

    parser.add_argument(
        "--futures-roll",
        type=str,
        default="7d",
        help="Future roll period (e.g., 7d)",
    )

    return parser.parse_args()


def build_futures_curve(symbol, intervals, futures_roll="7d"):
    """
    Build futures term structure from futures contract data.

    Args:
        symbol (str): Trading pair symbol
        intervals (list): List of time intervals
        futures_roll (str): Futures roll period

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Building futures curve for {symbol} with intervals {intervals}")

        processor = FuturesCurveProcessor(
            symbol=symbol, intervals=intervals, futures_roll=futures_roll
        )

        results = processor.process_all()

        # Check if we got any results
        if results and any(df is not None for df in results.values()):
            logger.info("Futures curve building completed successfully")
            return True
        else:
            logger.warning("No futures curve data was generated")
            return False

    except ProcessorError as e:
        logger.error(f"Error building futures curve: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error building futures curve: {e}")
        import traceback

        traceback.print_exc()
        return False


def build_market_data(symbol, intervals):
    """
    Build consolidated market data by merging various data sources.

    Args:
        symbol (str): Trading pair symbol
        intervals (list): List of time intervals

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Building market data for {symbol} with intervals {intervals}")

        processor = MarketBuilder(symbol=symbol, intervals=intervals)

        results = processor.build_all()

        # Check if we got any results
        if results and any(df is not None for df in results.values()):
            logger.info("Market data building completed successfully")
            return True
        else:
            logger.warning("No market data was generated")
            return False

    except ProcessorError as e:
        logger.error(f"Error building market data: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error building market data: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the complete data processing pipeline."""
    args = parse_arguments()

    # Process the intervals - convert from space-separated string to list
    intervals = args.intervals.split()

    logger.info(f"Running data pipeline for {args.symbol} with intervals: {intervals}")

    # Step 1: Build futures curve
    logger.info("=== Step 1: Building futures curve ===")
    futures_result = build_futures_curve(args.symbol, intervals, args.futures_roll)

    # Step 2: Build market data
    logger.info("=== Step 2: Building market data ===")
    market_result = build_market_data(args.symbol, intervals)

    # Summarize results
    if futures_result and market_result:
        logger.info("Pipeline completed successfully!")
        return 0
    elif not futures_result and not market_result:
        logger.error(
            "Pipeline failed: both futures curve and market data building failed"
        )
        return 1
    elif not futures_result:
        logger.warning("Pipeline partially completed: futures curve building failed")
        return 1
    else:  # not market_result
        logger.warning("Pipeline partially completed: market data building failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
