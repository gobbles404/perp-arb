#!/usr/bin/env python3
"""
Unified CLI for fetching data from Binance.

Usage examples:
  # Fetch spot data
  python bin/fetch.py spot --symbol BTCUSDT --interval 1d --start 2023-06-01 --end 2023-07-01

  # Fetch futures data (both perpetual and all available contract futures)
  python bin/fetch.py futures --symbol BTCUSDT

  # Fetch funding rates
  python bin/fetch.py funding --symbol BTCUSDT

  # Fetch contract details
  python bin/fetch.py contracts --type both

  # Fetch all data types for a single symbol (including all available futures contracts)
  python bin/fetch.py all --symbol BTCUSDT --interval 1d
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add the project root to Python's path to enable imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import core components and fetchers
from binance_data_pipeline.core.logger import get_logger
from binance_data_pipeline.core.config import config
from binance_data_pipeline.fetchers import (
    SpotFetcher,
    FuturesFetcher,
    FundingRatesFetcher,
    ContractDetailsFetcher,
)
from binance_data_pipeline.exceptions import FetcherError

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch data from Binance.")

    # Create subparsers for different data types
    subparsers = parser.add_subparsers(dest="command", help="Data type to fetch")

    # Common arguments for most fetchers
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--symbol",
        type=str,
        default=config.default_symbol,
        help=f"Trading pair symbol (e.g., {config.default_symbol})",
    )
    common_parser.add_argument(
        "--interval",
        type=str,
        help="Time interval(s), comma-separated (e.g., 1d,8h,1h)",
    )
    common_parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
        default=config.default_start_date,
    )
    common_parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
        default=config.default_end_date,
    )

    # Spot data
    spot_parser = subparsers.add_parser(
        "spot", parents=[common_parser], help="Fetch spot market data"
    )

    # Perp and contract futures data
    futures_parser = subparsers.add_parser(
        "futures",
        parents=[common_parser],
        help="Fetch all futures data (perpetual and contracts)",
    )
    futures_parser.add_argument(
        "--skip-contracts",
        action="store_true",
        help="Skip fetching contract futures, only fetch perpetual",
    )

    # Funding rates
    funding_parser = subparsers.add_parser(
        "funding", parents=[common_parser], help="Fetch funding rates"
    )

    # Contract details (no symbol required)
    contracts_parser = subparsers.add_parser(
        "contracts", help="Fetch contract specifications"
    )
    contracts_parser.add_argument(
        "--type",
        type=str,
        choices=["futures", "spot", "both"],
        default="both",
        help="Type of contracts to fetch",
    )

    # All-in-one command to run all fetchers for a symbol
    all_parser = subparsers.add_parser(
        "all", parents=[common_parser], help="Run all fetchers for a single symbol"
    )
    all_parser.add_argument(
        "--skip-contracts",
        action="store_true",
        help="Skip fetching contract futures, only fetch perpetual",
    )

    return parser.parse_args()


def fetch_spot(args):
    """Fetch spot market data."""
    try:
        intervals = args.interval.split(",") if args.interval else None

        fetcher = SpotFetcher(
            symbol=args.symbol,
            intervals=intervals,
            start_date=args.start,
            end_date=args.end,
        )
        results = fetcher.fetch_all()

        logger.info(f"Successfully fetched spot data for {args.symbol}")
        return results
    except Exception as e:
        logger.error(f"Error fetching spot data: {e}")
        raise


def fetch_futures(args):
    """Fetch futures data."""
    try:
        intervals = args.interval.split(",") if args.interval else None

        fetcher = FuturesFetcher(
            symbol=args.symbol,
            intervals=intervals,
            start_date=args.start,
            end_date=args.end,
        )

        # Use fetch_all_contracts to get both perpetual and contracts
        results = fetcher.fetch_all_contracts(
            base_symbol=args.symbol,
            intervals=intervals,
            skip_contracts=getattr(args, "skip_contracts", False),
        )

        logger.info(f"Successfully fetched futures data for {args.symbol}")
        return results
    except Exception as e:
        logger.error(f"Error fetching futures data: {e}")
        raise


def fetch_funding(args):
    """Fetch funding rates."""
    try:
        fetcher = FundingRatesFetcher(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
        )
        results = fetcher.fetch_all()

        logger.info(f"Successfully fetched funding rates for {args.symbol}")
        return results
    except Exception as e:
        logger.error(f"Error fetching funding rates: {e}")
        raise


def fetch_contracts(args):
    """Fetch contract details."""
    try:
        fetcher = ContractDetailsFetcher(contract_type=args.type)
        results = fetcher.fetch_all()

        logger.info(f"Successfully fetched contract details ({args.type})")
        return results
    except Exception as e:
        logger.error(f"Error fetching contract details: {e}")
        raise


def fetch_all(args):
    """Fetch all data types for a symbol."""
    try:
        intervals = args.interval.split(",") if args.interval else None
        skip_contracts = getattr(args, "skip_contracts", False)

        logger.info(f"Running all data fetchers for symbol: {args.symbol}")

        # 1. First fetch contract details
        logger.info("Fetching contract details...")
        contracts_fetcher = ContractDetailsFetcher(contract_type="both")
        contracts_fetcher.fetch_all()

        # 2. Fetch spot data
        logger.info(f"Fetching spot data for {args.symbol}...")
        spot_fetcher = SpotFetcher(
            symbol=args.symbol,
            intervals=intervals,
            start_date=args.start,
            end_date=args.end,
        )
        spot_fetcher.fetch_all()

        # 3. Fetch futures data (both perpetual and contracts)
        logger.info(f"Fetching futures data for {args.symbol}...")
        futures_fetcher = FuturesFetcher(
            symbol=args.symbol,
            intervals=intervals,
            start_date=args.start,
            end_date=args.end,
        )
        futures_fetcher.fetch_all_contracts(
            base_symbol=args.symbol, intervals=intervals, skip_contracts=skip_contracts
        )

        # 4. Fetch funding rates
        logger.info(f"Fetching funding rates for {args.symbol}...")
        funding_fetcher = FundingRatesFetcher(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
        )
        funding_fetcher.fetch_all()

        logger.info(f"All data fetching complete for {args.symbol}")

        # Run the data pipeline to build futures curve and market data
        run_data_pipeline(args.symbol, intervals)

        return True
    except Exception as e:
        logger.error(f"Error in fetch_all command: {e}")
        raise


def run_data_pipeline(symbol, intervals=None):
    """Run the data processing pipeline."""
    try:
        logger.info("Running data pipeline to process the fetched data...")

        # Convert intervals list to space-separated string for subprocess
        if intervals:
            interval_str = " ".join(intervals)
        else:
            interval_str = " ".join(config.default_intervals["futures"])

        # Get the path to pipeline.py script
        pipeline_script = Path(current_dir) / "pipeline.py"

        # Run the Python pipeline script with the symbol and intervals as arguments
        cmd = [sys.executable, str(pipeline_script), symbol, interval_str]
        logger.info(f"Executing: {' '.join(cmd)}")

        # Execute the process and capture both stdout and stderr
        process = subprocess.run(cmd, text=True, capture_output=True)

        # Process output - we'll determine if it's an error based on content, not stream
        if process.stdout:
            for line in process.stdout.splitlines():
                if line.strip():
                    logger.info(f"Pipeline: {line}")

        if process.stderr:
            for line in process.stderr.splitlines():
                line = line.strip()
                if not line:
                    continue

                # Check if the line looks like an actual error or just regular logging
                if " - ERROR - " in line:
                    logger.error(f"Pipeline error: {line}")
                else:
                    # It's just regular logging sent to stderr
                    logger.info(f"Pipeline: {line}")

        # Check return code
        if process.returncode == 0:
            logger.info("Data pipeline completed successfully")
            return True
        else:
            logger.error(f"Pipeline failed with return code {process.returncode}")
            return False

    except Exception as e:
        logger.error(f"Error running data pipeline: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function."""
    args = parse_arguments()

    if not args.command:
        logger.error("No command specified. Use --help to see available commands.")
        return 1

    try:
        # Execute the appropriate command
        if args.command == "spot":
            fetch_spot(args)
        elif args.command == "futures":
            fetch_futures(args)
        elif args.command == "funding":
            fetch_funding(args)
        elif args.command == "contracts":
            fetch_contracts(args)
        elif args.command == "all":
            fetch_all(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1

        return 0  # Success
    except FetcherError as e:
        logger.error(f"Fetcher error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
