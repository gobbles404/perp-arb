#!/usr/bin/env python3
"""
Unified CLI for fetching data from Binance.

Usage examples:
  # Fetch spot data
  python scripts/fetch.py spot --symbol BTCUSDT --interval 1d --start 2023-06-01 --end 2023-07-01

  # Fetch futures data (both perpetual and all available contract futures)
  python scripts/fetch.py futures --symbol BTCUSDT

  # Fetch funding rates
  python scripts/fetch.py funding --symbol BTCUSDT

  # Fetch contract details
  python scripts/fetch.py contracts --type both

  # Fetch all data types for a single symbol (including all available futures contracts)
  python scripts/fetch.py all --symbol BTCUSDT --interval 1d
"""

import argparse
import logging
import sys
import os
import csv
from datetime import datetime
import subprocess

# Add the scripts directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the fetchers
from fetchers import (
    SpotFetcher,
    FuturesFetcher,
    FundingRatesFetcher,
    PremiumIndexFetcher,
    ContractDetailsFetcher,
)

# Import fetcher defaults
from config import (
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_INTERVALS,
    DEFAULT_SYMBOL,
    LOG_LEVEL,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BinanceFetcher")


def get_matching_futures_contracts(base_symbol):
    """
    Read the fut_expirys.csv file and return all futures contracts matching the base symbol.

    Args:
        base_symbol (str): Base symbol, e.g., "BTCUSDT"

    Returns:
        list: List of matching futures contract symbols
    """
    futures_contracts = []

    # Path to the fut_expirys.csv file
    csv_path = os.path.join("data", "contracts", "fut_expirys.csv")

    # Check if the file exists
    if not os.path.exists(csv_path):
        logger.warning(f"Futures contracts CSV file not found: {csv_path}")
        return futures_contracts

    # Read the CSV file
    try:
        with open(csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)

            # Filter for matching symbols
            for row in reader:
                # Check if this contract's pair matches our base symbol
                if row.get("pair") == base_symbol:
                    contract_symbol = row.get("symbol")
                    if (
                        contract_symbol and contract_symbol != base_symbol
                    ):  # Avoid duplicating the perpetual
                        futures_contracts.append(contract_symbol)

    except Exception as e:
        logger.error(f"Error reading futures contracts CSV: {e}")

    return futures_contracts


def fetch_futures_data(symbol, intervals, start_date, end_date):
    """Helper function to fetch futures data for a symbol."""
    logger.info(f"Fetching futures data for {symbol}...")
    futures_fetcher = FuturesFetcher(
        symbol=symbol,
        intervals=intervals,
        start_date=start_date,
        end_date=end_date,
    )
    futures_fetcher.fetch_all()


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
        default=DEFAULT_SYMBOL,  # Use default symbol from config
        help=f"Trading pair symbol (e.g., {DEFAULT_SYMBOL})",
    )
    common_parser.add_argument(
        "--interval",
        type=str,
        help="Time interval(s), comma-separated (e.g., 1d,8h,1h)",
    )
    common_parser.add_argument(
        "--start", type=str, help="Start date (YYYY-MM-DD)", default=DEFAULT_START_DATE
    )
    common_parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
        default=DEFAULT_END_DATE,
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

    # Option to skip fetching contract futures
    futures_parser.add_argument(
        "--skip-contracts",
        action="store_true",
        help="Skip fetching contract futures, only fetch perpetual",
    )

    # Funding rates
    funding_parser = subparsers.add_parser(
        "funding", parents=[common_parser], help="Fetch funding rates"
    )

    # Premium index
    premium_parser = subparsers.add_parser(
        "premium", parents=[common_parser], help="Fetch premium index"
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

    # Option to skip fetching contract futures in all mode
    all_parser.add_argument(
        "--skip-contracts",
        action="store_true",
        help="Skip fetching contract futures, only fetch perpetual",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    if not args.command:
        logger.error("No command specified. Use --help to see available commands.")
        return

    # Process interval argument if provided
    intervals = None
    if hasattr(args, "interval") and args.interval:
        intervals = args.interval.split(",")

    try:
        # Create and run the appropriate fetcher
        if args.command == "spot":
            fetcher = SpotFetcher(
                symbol=args.symbol,
                intervals=intervals or DEFAULT_INTERVALS["spot"],
                start_date=args.start,
                end_date=args.end,
            )
            fetcher.fetch_all()

        elif args.command == "futures":
            # First fetch data for the perpetual contract
            fetch_futures_data(
                args.symbol,
                intervals or DEFAULT_INTERVALS["futures"],
                args.start,
                args.end,
            )

            # Then fetch data for all matching futures contracts if not skipped
            if not getattr(args, "skip_contracts", False):
                futures_contracts = get_matching_futures_contracts(args.symbol)
                if futures_contracts:
                    logger.info(
                        f"Found {len(futures_contracts)} futures contracts for {args.symbol}"
                    )
                    for contract in futures_contracts:
                        fetch_futures_data(
                            contract,
                            intervals or DEFAULT_INTERVALS["futures"],
                            args.start,
                            args.end,
                        )
                else:
                    logger.info(
                        f"No additional futures contracts found for {args.symbol}"
                    )

        elif args.command == "funding":
            fetcher = FundingRatesFetcher(
                symbol=args.symbol,
                intervals=DEFAULT_INTERVALS[
                    "funding"
                ],  # Always use default for funding
                start_date=args.start,
                end_date=args.end,
            )
            fetcher.fetch_all()

        elif args.command == "premium":
            fetcher = PremiumIndexFetcher(
                symbol=args.symbol,
                intervals=intervals or DEFAULT_INTERVALS["premium"],
                start_date=args.start,
                end_date=args.end,
            )
            fetcher.fetch_all()

        elif args.command == "contracts":
            fetcher = ContractDetailsFetcher(contract_type=args.type)
            fetcher.fetch_all()

        elif args.command == "all":
            logger.info(f"Running all data fetchers for symbol: {args.symbol}")

            # 1. First fetch contract details
            logger.info("Fetching contract details...")
            contracts_fetcher = ContractDetailsFetcher(contract_type="both")
            contracts_fetcher.fetch_all()

            # 2. Fetch spot data
            logger.info(f"Fetching spot data for {args.symbol}...")
            spot_fetcher = SpotFetcher(
                symbol=args.symbol,
                intervals=intervals or DEFAULT_INTERVALS["spot"],
                start_date=args.start,
                end_date=args.end,
            )
            spot_fetcher.fetch_all()

            # 3. Fetch perpetual futures data
            fetch_futures_data(
                args.symbol,
                intervals or DEFAULT_INTERVALS["futures"],
                args.start,
                args.end,
            )

            # 3.1 Fetch all matching futures contracts if not skipped
            if not getattr(args, "skip_contracts", False):
                futures_contracts = get_matching_futures_contracts(args.symbol)
                if futures_contracts:
                    logger.info(
                        f"Found {len(futures_contracts)} futures contracts for {args.symbol}"
                    )
                    for contract in futures_contracts:
                        fetch_futures_data(
                            contract,
                            intervals or DEFAULT_INTERVALS["futures"],
                            args.start,
                            args.end,
                        )
                else:
                    logger.info(
                        f"No additional futures contracts found for {args.symbol}"
                    )

            # 4. Fetch funding rates
            logger.info(f"Fetching funding rates for {args.symbol}...")
            funding_fetcher = FundingRatesFetcher(
                symbol=args.symbol,
                intervals=DEFAULT_INTERVALS["funding"],
                start_date=args.start,
                end_date=args.end,
            )
            funding_fetcher.fetch_all()

            # 5. Fetch premium index
            logger.info(f"Fetching premium index for {args.symbol}...")
            premium_fetcher = PremiumIndexFetcher(
                symbol=args.symbol,
                intervals=intervals or DEFAULT_INTERVALS["premium"],
                start_date=args.start,
                end_date=args.end,
            )
            premium_fetcher.fetch_all()

            logger.info(f"All data fetching complete for {args.symbol}")

            # Add this at the end of the "all" command section
            # Run the data pipeline to build futures curve and market data
            logger.info("Running data pipeline to process the fetched data...")
            try:
                # Convert intervals list to space-separated string
                interval_str = (
                    " ".join(intervals)
                    if intervals
                    else " ".join(DEFAULT_INTERVALS["futures"])
                )

                # Run the shell script with the symbol and intervals as arguments
                cmd = ["./data_pipeline.sh", args.symbol, interval_str]
                logger.info(f"Executing: {' '.join(cmd)}")

                result = subprocess.run(cmd, check=True, text=True, capture_output=True)

                # Log the output from the shell script
                if result.stdout:
                    for line in result.stdout.splitlines():
                        logger.info(f"Pipeline: {line}")

                if result.stderr:
                    for line in result.stderr.splitlines():
                        logger.warning(f"Pipeline error: {line}")

                logger.info("Data pipeline completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running data pipeline: {e}")
                if e.stderr:
                    logger.error(f"Error output: {e.stderr}")
            except Exception as e:
                logger.error(f"Error running data pipeline: {e}")

        else:
            logger.error(f"Unknown command: {args.command}")

    except Exception as e:
        logger.error(f"Error executing command {args.command}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
