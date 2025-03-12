#!/usr/bin/env python3
"""
Script to build futures term structure from CSV files for multiple intervals.
Usage:
  python build_futures_curve.py  # Uses defaults
  python build_futures_curve.py --symbol ETHUSDT --intervals 1d,8h,1h --futures-roll 7d
"""

import os
import glob
import logging
import csv
import re
import pandas as pd
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default parameters (still use these for backwards compatibility)
DEFAULT_INDEX = os.environ.get(
    "INDEX", "BTCUSDT"
)  # Get from environment variable or use default
DEFAULT_INTERVALS = os.environ.get("INTERVALS", "1d,8h,1h").split(
    ","
)  # Get from environment or use default
DEFAULT_FUTURES_ROLL = "7d"  # Roll to next contract 7 days before delivery


def calculate_roll_date(delivery_date, futures_roll):
    """
    Calculate the date to roll to the next futures contract.

    Args:
        delivery_date (str): Delivery date in Unix milliseconds
        futures_roll (str): Time before delivery to roll, in the format "#[mhdw]"
                           e.g., "7d" for 7 days, "12h" for 12 hours

    Returns:
        int: Roll date in Unix milliseconds
    """
    # Parse the futures_roll string with regex
    match = re.match(r"(\d+)([mhdw])", futures_roll)
    if not match:
        raise ValueError(f"Unsupported futures roll format: {futures_roll}")

    value, unit = int(match.group(1)), match.group(2)

    # Convert to milliseconds
    unit_to_ms = {
        "m": 60 * 1000,  # minutes to ms
        "h": 60 * 60 * 1000,  # hours to ms
        "d": 24 * 60 * 60 * 1000,  # days to ms
        "w": 7 * 24 * 60 * 60 * 1000,  # weeks to ms
    }

    if unit not in unit_to_ms:
        raise ValueError(f"Unsupported time unit: {unit}")

    ms_to_subtract = value * unit_to_ms[unit]

    # Convert delivery_date to int if it's a string
    delivery_date_ms = int(delivery_date)

    # Calculate roll date
    roll_date_ms = delivery_date_ms - ms_to_subtract

    return roll_date_ms


def load_expiry_data(file_path):
    """Load futures expiry data from CSV file."""
    expiry_data = {}

    try:
        with open(file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Create a key in the format INDEX_DATE (e.g., BTCUSDT_231229)
                if "symbol" in row:
                    key = row["symbol"]
                    expiry_data[key] = {
                        "deliveryDate": row.get("deliveryDate"),
                        "onboardDate": row.get("onboardDate"),
                    }

        logger.info(f"Loaded {len(expiry_data)} expiry records from {file_path}")
        return expiry_data

    except Exception as e:
        logger.error(f"Error loading expiry data: {e}")
        return {}


def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    return directory


def read_futures_csv(file_path):
    """Read futures CSV file and convert timestamp to datetime."""
    try:
        df = pd.read_csv(file_path)
        if "Timestamp" in df.columns:
            # Ensure timestamp is in datetime format
            if df["Timestamp"].dtype == "object":
                # Try to parse if it's a string timestamp
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            else:
                # If it's a numeric timestamp (milliseconds)
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
        else:
            logger.warning(f"CSV file {file_path} does not have a 'Timestamp' column")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        return None


def process_interval(index, interval, futures_roll):
    """
    Process data for a specific index and interval.

    Args:
        index (str): Index symbol like "BTCUSDT"
        interval (str): Time interval like "1h", "8h", or "1d"
        futures_roll (str): Roll period like "7d"

    Returns:
        pandas.DataFrame or None: The futures term structure if successful
    """
    # Construct the directory path with the new futures_contracts subdirectory
    data_dir = os.path.join("data", "raw", index, interval, "futures_contracts")

    # Load expiry data
    expiry_file = os.path.join("data", "contracts", "fut_expirys.csv")
    expiry_data = load_expiry_data(expiry_file)

    # Find all files with *.csv pattern in the futures_contracts directory
    pattern = os.path.join(data_dir, "*.csv")
    logger.info(f"Looking for files matching pattern: {pattern}")

    # Use glob to find matching files
    csv_files = glob.glob(pattern)

    # Log the files found
    logger.info(f"Found {len(csv_files)} files matching the pattern")

    # Create a list to store contract information
    contracts = []

    # Process each file to collect metadata
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        logger.info(f"Found file: {filename}")

        # Extract the date part (e.g., "231229" from "231229.csv")
        date_part = filename.split(".")[0]

        # Create the key to look up in expiry data
        lookup_key = f"{index}_{date_part}"

        # Look up the key in expiry data
        if lookup_key in expiry_data:
            delivery_date = int(expiry_data[lookup_key]["deliveryDate"])
            onboard_date = int(expiry_data[lookup_key]["onboardDate"])

            # Calculate roll date
            try:
                roll_date_ms = calculate_roll_date(delivery_date, futures_roll)

                # Format dates for display
                delivery_date_str = datetime.fromtimestamp(
                    delivery_date / 1000
                ).strftime("%Y-%m-%d %H:%M:%S")
                roll_date_str = datetime.fromtimestamp(roll_date_ms / 1000).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                onboard_date_str = datetime.fromtimestamp(onboard_date / 1000).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                logger.info(f"  Matched with expiry data - Key: {lookup_key}")
                logger.info(f"  Delivery Date: {delivery_date} ({delivery_date_str})")
                logger.info(f"  Onboard Date: {onboard_date} ({onboard_date_str})")
                logger.info(
                    f"  Roll Date ({futures_roll} before delivery): {roll_date_ms} ({roll_date_str})"
                )

                # Add contract info to the list
                contracts.append(
                    {
                        "symbol": lookup_key,
                        "file_path": file_path,
                        "onboard_date": onboard_date,
                        "delivery_date": delivery_date,
                        "roll_date": roll_date_ms,
                    }
                )
            except ValueError as e:
                logger.error(f"  Error calculating roll date: {e}")
        else:
            logger.warning(f"  No matching expiry data found for key: {lookup_key}")

    if not contracts:
        logger.error(f"No valid contracts found for {index}_{interval}. Skipping.")
        return None

    # Sort contracts by delivery date
    sorted_contracts = sorted(contracts, key=lambda x: x["delivery_date"])
    logger.info(f"Sorted {len(sorted_contracts)} contracts by delivery date")

    # Load all contract data with additional metadata
    contract_data_frames = []

    for contract in sorted_contracts:
        symbol = contract["symbol"]
        df = read_futures_csv(contract["file_path"])

        if df is not None and not df.empty:
            # Add contract metadata to each row
            df["Contract"] = symbol

            # Calculate days_till_expiry for each row
            df["timestamp_ms"] = df["Timestamp"].astype(int) // 10**6
            ms_per_day = 24 * 60 * 60 * 1000
            df["days_till_expiry"] = (
                contract["delivery_date"] - df["timestamp_ms"]
            ) / ms_per_day

            # Remove the temporary timestamp_ms column
            df = df.drop(columns=["timestamp_ms"])

            # Add to our list of dataframes
            contract_data_frames.append(df)

    if not contract_data_frames:
        logger.error(f"No data available for {index}_{interval}. Skipping.")
        return None

    # Combine all contract data into a single DataFrame
    all_contracts_df = pd.concat(contract_data_frames, ignore_index=True)

    # Group by timestamp to find all contracts trading at each point in time
    term_structure_rows = []

    # Get unique timestamps
    unique_timestamps = all_contracts_df["Timestamp"].unique()
    logger.info(f"Processing {len(unique_timestamps)} unique timestamps")

    # Parse the roll threshold from FUTURES_ROLL
    roll_match = re.match(r"(\d+)([mhdw])", futures_roll)
    if not roll_match:
        raise ValueError(f"Unsupported futures roll format: {futures_roll}")

    roll_value = int(roll_match.group(1))
    roll_unit = roll_match.group(2)

    # Convert to days for comparison with days_till_expiry
    unit_to_days = {
        "m": 1 / (24 * 60),  # minutes to days
        "h": 1 / 24,  # hours to days
        "d": 1,  # days to days
        "w": 7,  # weeks to days
    }
    roll_days_threshold = roll_value * unit_to_days[roll_unit]
    logger.info(f"Using roll threshold of {roll_days_threshold} days")

    for timestamp in unique_timestamps:
        # Get data for this timestamp
        timestamp_data = all_contracts_df[all_contracts_df["Timestamp"] == timestamp]

        # Sort by days_till_expiry (or equivalently by delivery_date)
        timestamp_data = timestamp_data.sort_values("days_till_expiry")

        # Check if we have contracts to process
        if len(timestamp_data) > 0:
            # Apply rolling logic: if the nearest contract is past the roll threshold,
            # we should use the next contract as the prompt

            # Find the index of the first contract with days_till_expiry >= roll_days_threshold
            valid_contracts = timestamp_data[
                timestamp_data["days_till_expiry"] >= roll_days_threshold
            ]

            if len(valid_contracts) >= 2:
                # We have at least two valid contracts
                prompt_contract = valid_contracts.iloc[0]
                next_contract = valid_contracts.iloc[1]

                row = {
                    "Timestamp": timestamp,
                    "prompt_Open": prompt_contract["Open"],
                    "prompt_High": prompt_contract["High"],
                    "prompt_Low": prompt_contract["Low"],
                    "prompt_Close": prompt_contract["Close"],
                    "prompt_Contract": prompt_contract["Contract"],
                    "prompt_days_till_expiry": prompt_contract["days_till_expiry"],
                    "next_Open": next_contract["Open"],
                    "next_High": next_contract["High"],
                    "next_Low": next_contract["Low"],
                    "next_Close": next_contract["Close"],
                    "next_Contract": next_contract["Contract"],
                    "next_days_till_expiry": next_contract["days_till_expiry"],
                }
                term_structure_rows.append(row)
            elif len(valid_contracts) == 1:
                # We have one valid contract but maybe a second expired contract
                prompt_contract = valid_contracts.iloc[0]

                # Check if there's another contract that we can use as next (even if expired)
                if len(timestamp_data) >= 2:
                    # Use the expired contract as next if available
                    # Should be the second contract after the valid one we found
                    next_idx = (
                        timestamp_data.index.get_indexer([prompt_contract.name])[0] + 1
                    )
                    if next_idx < len(timestamp_data):
                        next_contract = timestamp_data.iloc[next_idx]

                        row = {
                            "Timestamp": timestamp,
                            "prompt_Open": prompt_contract["Open"],
                            "prompt_High": prompt_contract["High"],
                            "prompt_Low": prompt_contract["Low"],
                            "prompt_Close": prompt_contract["Close"],
                            "prompt_Contract": prompt_contract["Contract"],
                            "prompt_days_till_expiry": prompt_contract[
                                "days_till_expiry"
                            ],
                            "next_Open": next_contract["Open"],
                            "next_High": next_contract["High"],
                            "next_Low": next_contract["Low"],
                            "next_Close": next_contract["Close"],
                            "next_Contract": next_contract["Contract"],
                            "next_days_till_expiry": next_contract["days_till_expiry"],
                        }
                        term_structure_rows.append(row)
                    else:
                        # No next contract available
                        row = {
                            "Timestamp": timestamp,
                            "prompt_Open": prompt_contract["Open"],
                            "prompt_High": prompt_contract["High"],
                            "prompt_Low": prompt_contract["Low"],
                            "prompt_Close": prompt_contract["Close"],
                            "prompt_Contract": prompt_contract["Contract"],
                            "prompt_days_till_expiry": prompt_contract[
                                "days_till_expiry"
                            ],
                            "next_Open": None,
                            "next_High": None,
                            "next_Low": None,
                            "next_Close": None,
                            "next_Contract": None,
                            "next_days_till_expiry": None,
                        }
                        term_structure_rows.append(row)
                else:
                    # Only one valid contract and no other contracts
                    row = {
                        "Timestamp": timestamp,
                        "prompt_Open": prompt_contract["Open"],
                        "prompt_High": prompt_contract["High"],
                        "prompt_Low": prompt_contract["Low"],
                        "prompt_Close": prompt_contract["Close"],
                        "prompt_Contract": prompt_contract["Contract"],
                        "prompt_days_till_expiry": prompt_contract["days_till_expiry"],
                        "next_Open": None,
                        "next_High": None,
                        "next_Low": None,
                        "next_Close": None,
                        "next_Contract": None,
                        "next_days_till_expiry": None,
                    }
                    term_structure_rows.append(row)
            elif len(timestamp_data) >= 2:
                # No valid contracts beyond the roll threshold, but we have at least two contracts
                # We'll use the two contracts with the highest days_till_expiry
                logger.warning(
                    f"No contracts beyond roll threshold at {timestamp}, using available contracts"
                )
                prompt_contract = timestamp_data.iloc[0]
                next_contract = timestamp_data.iloc[1]

                row = {
                    "Timestamp": timestamp,
                    "prompt_Open": prompt_contract["Open"],
                    "prompt_High": prompt_contract["High"],
                    "prompt_Low": prompt_contract["Low"],
                    "prompt_Close": prompt_contract["Close"],
                    "prompt_Contract": prompt_contract["Contract"],
                    "prompt_days_till_expiry": prompt_contract["days_till_expiry"],
                    "next_Open": next_contract["Open"],
                    "next_High": next_contract["High"],
                    "next_Low": next_contract["Low"],
                    "next_Close": next_contract["Close"],
                    "next_Contract": next_contract["Contract"],
                    "next_days_till_expiry": next_contract["days_till_expiry"],
                }
                term_structure_rows.append(row)
            elif len(timestamp_data) == 1:
                # Only one contract available (and it's below the threshold)
                logger.warning(
                    f"Only one contract available at {timestamp} and it's below roll threshold"
                )
                prompt_contract = timestamp_data.iloc[0]

                row = {
                    "Timestamp": timestamp,
                    "prompt_Open": prompt_contract["Open"],
                    "prompt_High": prompt_contract["High"],
                    "prompt_Low": prompt_contract["Low"],
                    "prompt_Close": prompt_contract["Close"],
                    "prompt_Contract": prompt_contract["Contract"],
                    "prompt_days_till_expiry": prompt_contract["days_till_expiry"],
                    "next_Open": None,
                    "next_High": None,
                    "next_Low": None,
                    "next_Close": None,
                    "next_Contract": None,
                    "next_days_till_expiry": None,
                }
                term_structure_rows.append(row)

    # Create DataFrame from term structure rows
    term_structure_df = pd.DataFrame(term_structure_rows)

    # Sort by timestamp
    term_structure_df = term_structure_df.sort_values("Timestamp")

    # Save the term structure data
    if not term_structure_df.empty:
        output_dir = os.path.join("data", "processed", index, interval)
        ensure_dir_exists(output_dir)

        output_filename = f"futures_curve_{futures_roll}_roll.csv"
        output_path = os.path.join(output_dir, output_filename)
        term_structure_df.to_csv(output_path, index=False)

        logger.info(
            f"Successfully created futures term structure with {len(term_structure_df)} rows"
        )
        logger.info(f"Saved to {output_path}")

        return term_structure_df
    else:
        logger.error(f"No term structure data available for {index}_{interval}")
        return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build futures term structure.")
    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_INDEX,
        help=f"Symbol to process (e.g., {DEFAULT_INDEX})",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default=",".join(DEFAULT_INTERVALS),
        help=f"Time intervals, comma-separated (e.g., {','.join(DEFAULT_INTERVALS)})",
    )
    parser.add_argument(
        "--futures-roll",
        type=str,
        default=DEFAULT_FUTURES_ROLL,
        help=f"Future roll period (e.g., {DEFAULT_FUTURES_ROLL})",
    )
    return parser.parse_args()


def main():
    """Main function to build futures term structure for multiple intervals."""
    # Parse command line arguments
    args = parse_arguments()

    symbol = args.symbol
    intervals = args.intervals.split(",") if "," in args.intervals else [args.intervals]
    futures_roll = args.futures_roll

    logger.info(
        f"Building futures term structure for {symbol} with intervals {intervals}"
    )

    results = {}

    for interval in intervals:
        logger.info(f"Processing index {symbol} with interval {interval}")
        result = process_interval(symbol, interval, futures_roll)
        results[interval] = result
        logger.info(f"Completed processing for {symbol}_{interval}")
        logger.info("-" * 80)  # Separator for better readability in logs

    # Return the results dictionary
    return results


if __name__ == "__main__":
    main()
