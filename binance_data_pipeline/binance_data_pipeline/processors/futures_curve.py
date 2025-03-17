# binance_data_pipeline/processors/futures_curve.py
"""
Processor for building futures term structure from CSV files.

This module takes raw futures contract data and builds a consolidated
term structure with prompt and next contract data.
"""

import os
import glob
import re
import pandas as pd
from datetime import datetime
from pathlib import Path

from ..core.logger import get_logger
from ..core.config import config
from ..exceptions import ProcessorError
from ..utils.file_utils import save_to_csv

logger = get_logger(__name__)


class FuturesCurveProcessor:
    """
    Processor for building futures term structure from multiple contract CSVs.

    This class generates consolidated futures curve data by analyzing multiple
    contract CSVs and applying roll logic based on delivery dates.
    """

    def __init__(self, symbol=None, intervals=None, futures_roll="7d"):
        """
        Initialize the futures curve processor.

        Args:
            symbol (str, optional): Trading pair symbol (e.g., "BTCUSDT")
            intervals (list or str, optional): Time intervals to process
            futures_roll (str): Roll period before expiry (e.g., "7d" for 7 days)
        """
        self.symbol = symbol or config.default_symbol

        # Handle intervals input - convert string to list if needed
        if intervals is None:
            self.intervals = config.default_intervals.get("futures", ["1d"])
        elif isinstance(intervals, str):
            self.intervals = [intervals]
        else:
            self.intervals = intervals

        self.futures_roll = futures_roll

    def process_all(self):
        """
        Process all intervals for the configured symbol.

        Returns:
            dict: Dictionary mapping intervals to their processed data
        """
        logger.info(f"Building futures term structure for {self.symbol}")

        results = {}
        for interval in self.intervals:
            logger.info(f"Processing {self.symbol} with interval {interval}")
            result = self.process_interval(interval)
            results[interval] = result
            logger.info(f"Completed processing for {self.symbol}_{interval}")
            logger.info("-" * 80)  # Separator for better readability in logs

        return results

    def process_interval(self, interval):
        """
        Process a specific interval for the configured symbol.

        Args:
            interval (str): Time interval (e.g., "1d", "8h", "1h")

        Returns:
            pandas.DataFrame or None: The processed futures curve data or None if no data
        """
        try:
            # Construct the directory path for futures contracts
            data_dir = (
                Path(config.raw_dir) / self.symbol / interval / "futures_contracts"
            )

            # Load expiry data
            expiry_file = Path(config.contracts_dir) / "fut_expirys.csv"
            expiry_data = self._load_expiry_data(expiry_file)

            # Find all CSV files in the futures_contracts directory
            pattern = os.path.join(data_dir, "*.csv")
            logger.info(f"Looking for files matching pattern: {pattern}")

            # Use glob to find matching files
            csv_files = glob.glob(pattern)
            logger.info(f"Found {len(csv_files)} files matching the pattern")

            if not csv_files:
                logger.warning(
                    f"No contract data files found for {self.symbol}_{interval}"
                )
                return None

            # Process contracts and build term structure
            contracts = self._collect_contract_info(csv_files, expiry_data)

            if not contracts:
                logger.error(
                    f"No valid contracts found for {self.symbol}_{interval}. Skipping."
                )
                return None

            # Sort contracts by delivery date
            sorted_contracts = sorted(contracts, key=lambda x: x["delivery_date"])
            logger.info(f"Sorted {len(sorted_contracts)} contracts by delivery date")

            # Load contract data and build term structure
            term_structure_df = self._build_term_structure(
                sorted_contracts, self.futures_roll
            )

            if term_structure_df is not None and not term_structure_df.empty:
                # Save the term structure data
                output_dir = Path(config.processed_dir) / self.symbol / interval
                output_dir.mkdir(parents=True, exist_ok=True)

                output_filename = f"futures_curve_{self.futures_roll}_roll.csv"
                output_path = output_dir / output_filename

                term_structure_df.to_csv(output_path, index=False)
                logger.info(
                    f"Successfully created futures term structure with {len(term_structure_df)} rows"
                )
                logger.info(f"Saved to {output_path}")

                return term_structure_df
            else:
                logger.error(
                    f"No term structure data available for {self.symbol}_{interval}"
                )
                return None

        except Exception as e:
            logger.error(
                f"Error processing futures curve for {self.symbol}_{interval}: {e}"
            )
            raise ProcessorError(f"Failed to process futures curve: {e}")

    def _load_expiry_data(self, file_path):
        """
        Load futures expiry data from CSV file.

        Args:
            file_path (Path): Path to the expiry data CSV

        Returns:
            dict: Dictionary mapping contract symbols to their expiry details
        """
        expiry_data = {}

        try:
            if not file_path.exists():
                logger.warning(f"Expiry data file not found: {file_path}")
                return expiry_data

            # Read the CSV file using pandas
            df = pd.read_csv(file_path)

            # Create a dictionary with contract symbols as keys
            for _, row in df.iterrows():
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

    def _collect_contract_info(self, csv_files, expiry_data):
        """
        Collect contract information from CSV files and match with expiry data.

        Args:
            csv_files (list): List of CSV file paths
            expiry_data (dict): Contract expiry information

        Returns:
            list: List of contract dictionaries with metadata
        """
        contracts = []

        for file_path in csv_files:
            filename = os.path.basename(file_path)
            logger.info(f"Found file: {filename}")

            # Extract the date part (e.g., "231229" from "231229.csv")
            date_part = filename.split(".")[0]

            # Create the key to look up in expiry data
            lookup_key = f"{self.symbol}_{date_part}"

            # Look up the key in expiry data
            if lookup_key in expiry_data:
                delivery_date = int(expiry_data[lookup_key]["deliveryDate"])
                onboard_date = int(expiry_data[lookup_key]["onboardDate"])

                # Calculate roll date
                try:
                    roll_date_ms = self._calculate_roll_date(
                        delivery_date, self.futures_roll
                    )

                    # Format dates for display
                    delivery_date_str = datetime.fromtimestamp(
                        delivery_date / 1000
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    roll_date_str = datetime.fromtimestamp(
                        roll_date_ms / 1000
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    onboard_date_str = datetime.fromtimestamp(
                        onboard_date / 1000
                    ).strftime("%Y-%m-%d %H:%M:%S")

                    logger.info(f"  Matched with expiry data - Key: {lookup_key}")
                    logger.info(
                        f"  Delivery Date: {delivery_date} ({delivery_date_str})"
                    )
                    logger.info(f"  Onboard Date: {onboard_date} ({onboard_date_str})")
                    logger.info(
                        f"  Roll Date ({self.futures_roll} before delivery): {roll_date_ms} ({roll_date_str})"
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

        return contracts

    def _calculate_roll_date(self, delivery_date, futures_roll):
        """
        Calculate the date to roll to the next futures contract.

        Args:
            delivery_date (int): Delivery date in Unix milliseconds
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

    def _read_futures_csv(self, file_path):
        """
        Read futures CSV file and convert timestamp to datetime.

        Args:
            file_path (str): Path to the CSV file

        Returns:
            pandas.DataFrame or None: DataFrame with futures data or None if error
        """
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
                logger.warning(
                    f"CSV file {file_path} does not have a 'Timestamp' column"
                )
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return None

    def _build_term_structure(self, contracts, futures_roll):
        """
        Build the futures term structure from contract data.

        Args:
            contracts (list): List of contract dictionaries with metadata
            futures_roll (str): Roll period before expiry

        Returns:
            pandas.DataFrame: The futures term structure
        """
        # Load all contract data with additional metadata
        contract_data_frames = []

        for contract in contracts:
            symbol = contract["symbol"]
            df = self._read_futures_csv(contract["file_path"])

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
            logger.error(f"No data available for {self.symbol}. Skipping.")
            return None

        # Combine all contract data into a single DataFrame
        all_contracts_df = pd.concat(contract_data_frames, ignore_index=True)

        # Group by timestamp to find all contracts trading at each point in time
        term_structure_rows = []

        # Get unique timestamps
        unique_timestamps = all_contracts_df["Timestamp"].unique()
        logger.info(f"Processing {len(unique_timestamps)} unique timestamps")

        # Parse the roll threshold from futures_roll
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

        # For each timestamp, determine the prompt and next contracts
        for timestamp in unique_timestamps:
            # Get data for this timestamp
            timestamp_data = all_contracts_df[
                all_contracts_df["Timestamp"] == timestamp
            ]

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
                            timestamp_data.index.get_indexer([prompt_contract.name])[0]
                            + 1
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
                                "next_days_till_expiry": next_contract[
                                    "days_till_expiry"
                                ],
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
        if term_structure_rows:
            term_structure_df = pd.DataFrame(term_structure_rows)

            # Sort by timestamp
            term_structure_df = term_structure_df.sort_values("Timestamp")

            return term_structure_df
        else:
            logger.warning(f"No term structure data generated for {self.symbol}")
            return None
