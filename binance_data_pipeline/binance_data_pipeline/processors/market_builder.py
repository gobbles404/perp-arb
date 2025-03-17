# binance_data_pipeline/processors/market_builder.py
"""
Processor for building consolidated market data.

This module combines various market data sources (spot, perps, funding rates, etc.)
into a single CSV file with consistent column naming.
"""

import pandas as pd
from pathlib import Path

from ..core.logger import get_logger
from ..core.config import config
from ..exceptions import ProcessorError
from ..utils.file_utils import save_to_csv

logger = get_logger(__name__)


class MarketBuilder:
    """
    Processor for consolidating various market data sources.

    This class brings together spot prices, perpetual futures, funding rates,
    and futures term structure data into a unified market data file.
    """

    def __init__(self, symbol=None, intervals=None):
        """
        Initialize the market builder.

        Args:
            symbol (str, optional): Trading pair symbol (e.g., "BTCUSDT")
            intervals (list or str, optional): Time intervals to process
        """
        self.symbol = symbol or config.default_symbol

        # Handle intervals input - convert string to list if needed
        if intervals is None:
            self.intervals = config.default_intervals.get("spot", ["1d", "8h", "1h"])
        elif isinstance(intervals, str):
            self.intervals = [intervals]
        else:
            self.intervals = intervals

    def build_all(self):
        """
        Build market data for all configured intervals.

        Returns:
            dict: Dictionary mapping intervals to their processed data
        """
        logger.info(f"Building market data for {self.symbol} with all intervals")

        results = {}
        for interval in self.intervals:
            logger.info(f"Processing interval: {interval}")
            df = self.build_for_interval(interval)
            results[interval] = df
            logger.info(f"Completed processing for {self.symbol}_{interval}")
            logger.info("-" * 80)  # Separator for better readability in logs

        return results

    def build_for_interval(self, interval):
        """
        Build market data for a specific interval.

        Args:
            interval (str): Time interval (e.g., "1d", "8h", "1h")

        Returns:
            pandas.DataFrame or None: The consolidated market data or None if error
        """
        try:
            # Define directory paths
            raw_dir = Path(config.raw_dir) / self.symbol / interval
            processed_dir = Path(config.processed_dir) / self.symbol / interval
            markets_dir = Path(config.markets_dir)
            markets_dir.mkdir(parents=True, exist_ok=True)

            # Load all data sources
            spot_df = self._load_csv(raw_dir / "spot.csv", "spot")
            perp_df = self._load_csv(raw_dir / "perps.csv", "perpetual futures")

            # For funding rates, we need to look in the 8h directory if we're processing other intervals
            if interval in ["1d", "1h", "1m"]:
                funding_path = (
                    Path(config.raw_dir) / self.symbol / "8h" / "funding_rates.csv"
                )
                funding_df = self._load_csv(funding_path, f"funding rates (8h)")
                logger.info(
                    f"Using 8h funding rates for {interval} interval from {funding_path}"
                )
            else:
                funding_df = self._load_csv(
                    raw_dir / "funding_rates.csv", "funding rates"
                )

            # Load futures curve data
            futures_curve_df = self._load_csv(
                processed_dir / "futures_curve_7d_roll.csv", "7-day futures curve"
            )

            # Check if we have the minimum required data
            if spot_df is None or perp_df is None:
                logger.error(
                    f"Missing critical data (spot or perp) for {self.symbol} {interval}"
                )
                return None

            # Start building the merged dataframe with spot data as base
            logger.info(
                f"Beginning merge process with {len(spot_df)} rows of spot data"
            )

            # Prepare spot data with renamed columns
            merged_df = spot_df.copy()
            for col in ["Open", "High", "Low", "Close"]:
                if col in merged_df.columns:
                    merged_df.rename(columns={col: f"spot_{col.lower()}"}, inplace=True)

            # Prepare and merge perpetual futures data
            if perp_df is not None:
                perp_temp = perp_df.copy()
                for col in ["Open", "High", "Low", "Close"]:
                    if col in perp_temp.columns:
                        perp_temp.rename(
                            columns={col: f"perp_{col.lower()}"}, inplace=True
                        )

                merged_df = pd.merge(merged_df, perp_temp, on="Timestamp", how="outer")
                logger.info(f"Merged perpetual futures data ({len(perp_df)} rows)")

            # Prepare and merge funding rate data
            if funding_df is not None:
                funding_temp = self._prepare_funding_rates(funding_df, interval)

                if funding_temp is not None:
                    merged_df = pd.merge(
                        merged_df, funding_temp, on="Timestamp", how="outer"
                    )
                    logger.info(f"Merged funding rate data")

            # Prepare and merge futures curve data
            if futures_curve_df is not None:
                futures_curve_temp = futures_curve_df.copy()

                # Create properly prefixed column names for the prompt contract
                prompt_columns = {
                    "prompt_Open": "prompt_open",
                    "prompt_High": "prompt_high",
                    "prompt_Low": "prompt_low",
                    "prompt_Close": "prompt_close",
                    "prompt_Contract": "prompt_contract",
                    "prompt_days_till_expiry": "prompt_days_till_expiry",
                }

                # Create properly prefixed column names for the next contract
                next_columns = {
                    "next_Open": "next_open",
                    "next_High": "next_high",
                    "next_Low": "next_low",
                    "next_Close": "next_close",
                    "next_Contract": "next_contract",
                    "next_days_till_expiry": "next_days_till_expiry",
                }

                # Rename all columns to follow the naming convention
                futures_curve_temp.rename(
                    columns={**prompt_columns, **next_columns}, inplace=True
                )

                # Merge with the main dataframe
                merged_df = pd.merge(
                    merged_df, futures_curve_temp, on="Timestamp", how="outer"
                )
                logger.info(
                    f"Merged 7-day futures curve data ({len(futures_curve_df)} rows)"
                )

            # Sort by timestamp
            merged_df.sort_values("Timestamp", inplace=True)

            # Save merged data to CSV
            output_file = markets_dir / f"{self.symbol}_{interval}.csv"
            merged_df.to_csv(output_file, index=False)
            logger.info(f"Successfully saved consolidated market data to {output_file}")
            logger.info(
                f"Final dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns"
            )

            return merged_df

        except Exception as e:
            logger.error(
                f"Error building market data for {self.symbol} {interval}: {e}"
            )
            raise ProcessorError(f"Failed to build market data: {e}")

    def _load_csv(self, filepath, source_name=None):
        """
        Load a CSV file and ensure timestamp is correctly formatted.

        Args:
            filepath (Path): Path to the CSV file
            source_name (str, optional): Name identifier for logging

        Returns:
            pd.DataFrame or None: Loaded DataFrame or None if loading failed
        """
        try:
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                return None

            name = source_name or filepath.stem
            logger.info(f"Loading {name} data from {filepath}")

            df = pd.read_csv(filepath)

            # Ensure Timestamp is in datetime format
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])

            return df
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None

    def _prepare_funding_rates(self, funding_df, interval):
        """
        Prepare funding rate data for the specified interval.

        Args:
            funding_df (pd.DataFrame): Raw funding rate data
            interval (str): Time interval for processing

        Returns:
            pd.DataFrame or None: Processed funding rate data or None if error
        """
        try:
            # For funding rates, we only need specific columns
            if "symbol" in funding_df.columns:
                # Verify symbol matches our target or filter
                if not all(funding_df["symbol"] == self.symbol):
                    logger.warning(
                        f"Found mixed symbols in funding data. Filtering for {self.symbol}"
                    )
                    funding_df = funding_df[funding_df["symbol"] == self.symbol]

            # Special handling for funding rates in 1d interval
            if interval == "1d":
                logger.info("Detected 1d interval - averaging funding rates per day")

                # Extract date only for grouping (remove time component)
                funding_df["date"] = funding_df["Timestamp"].dt.date

                # Group by date and calculate average Rate and markPrice
                daily_funding = (
                    funding_df.groupby("date")
                    .agg({"Rate": "mean", "markPrice": "mean"})
                    .reset_index()
                )

                # Convert date back to datetime to match format
                daily_funding["Timestamp"] = pd.to_datetime(daily_funding["date"])
                daily_funding = daily_funding.drop("date", axis=1)

                # Add annualized funding rate (365 days per year)
                daily_funding["Rate_Annualized"] = daily_funding["Rate"] * 365 * 3

                # Rename columns to match our convention
                funding_temp = daily_funding.rename(
                    columns={
                        "Rate": "funding_rate",
                        "markPrice": "funding_markPrice",
                        "Rate_Annualized": "funding_annualized",
                    }
                )

                logger.info(
                    f"Aggregated funding rates from {len(funding_df)} entries to {len(daily_funding)} daily averages"
                )

                # Ensure the daily funding data timestamps match the format of 1d data
                # This is important as 1d data typically has 00:00:00 timestamps
                funding_temp["Timestamp"] = funding_temp["Timestamp"].dt.normalize()

                return funding_temp

            # Special handling for funding rates in 1h interval
            elif interval == "1h":
                logger.info(
                    "Detected 1h interval - distributing 8h funding rates to hourly data"
                )

                # We need spot data for timestamp reference
                spot_df = self._load_csv(
                    Path(config.raw_dir) / self.symbol / interval / "spot.csv", "spot"
                )

                if spot_df is None or len(spot_df) == 0:
                    logger.warning(
                        "Cannot process 1h funding rates without spot data for timestamp reference"
                    )
                    return None

                # Create a complete hourly series based on spot data time range
                start_time = spot_df["Timestamp"].min().floor("h")
                end_time = spot_df["Timestamp"].max().ceil("h")
                hourly_range = pd.date_range(start=start_time, end=end_time, freq="h")

                # Create a DataFrame with the complete hourly range
                hourly_df = pd.DataFrame({"Timestamp": hourly_range})

                # Merge with funding data to get funding rates at their specific times
                funding_with_times = pd.merge(
                    hourly_df,
                    funding_df[["Timestamp", "Rate", "markPrice"]],
                    on="Timestamp",
                    how="left",
                )

                # Forward fill the funding rates to apply the most recent rate to each hour
                funding_with_times["Rate"] = funding_with_times["Rate"].ffill()
                funding_with_times["markPrice"] = funding_with_times[
                    "markPrice"
                ].ffill()

                # Divide the 8h funding rate by 8 to get the hourly equivalent
                funding_with_times["Rate"] = funding_with_times["Rate"] / 8

                # Add annualized funding rate (24 hours * 365 days per year)
                funding_with_times["Rate_Annualized"] = (
                    funding_with_times["Rate"] * 24 * 365
                )

                # Rename columns to match our convention
                funding_temp = funding_with_times.rename(
                    columns={
                        "Rate": "funding_rate",
                        "markPrice": "funding_markPrice",
                        "Rate_Annualized": "funding_annualized",
                    }
                )

                logger.info(
                    f"Distributed 8h funding rates to {len(funding_temp)} hourly intervals"
                )

                return funding_temp

            # Special handling for funding rates in 1m interval
            elif interval == "1m":
                logger.info(
                    "Detected 1m interval - distributing 8h funding rates to minute data"
                )

                # We need spot data for timestamp reference
                spot_df = self._load_csv(
                    Path(config.raw_dir) / self.symbol / interval / "spot.csv", "spot"
                )

                if spot_df is None or len(spot_df) == 0:
                    logger.warning(
                        "Cannot process 1m funding rates without spot data for timestamp reference"
                    )
                    return None

                # Create a complete minutely series based on spot data time range
                start_time = spot_df["Timestamp"].min().floor("min")
                end_time = spot_df["Timestamp"].max().ceil("min")
                minute_range = pd.date_range(start=start_time, end=end_time, freq="min")

                # Create a DataFrame with the complete minutely range
                minute_df = pd.DataFrame({"Timestamp": minute_range})

                # Merge with funding data to get funding rates at their specific times
                funding_with_times = pd.merge(
                    minute_df,
                    funding_df[["Timestamp", "Rate", "markPrice"]],
                    on="Timestamp",
                    how="left",
                )

                # Forward fill the funding rates to apply the most recent rate to each minute
                funding_with_times["Rate"] = funding_with_times["Rate"].ffill()
                funding_with_times["markPrice"] = funding_with_times[
                    "markPrice"
                ].ffill()

                # Divide the 8h funding rate by (8 * 60) to get the per-minute equivalent
                funding_with_times["Rate"] = funding_with_times["Rate"] / (8 * 60)

                # Add annualized funding rate (24 hours * 60 minutes * 365 days per year)
                funding_with_times["Rate_Annualized"] = (
                    funding_with_times["Rate"] * 24 * 60 * 365
                )

                # Rename columns to match our convention
                funding_temp = funding_with_times.rename(
                    columns={
                        "Rate": "funding_rate",
                        "markPrice": "funding_markPrice",
                        "Rate_Annualized": "funding_annualized",
                    }
                )

                logger.info(
                    f"Distributed 8h funding rates to {len(funding_temp)} minute intervals"
                )

                return funding_temp

            else:
                # For 8h interval (the native interval of funding data)
                funding_temp = funding_df[["Timestamp", "Rate", "markPrice"]].copy()

                # Add annualized funding rate (3 funding events per day * 365 days per year)
                funding_temp["Rate_Annualized"] = funding_temp["Rate"] * 3 * 365

                funding_temp.rename(
                    columns={
                        "Rate": "funding_rate",
                        "markPrice": "funding_markPrice",
                        "Rate_Annualized": "funding_annualized",
                    },
                    inplace=True,
                )

                return funding_temp

        except Exception as e:
            logger.error(f"Error preparing funding rates for {interval}: {e}")
            return None
