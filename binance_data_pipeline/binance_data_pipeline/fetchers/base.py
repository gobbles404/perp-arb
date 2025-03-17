# binance_data_pipeline/fetchers/base.py
import pandas as pd
from datetime import datetime
from abc import ABC, abstractmethod

from ..core.logger import get_logger
from ..core.client import client
from ..core.config import config
from ..utils.date_utils import date_to_timestamp, get_last_expected_timestamp
from ..utils.file_utils import save_to_csv
from ..exceptions import FetcherError


class BinanceFetcher(ABC):
    """
    Base class for all Binance data fetchers.

    This abstract class implements common functionality shared across all fetchers
    and defines the interface that specific fetchers must implement.
    """

    def __init__(self, symbol=None, intervals=None, start_date=None, end_date=None):
        """
        Initialize the fetcher with common parameters.

        Args:
            symbol (str): Trading pair symbol (e.g., "BTCUSDT")
            intervals (list or str): Time interval(s) (e.g., ["1d", "8h", "1h"] or "1d")
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        self.symbol = symbol or config.default_symbol

        # Handle intervals input - convert string to list if needed
        if intervals is None:
            self.intervals = config.default_intervals.get(
                self._get_fetcher_type(), ["1d"]
            )
        elif isinstance(intervals, str):
            self.intervals = [intervals]
        else:
            self.intervals = intervals

        self.start_date = start_date or config.default_start_date
        self.end_date = end_date or config.default_end_date
        self.client = client
        self.logger = get_logger(self.__class__.__name__)

    def _get_fetcher_type(self):
        """
        Get the fetcher type based on the class name.
        Used to determine default intervals if not specified.
        """
        class_name = self.__class__.__name__.lower()
        if "spot" in class_name:
            return "spot"
        elif "futures" in class_name:
            return "futures"
        elif "funding" in class_name:
            return "funding"
        elif "premium" in class_name:
            return "premium"
        else:
            return "spot"  # Default to spot intervals

    def fetch_all(self):
        """
        Fetch data for all configured intervals.

        Returns:
            dict: Dictionary mapping intervals to their fetched data.
        """
        self.logger.info(f"Starting data fetch for {self.symbol}")

        results = {}
        for interval in self.intervals:
            self.logger.info(f"Processing {self.symbol} with interval {interval}")
            data = self.fetch_for_interval(interval)
            results[interval] = data

        self.logger.info(f"Completed data fetch for {self.symbol}")
        return results

    @abstractmethod
    def fetch_for_interval(self, interval):
        """
        Fetch data for a specific interval. Must be implemented by subclasses.

        Args:
            interval (str): Time interval (e.g., "1d", "8h", "1h")

        Returns:
            pandas.DataFrame or None: The fetched data or None if no data available
        """
        raise NotImplementedError("Subclasses must implement fetch_for_interval method")

    def _get_timestamp_range(self):
        """
        Convert date strings to timestamps.

        Returns:
            tuple: (start_timestamp, end_timestamp) in milliseconds
        """
        return date_to_timestamp(self.start_date, self.end_date)

    def _fetch_klines(self, interval, start_ts, end_ts, kline_type="spot"):
        """
        Fetch kline data with automatic pagination.

        Args:
            interval (str): Time interval
            start_ts (int): Start timestamp in milliseconds
            end_ts (int): End timestamp in milliseconds
            kline_type (str): Type of klines to fetch ("spot", "futures", "premium")

        Returns:
            pandas.DataFrame or None: DataFrame with kline data or None if no data
        """
        all_data = []
        current_ts = start_ts
        limit = 1000

        # Get the most recent possible timestamp for this interval
        last_expected_ts = get_last_expected_timestamp(interval)

        while current_ts < end_ts:
            self.logger.info(
                f"Fetching {self.symbol} ({interval}) from {pd.to_datetime(current_ts, unit='ms')}"
            )

            try:
                # Select the appropriate kline fetch method based on type
                if kline_type == "futures":
                    klines = self.client.futures_klines(
                        symbol=self.symbol,
                        interval=interval,
                        startTime=current_ts,
                        limit=limit,
                    )
                elif kline_type == "premium":
                    klines = self.client.futures_premium_index_klines(
                        symbol=self.symbol,
                        interval=interval,
                        startTime=current_ts,
                        endTime=end_ts,
                        limit=limit,
                    )
                else:  # default to spot
                    klines = self.client.get_klines(
                        symbol=self.symbol,
                        interval=interval,
                        startTime=current_ts,
                        limit=limit,
                    )

                # Process response and check for empty results
                if not klines:
                    self.logger.warning(
                        f"No data returned for {self.symbol} ({interval}) at {pd.to_datetime(current_ts, unit='ms')}"
                    )
                    break

                # Parse the klines into a DataFrame
                df = pd.DataFrame(
                    klines,
                    columns=[
                        "Timestamp",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "Close Time",
                        "Quote Asset Volume",
                        "Trades",
                        "Taker Buy Base",
                        "Taker Buy Quote",
                        "Ignore",
                    ],
                )

                # Keep only the columns we need
                df = df[["Timestamp", "Open", "High", "Low", "Close"]]
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")

                # Ensure numeric types for price columns
                for col in ["Open", "High", "Low", "Close"]:
                    df[col] = df[col].astype(float)

                # Add the data to our collection
                all_data.extend(df.values.tolist())

                # Get the timestamp of the last row for pagination
                last_timestamp = int(df.iloc[-1, 0].timestamp() * 1000)

                # Check if we've reached the end conditions
                if last_timestamp >= last_expected_ts:
                    self.logger.info(
                        f"Reached last expected timestamp at {pd.to_datetime(last_expected_ts, unit='ms')}, stopping."
                    )
                    break

                if len(klines) < limit:
                    self.logger.info(
                        f"Last batch received ({len(df)} rows) → End of available data."
                    )
                    break

                # Set up the next batch start time
                current_ts = last_timestamp + 1
                self.logger.info(
                    f"Fetched {len(df)} rows. Next start: {pd.to_datetime(current_ts, unit='ms')}"
                )

            except Exception as e:
                self.logger.error(
                    f"Error fetching {self.symbol} ({interval}): {str(e)}"
                )
                raise FetcherError(f"Error fetching klines: {e}")

        # Convert the accumulated data into a DataFrame
        if all_data:
            df = pd.DataFrame(
                all_data, columns=["Timestamp", "Open", "High", "Low", "Close"]
            )
            return df
        else:
            self.logger.warning(f"No data collected for {self.symbol} ({interval}).")
            return None

    def _fetch_funding_rates(self, interval, start_ts, end_ts):
        """
        Fetch funding rate data with pagination.

        Args:
            interval (str): Time interval (should be "8h" for funding rates)
            start_ts (int): Start timestamp in milliseconds
            end_ts (int): End timestamp in milliseconds

        Returns:
            pandas.DataFrame or None: DataFrame with funding rate data or None if no data
        """
        funding_rates = []
        current_ts = start_ts
        limit = 1000

        self.logger.info(f"Fetching funding rates for {self.symbol}")

        while current_ts < end_ts:
            try:
                self.logger.info(
                    f"Fetching {self.symbol} funding rates from {pd.to_datetime(current_ts, unit='ms')}"
                )

                rates = self.client.futures_funding_rate(
                    symbol=self.symbol,
                    startTime=current_ts,
                    endTime=end_ts,
                    limit=limit,
                )

                if not rates:
                    self.logger.warning(
                        f"No funding rate data returned for {self.symbol} at {pd.to_datetime(current_ts, unit='ms')}"
                    )
                    break

                funding_rates.extend(rates)

                # Get last timestamp and continue from there
                last_timestamp = rates[-1]["fundingTime"] + 1

                if last_timestamp >= end_ts:
                    self.logger.info(
                        f"Reached end timestamp {pd.to_datetime(end_ts, unit='ms')}, stopping."
                    )
                    break

                if len(rates) < limit:
                    self.logger.info(
                        f"Last batch received ({len(rates)} rows) → End of available data."
                    )
                    break

                current_ts = last_timestamp
                self.logger.info(
                    f"Fetched {len(rates)} rows. Next start: {pd.to_datetime(current_ts, unit='ms')}"
                )

            except Exception as e:
                self.logger.error(
                    f"Error fetching funding rates for {self.symbol}: {e}"
                )
                raise FetcherError(f"Error fetching funding rates: {e}")

        # Process and return data if available
        if funding_rates:
            try:
                df = pd.DataFrame(funding_rates)
                df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
                df["fundingRate"] = df["fundingRate"].astype(float)

                # Rename columns to match our standard format
                df = df.rename(
                    columns={"fundingTime": "Timestamp", "fundingRate": "Rate"}
                )

                # Format Timestamp column to ensure consistency
                df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

                return df

            except Exception as e:
                self.logger.error(
                    f"Error processing funding rate data for {self.symbol}: {e}"
                )
                return None
        else:
            self.logger.warning(f"No funding rate data collected for {self.symbol}")
            return None

    def _save_data(self, df, filepath, interval):
        """
        Save data to CSV.

        Args:
            df (pandas.DataFrame): DataFrame to save
            filepath (str or Path): Path where to save the file
            interval (str): Time interval

        Returns:
            Path: Path to the saved file
        """
        if df is not None and not df.empty:
            return save_to_csv(df, filepath, self.symbol, interval)
        else:
            self.logger.error(
                f"No valid data available to save for {self.symbol} ({interval})"
            )
            return None
