#!/usr/bin/env python3
"""
Fetcher for funding rates data from Binance.
"""

import pandas as pd
import sys
import os

# Import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import date_to_timestamp, save_to_csv
from .base import BinanceFetcher


class FundingRatesFetcher(BinanceFetcher):
    """Fetcher for funding rates data from perpetual futures"""

    def __init__(self, symbol, intervals, start_date, end_date):
        """Initialize with default 8h interval if not specified"""
        # Funding rates are generally on 8h interval, ensure it's included
        if isinstance(intervals, list) and "8h" not in intervals:
            intervals.append("8h")
        elif intervals != "8h":
            intervals = ["8h"]

        super().__init__(symbol, intervals, start_date, end_date)

    def fetch_for_interval(self, interval):
        """
        Fetch funding rates data for a specific interval.

        Args:
            interval (str): Time interval (should be "8h" for funding rates)
        """
        # Funding rates are only available at 8h intervals
        if interval != "8h":
            self.logger.warning(
                f"Funding rates are only available at 8h intervals, not {interval}"
            )
            return

        start_ts, end_ts = self._get_timestamp_range()
        df = self._fetch_funding_rates(start_ts, end_ts)

        if df is not None and not df.empty:
            # Create path using new structure
            filepath = f"data/raw/{self.symbol}/{interval}/funding_rates.csv"
            self._save_data(df, filepath, interval)
        else:
            self.logger.error(
                f"No valid data available to save for {self.symbol} ({interval})"
            )

    def _get_timestamp_range(self):
        """Convert date strings to timestamps"""
        return date_to_timestamp(self.start_date, self.end_date)

    def _fetch_funding_rates(self, start_ts, end_ts):
        """
        Fetch funding rates with pagination.

        Args:
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
                break

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

                # Format Timestamp column to just display HH:MM:SS time format
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
            filepath (str): Full relative path where the file should be saved
            interval (str): Time interval
        """
        full_path = save_to_csv(df, filepath, self.symbol, interval)
        self.logger.info(
            f"Successfully saved {len(df)} rows for {self.symbol} ({interval}) to {filepath}"
        )
