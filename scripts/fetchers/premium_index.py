#!/usr/bin/env python3
"""
Fetcher for premium index data from Binance.
"""

import pandas as pd
import sys
import os

# Import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import date_to_timestamp, save_to_csv
from .base import BinanceFetcher


class PremiumIndexFetcher(BinanceFetcher):
    """Fetcher for premium index data from perpetual futures"""

    def fetch_for_interval(self, interval):
        """
        Fetch premium index data for a specific interval.

        Args:
            interval (str): Time interval (e.g. "1d", "8h", "1h")
        """
        start_ts, end_ts = self._get_timestamp_range()
        df = self._fetch_premium_klines(interval, start_ts, end_ts)

        if df is not None and not df.empty:
            # Create path using new structure
            filepath = f"data/raw/{self.symbol}/{interval}/premium_index.csv"
            self._save_data(df, filepath, interval)
        else:
            self.logger.error(
                f"No valid data available to save for {self.symbol} ({interval})"
            )

    def _get_timestamp_range(self):
        """Convert date strings to timestamps"""
        return date_to_timestamp(self.start_date, self.end_date)

    def _fetch_premium_klines(self, interval, start_ts, end_ts):
        """
        Fetch premium index klines with pagination.

        Args:
            interval (str): Time interval
            start_ts (int): Start timestamp in milliseconds
            end_ts (int): End timestamp in milliseconds

        Returns:
            pandas.DataFrame or None: DataFrame with premium index data or None if no data
        """
        all_data = []
        current_ts = start_ts
        limit = 1000

        while current_ts < end_ts:
            try:
                self.logger.info(
                    f"Fetching {self.symbol} ({interval}) from {pd.to_datetime(current_ts, unit='ms')}"
                )

                klines = self.client.futures_premium_index_klines(
                    symbol=self.symbol,
                    interval=interval,
                    startTime=current_ts,
                    endTime=end_ts,
                    limit=limit,
                )

                if not klines:
                    self.logger.warning(
                        f"No data returned for {self.symbol} ({interval}) at {pd.to_datetime(current_ts, unit='ms')}"
                    )
                    break

                # Convert response to DataFrame
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
                df = df[["Timestamp", "Open", "High", "Low", "Close"]]
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                df[["Open", "High", "Low", "Close"]] = df[
                    ["Open", "High", "Low", "Close"]
                ].astype(float)

                all_data.extend(df.values.tolist())

                # Get last timestamp and continue from there
                last_timestamp = int(df.iloc[-1, 0].timestamp() * 1000)
                if last_timestamp >= end_ts:
                    self.logger.info(
                        f"✅ Reached end timestamp {pd.to_datetime(end_ts, unit='ms')}, stopping."
                    )
                    break

                current_ts = last_timestamp + 1  # Move past last timestamp
                self.logger.info(
                    f"Fetched {len(df)} rows. Next start: {pd.to_datetime(current_ts, unit='ms')}"
                )

            except Exception as e:
                self.logger.error(
                    f"Error fetching premium index klines for {self.symbol} ({interval}): {e}"
                )
                break

        # Return results if data is available
        if all_data:
            df = pd.DataFrame(
                all_data, columns=["Timestamp", "Open", "High", "Low", "Close"]
            )
            return df
        else:
            self.logger.warning(
                f"No premium index data collected for {self.symbol} ({interval})."
            )
            return None

    def _save_data(self, df, filepath, interval):
        """
        Save data to CSV.

        Args:
            df (pandas.DataFrame): DataFrame to save
            filename (str): Filename
            interval (str): Time interval
        """
        full_path = save_to_csv(df, filepath, self.symbol, interval)
        self.logger.info(
            f"Successfully saved {len(df)} rows for {self.symbol} ({interval}) to {filepath}"
        )
