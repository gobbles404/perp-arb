#!/usr/bin/env python3
"""
Fetcher for spot market data from Binance.
"""

import pandas as pd
import sys
import os

# Import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_last_expected_timestamp, date_to_timestamp, save_to_csv
from .base import BinanceFetcher


class SpotFetcher(BinanceFetcher):
    """Fetcher for spot market data"""

    def fetch_for_interval(self, interval):
        """
        Fetch spot market data for a specific interval.

        Args:
            interval (str): Time interval (e.g. "1d", "8h", "1h")
        """
        start_ts, end_ts = self._get_timestamp_range()
        df = self._fetch_klines(interval, start_ts, end_ts)

        if df is not None and not df.empty:
            # Create path using new structure
            filepath = f"data/raw/{self.symbol}/{interval}/spot.csv"
            self._save_data(df, filepath, interval)
        else:
            self.logger.error(
                f"No valid data available to save for {self.symbol} ({interval})"
            )

    def _get_timestamp_range(self):
        """Convert date strings to timestamps"""
        return date_to_timestamp(self.start_date, self.end_date)

    def _fetch_klines(self, interval, start_ts, end_ts):
        """
        Fetch kline data with pagination.

        Args:
            interval (str): Time interval
            start_ts (int): Start timestamp in milliseconds
            end_ts (int): End timestamp in milliseconds

        Returns:
            pandas.DataFrame or None: DataFrame with kline data or None if no data
        """
        all_data = []
        current_ts = start_ts
        limit = 1000

        last_expected_ts = get_last_expected_timestamp(interval)

        while current_ts < end_ts:
            self.logger.info(
                f"Fetching {self.symbol} ({interval}) from {pd.to_datetime(current_ts, unit='ms')}"
            )

            try:
                klines = self.client.get_klines(
                    symbol=self.symbol,
                    interval=interval,
                    startTime=current_ts,
                    limit=limit,
                )

                if not klines:
                    self.logger.warning(
                        f"No data returned for {self.symbol} ({interval}) at {pd.to_datetime(current_ts, unit='ms')}"
                    )
                    break

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
                df["Close"] = df["Close"].astype(float)

                all_data.extend(df.values.tolist())

                last_timestamp = int(df.iloc[-1, 0].timestamp() * 1000)
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

                current_ts = last_timestamp + 1
                self.logger.info(
                    f"Fetched {len(df)} rows. Next start: {pd.to_datetime(current_ts, unit='ms')}"
                )

            except Exception as e:
                self.logger.error(
                    f"Error fetching {self.symbol} ({interval}): {str(e)}"
                )
                break

        if all_data:
            df = pd.DataFrame(
                all_data, columns=["Timestamp", "Open", "High", "Low", "Close"]
            )
            return df
        else:
            self.logger.warning(f"No data collected for {self.symbol} ({interval}).")
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
