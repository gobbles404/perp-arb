# binance_data_pipeline/fetchers/spot.py
from pathlib import Path

from ..core.config import config
from ..exceptions import FetcherError
from .base import BinanceFetcher


class SpotFetcher(BinanceFetcher):
    """Fetcher for spot market data from Binance."""

    def fetch_for_interval(self, interval):
        """
        Fetch spot market data for a specific interval.

        Args:
            interval (str): Time interval (e.g., "1d", "8h", "1h")

        Returns:
            pandas.DataFrame or None: The fetched spot data or None if no data available
        """
        try:
            self.logger.info(f"Fetching spot data for {self.symbol} ({interval})")

            # Get timestamp range
            start_ts, end_ts = self._get_timestamp_range()

            # Fetch klines data - use the common base class method with "spot" type
            df = self._fetch_klines(interval, start_ts, end_ts, kline_type="spot")

            if df is not None and not df.empty:
                # Create the filepath using standardized directory structure
                # data/raw/{symbol}/{interval}/spot.csv
                filepath = Path(config.raw_dir) / self.symbol / interval / "spot.csv"

                # Save the data
                self._save_data(df, filepath, interval)

                self.logger.info(
                    f"Successfully fetched spot data for {self.symbol} ({interval}): {len(df)} rows"
                )
                return df
            else:
                self.logger.warning(
                    f"No spot data available for {self.symbol} ({interval})"
                )
                return None

        except Exception as e:
            self.logger.error(
                f"Error fetching spot data for {self.symbol} ({interval}): {e}"
            )
            raise FetcherError(f"Failed to fetch spot data: {e}")
