# binance_data_pipeline/fetchers/funding_rates.py
from pathlib import Path

from ..core.config import config
from ..exceptions import FetcherError
from .base import BinanceFetcher


class FundingRatesFetcher(BinanceFetcher):
    """Fetcher for funding rates data from perpetual futures."""

    def __init__(self, symbol=None, intervals=None, start_date=None, end_date=None):
        """
        Initialize with default 8h interval if not specified.

        Args:
            symbol (str, optional): Trading pair symbol
            intervals (list, optional): Time intervals (funding rates are always 8h)
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
        """
        # Funding rates are generally on 8h interval, ensure it's included
        if isinstance(intervals, list) and "8h" not in intervals:
            intervals.append("8h")
        elif intervals != "8h" and intervals is not None:
            intervals = ["8h"]
        else:
            intervals = ["8h"]

        super().__init__(symbol, intervals, start_date, end_date)

    def fetch_for_interval(self, interval):
        """
        Fetch funding rates data for a specific interval.

        Args:
            interval (str): Time interval (should be "8h" for funding rates)

        Returns:
            pandas.DataFrame or None: The fetched funding rates data or None if no data available
        """
        # Funding rates are only available at 8h intervals
        if interval != "8h":
            self.logger.warning(
                f"Funding rates are only available at 8h intervals, not {interval}"
            )
            return None

        try:
            self.logger.info(f"Fetching funding rates for {self.symbol} ({interval})")

            # Get timestamp range
            start_ts, end_ts = self._get_timestamp_range()

            # Use our specialized funding rates fetcher from the base class
            df = self._fetch_funding_rates(interval, start_ts, end_ts)

            if df is not None and not df.empty:
                # Create path: data/raw/{symbol}/{interval}/funding_rates.csv
                filepath = (
                    Path(config.raw_dir) / self.symbol / interval / "funding_rates.csv"
                )

                # Save the data
                self._save_data(df, filepath, interval)

                self.logger.info(
                    f"Successfully fetched funding rates for {self.symbol} ({interval}): {len(df)} rows"
                )
                return df
            else:
                self.logger.warning(
                    f"No funding rates available for {self.symbol} ({interval})"
                )
                return None

        except Exception as e:
            self.logger.error(
                f"Error fetching funding rates for {self.symbol} ({interval}): {e}"
            )
            raise FetcherError(f"Failed to fetch funding rates: {e}")

    def get_annualized_rate(self, rate):
        """
        Calculate the annualized funding rate.

        Args:
            rate (float): The 8-hour funding rate

        Returns:
            float: Annualized funding rate (3 funding events per day * 365 days)
        """
        return rate * 3 * 365  # 8h funding occurs 3 times per day
