# binance_data_pipeline/fetchers/futures.py
from pathlib import Path

from ..core.config import config
from ..exceptions import FetcherError
from ..utils.date_utils import extract_base_symbol
from .base import BinanceFetcher


class FuturesFetcher(BinanceFetcher):
    """Fetcher for both perpetual and quarterly futures data from Binance."""

    def fetch_for_interval(self, interval):
        """
        Fetch futures data for a specific interval.

        Args:
            interval (str): Time interval (e.g., "1d", "8h", "1h")

        Returns:
            pandas.DataFrame or None: The fetched futures data or None if no data available
        """
        try:
            self.logger.info(f"Fetching futures data for {self.symbol} ({interval})")

            # Get timestamp range
            start_ts, end_ts = self._get_timestamp_range()

            # Fetch klines data - use "futures" type for the kline fetcher
            df = self._fetch_klines(interval, start_ts, end_ts, kline_type="futures")

            if df is not None and not df.empty:
                # Determine whether this is a perpetual or quarterly futures contract
                if "_" in self.symbol:
                    # This is a quarterly futures contract (e.g. BTCUSDT_250627)
                    base_symbol = extract_base_symbol(self.symbol)
                    contract_suffix = self.symbol.split("_")[1]

                    # Create path: data/raw/{base_symbol}/{interval}/futures_contracts/{suffix}.csv
                    filepath = (
                        Path(config.raw_dir)
                        / base_symbol
                        / interval
                        / "futures_contracts"
                        / f"{contract_suffix}.csv"
                    )

                    self.logger.info(
                        f"Processing as quarterly futures contract: {self.symbol}"
                    )
                else:
                    # This is a perpetual futures contract (e.g. BTCUSDT)
                    # Create path: data/raw/{symbol}/{interval}/perps.csv
                    filepath = (
                        Path(config.raw_dir) / self.symbol / interval / "perps.csv"
                    )
                    self.logger.info(
                        f"Processing as perpetual futures contract: {self.symbol}"
                    )

                # Save the data
                self._save_data(df, filepath, interval)

                self.logger.info(
                    f"Successfully fetched futures data for {self.symbol} ({interval}): {len(df)} rows"
                )
                return df
            else:
                self.logger.warning(
                    f"No futures data available for {self.symbol} ({interval})"
                )
                return None

        except Exception as e:
            self.logger.error(
                f"Error fetching futures data for {self.symbol} ({interval}): {e}"
            )
            raise FetcherError(f"Failed to fetch futures data: {e}")

    def fetch_all_contracts(
        self, base_symbol=None, intervals=None, skip_contracts=False
    ):
        """
        Fetch data for both perpetual and all matching quarterly futures contracts.

        Args:
            base_symbol (str, optional): Base symbol to fetch contracts for. Defaults to self.symbol.
            intervals (list, optional): List of intervals to fetch. Defaults to self.intervals.
            skip_contracts (bool): If True, only fetch perpetual futures. Defaults to False.

        Returns:
            dict: Dictionary of fetched data by contract and interval
        """
        base_symbol = base_symbol or self.symbol
        intervals = intervals or self.intervals

        results = {base_symbol: {}}

        # First fetch data for the perpetual contract
        self.logger.info(f"Fetching perpetual futures data for {base_symbol}")

        # Temporarily store original symbol if different from base_symbol
        original_symbol = self.symbol

        try:
            # Set symbol to base_symbol for fetching perpetual
            self.symbol = base_symbol

            # Fetch for each interval
            for interval in intervals:
                results[base_symbol][interval] = self.fetch_for_interval(interval)

            # Then fetch data for all matching futures contracts if not skipped
            if not skip_contracts:
                contracts = self._get_matching_futures_contracts(base_symbol)

                if contracts:
                    self.logger.info(
                        f"Found {len(contracts)} futures contracts for {base_symbol}"
                    )

                    for contract in contracts:
                        self.logger.info(
                            f"Fetching data for futures contract: {contract}"
                        )
                        self.symbol = contract

                        results[contract] = {}
                        for interval in intervals:
                            results[contract][interval] = self.fetch_for_interval(
                                interval
                            )
                else:
                    self.logger.info(
                        f"No additional futures contracts found for {base_symbol}"
                    )

            return results

        finally:
            # Restore original symbol
            self.symbol = original_symbol

    def _get_matching_futures_contracts(self, base_symbol):
        """
        Read the fut_expirys.csv file and return all futures contracts matching the base symbol.

        Args:
            base_symbol (str): Base symbol to match contracts for

        Returns:
            list: List of matching contract symbols
        """
        import csv
        import os

        futures_contracts = []

        # Path to the fut_expirys.csv file
        csv_path = Path(config.contracts_dir) / "fut_expirys.csv"

        # Check if the file exists
        if not os.path.exists(csv_path):
            self.logger.warning(f"Futures contracts CSV file not found: {csv_path}")
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
            self.logger.error(f"Error reading futures contracts CSV: {e}")

        return futures_contracts
