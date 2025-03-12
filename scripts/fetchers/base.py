#!/usr/bin/env python3
"""
Base class for all Binance data fetchers.
"""

import logging
from datetime import datetime, timezone
import pandas as pd
import sys
import os

# Import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binance_client import client
from utils import date_to_timestamp, save_to_csv


class BinanceFetcher:
    """Base class for fetching data from Binance"""

    def __init__(self, symbol, intervals, start_date, end_date):
        """
        Initialize the fetcher with common parameters.

        Args:
            symbol (str): Trading pair symbol
            intervals (list): List of time intervals (e.g. ["1d", "8h", "1h"])
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        self.symbol = symbol
        self.intervals = intervals if isinstance(intervals, list) else [intervals]
        self.start_date = start_date
        self.end_date = end_date
        self.client = client
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for this fetcher"""
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:  # Only add handler if not already configured
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def fetch_all(self):
        """Fetch data for all configured intervals"""
        self.logger.info(f"Starting data fetch for {self.symbol}")

        for interval in self.intervals:
            self.logger.info(f"Processing {self.symbol} with interval {interval}")
            self.fetch_for_interval(interval)

        self.logger.info("Completed data fetch")

    def fetch_for_interval(self, interval):
        """
        Fetch data for a specific interval - to be implemented by subclasses

        Args:
            interval (str): Time interval (e.g. "1d", "8h", "1h")
        """
        raise NotImplementedError("Subclasses must implement this method")
