# binance_data_pipeline/core/config.py
import os
from datetime import datetime, timedelta
from pathlib import Path


class Config:
    """Configuration management for Binance data pipeline."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._initialize()
        return cls._instance

    @classmethod
    def _initialize(cls):
        # Project paths
        cls.root_dir = Path(__file__).parent.parent.parent.parent
        cls.package_dir = Path(__file__).parent.parent

        # Date settings - migrate from your current config.py
        cls.default_start_date = "2023-06-01"
        cls.default_end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Default intervals - migrate from your current config.py
        cls.default_intervals = {
            "spot": ["1d", "8h", "1h"],
            "perpetuals": ["1d", "8h", "1h"],
            "futures": ["1d", "8h", "1h"],
            "funding": ["8h"],
        }

        # Default symbol
        cls.default_symbol = "BTCUSDT"

        # Directory structure
        cls.data_dir = cls.root_dir / "binance_data_pipeline" / "data"
        cls.raw_dir = cls.data_dir / "raw"
        cls.processed_dir = cls.data_dir / "processed"
        cls.markets_dir = cls.data_dir / "markets"
        cls.contracts_dir = cls.data_dir / "contracts"

        # Ensure directories exist
        cls._ensure_directories()

    @classmethod
    def _ensure_directories(cls):
        """Create data directories if they don't exist."""
        for directory in [
            cls.data_dir,
            cls.raw_dir,
            cls.processed_dir,
            cls.markets_dir,
            cls.contracts_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_raw_data_path(cls, symbol, interval, data_type):
        """Generate path for raw data files."""
        return cls.raw_dir / symbol / interval / f"{data_type}.csv"

    @classmethod
    def get_processed_data_path(cls, symbol, interval, filename):
        """Generate path for processed data files."""
        return cls.processed_dir / symbol / interval / filename

    @classmethod
    def get_market_data_path(cls, symbol, interval):
        """Generate path for consolidated market data."""
        return cls.markets_dir / f"{symbol}_{interval}.csv"


# Create a global config instance
config = Config()
