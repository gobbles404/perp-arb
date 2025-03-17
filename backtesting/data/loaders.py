# perp_arb/backtesting/data/loaders.py
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple

# Import the logger
from core.logger import get_logger

# Create a logger instance
logger = get_logger(__name__)


class BaseDataLoader:
    """Base class for data loaders with common functionality."""

    def __init__(self, file_path: str):
        """
        Initialize the base data loader.

        Args:
            file_path: Path to the CSV file
        """
        self.file_path = file_path
        self.data = None
        logger.debug(f"Initialized BaseDataLoader with file: {file_path}")

    def load(self) -> pd.DataFrame:
        """Load the CSV file into a pandas DataFrame."""
        if self.data is None:
            if not os.path.exists(self.file_path):
                logger.error(f"File not found: {self.file_path}")
                raise FileNotFoundError(f"File not found: {self.file_path}")

            logger.debug(f"Loading data from: {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            logger.debug(f"Loaded {len(self.data)} rows from {self.file_path}")

        return self.data

    def get_data(self) -> pd.DataFrame:
        """Get the loaded data, loading it first if necessary."""
        if self.data is None:
            return self.load()
        return self.data


class MarketDataLoader(BaseDataLoader):
    """Loads and preprocesses market data from CSV files."""

    def __init__(self, file_path: str, symbol: Optional[str] = None):
        """
        Initialize the market data loader.

        Args:
            file_path: Path to the CSV file containing market data
            symbol: Trading symbol, if not provided will be extracted from file name
        """
        super().__init__(file_path)

        # Extract symbol and interval from file path if not provided
        if symbol is None:
            # Extract from filename (assuming format: {SYMBOL}_{INTERVAL}.csv)
            filename = os.path.basename(file_path)
            if "_" in filename:
                symbol = filename.split("_")[0]
            else:
                symbol = filename.replace(".csv", "")
            logger.debug(f"Extracted symbol '{symbol}' from filename")

        self.symbol = symbol
        logger.info(f"Initialized MarketDataLoader for symbol {symbol}")

    def load(self) -> pd.DataFrame:
        """Load the market data CSV file with appropriate date parsing."""
        if self.data is None:
            # Read the CSV file first
            logger.debug(f"Loading market data from {self.file_path}")
            raw_data = pd.read_csv(self.file_path)

            # Convert Timestamp column to datetime
            if "Timestamp" in raw_data.columns:
                logger.debug("Converting Timestamp column to datetime")
                raw_data["Timestamp"] = pd.to_datetime(
                    raw_data["Timestamp"], format="%Y-%m-%d"
                )

            self.data = raw_data
            logger.info(
                f"Loaded {len(self.data)} rows of market data for {self.symbol}"
            )

        return self.data

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the market data for backtesting.
        - Calculate basis (difference between perp and spot)
        - Ensure proper types
        - Handle missing values
        """
        logger.debug(f"Preprocessing market data for {self.symbol}")
        df = self.load().copy()

        # Calculate basis (difference between perp and spot prices)
        logger.debug("Calculating basis")
        df["basis"] = (df["perp_close"] / df["spot_close"] - 1) * 100  # as percentage

        # Calculate daily returns - use fill_method=None to avoid deprecation warning
        logger.debug("Calculating daily returns")
        df["spot_return"] = df["spot_close"].pct_change(fill_method=None) * 100
        df["perp_return"] = df["perp_close"].pct_change(fill_method=None) * 100

        # Handle prompt contract data if available
        if "prompt_close" in df.columns and not df["prompt_close"].isna().all():
            logger.debug("Processing prompt contract data")
            df["prompt_basis"] = (df["prompt_close"] / df["spot_close"] - 1) * 100
            df["prompt_return"] = df["prompt_close"].pct_change(fill_method=None) * 100

        # Handle next contract data if available
        if "next_close" in df.columns and not df["next_close"].isna().all():
            logger.debug("Processing next contract data")
            df["next_basis"] = (df["next_close"] / df["spot_close"] - 1) * 100
            df["next_return"] = df["next_close"].pct_change(fill_method=None) * 100

        # Ensure funding rate is available, set to 0 if not
        if "funding_rate" not in df.columns:
            logger.debug("No funding rate column found, adding with zeros")
            df["funding_rate"] = 0
        else:
            logger.debug("Forward-filling missing funding rates")
            # Forward fill any missing values - use ffill() instead of fillna(method='ffill')
            df["funding_rate"] = df["funding_rate"].ffill().fillna(0)

        # Calculate annualized funding rate if not available
        if "funding_annualized" not in df.columns and "funding_rate" in df.columns:
            logger.debug("Calculating annualized funding rate")
            # Assuming 3 funding payments per day (8h intervals)
            df["funding_annualized"] = df["funding_rate"] * 3 * 365

        # Calculate funding payments (daily effect of funding rate)
        df["funding_daily"] = df["funding_rate"]

        # Add symbol column for multi-asset processing
        df["symbol"] = self.symbol

        logger.info(f"Preprocessed market data for {self.symbol}, shape: {df.shape}")
        return df

    def get_events(self) -> List[Dict]:
        """
        Convert the preprocessed data into a list of market event dictionaries
        for the backtesting engine.
        """
        logger.debug(f"Converting market data for {self.symbol} to events")
        df = self.preprocess()

        events = []
        for _, row in df.iterrows():
            event = {
                "timestamp": row["Timestamp"],
                "symbol": self.symbol,
                "spot_open": row["spot_open"],
                "spot_high": row["spot_high"],
                "spot_low": row["spot_low"],
                "spot_close": row["spot_close"],
                "perp_open": row["perp_open"],
                "perp_high": row["perp_high"],
                "perp_low": row["perp_low"],
                "perp_close": row["perp_close"],
                "funding_rate": row["funding_rate"],
                "basis": row["basis"],
            }

            # Add contract data if available
            if "prompt_close" in row and not pd.isna(row["prompt_close"]):
                event["prompt_close"] = row["prompt_close"]
                event["prompt_contract"] = row.get("prompt_contract", None)
                event["prompt_days_till_expiry"] = row.get(
                    "prompt_days_till_expiry", None
                )

            if "next_close" in row and not pd.isna(row["next_close"]):
                event["next_close"] = row["next_close"]
                event["next_contract"] = row.get("next_contract", None)
                event["next_days_till_expiry"] = row.get("next_days_till_expiry", None)

            events.append(event)

        logger.info(f"Created {len(events)} events for {self.symbol}")
        return events


class ContractSpecsLoader(BaseDataLoader):
    """Loads contract specifications from CSV file."""

    def __init__(self, file_path: str):
        """
        Initialize the contract specifications loader.

        Args:
            file_path: Path to the CSV file containing contract specs
        """
        super().__init__(file_path)
        self._specs_by_symbol = None
        logger.info(f"Initialized ContractSpecsLoader with {file_path}")

    def load(self) -> pd.DataFrame:
        """Load the contract specs CSV file."""
        logger.debug(f"Loading contract specifications from {self.file_path}")
        return super().load()

    def get_specs_by_symbol(self) -> Dict[str, Dict]:
        """
        Convert contract specs to a lookup dictionary by symbol.
        """
        if self._specs_by_symbol is None:
            logger.debug("Building contract specs dictionary by symbol")
            df = self.load()

            self._specs_by_symbol = {}
            for _, row in df.iterrows():
                symbol = row["symbol"]

                # Create a dictionary of specs for this symbol
                specs = {
                    "symbol": symbol,
                    "pair": row["pair"],
                    "contract_type": row["contractType"],
                    "delivery_date": row["deliveryDate"],
                    "onboard_date": row["onboardDate"],
                    "status": row["status"],
                    "maint_margin_pct": row["maintMarginPercent"],
                    "required_margin_pct": row["requiredMarginPercent"],
                    "base_asset": row["baseAsset"],
                    "quote_asset": row["quoteAsset"],
                    "margin_asset": row["marginAsset"],
                    "liquidation_fee": row["liquidationFee"],
                }

                self._specs_by_symbol[symbol] = specs

            logger.info(
                f"Created specs dictionary with {len(self._specs_by_symbol)} symbols"
            )

        return self._specs_by_symbol

    def get_specs(self, symbol: str) -> Optional[Dict]:
        """
        Get contract specifications for a specific symbol.

        Args:
            symbol: Contract symbol

        Returns:
            Dictionary of contract specifications or None if not found
        """
        specs_by_symbol = self.get_specs_by_symbol()
        specs = specs_by_symbol.get(symbol)

        if specs:
            logger.debug(f"Found contract specs for {symbol}")
        else:
            logger.warning(f"No contract specs found for {symbol}")

        return specs


class DataManager:
    """
    Manages loading and joining market data with contract specifications.
    """

    def __init__(self, market_data_dir: str, contract_specs_file: str):
        """
        Initialize the data manager.

        Args:
            market_data_dir: Directory containing market data CSV files
            contract_specs_file: Path to contract specifications CSV file
        """
        self.market_data_dir = market_data_dir
        self.contract_specs_file = contract_specs_file

        logger.info(f"Initializing DataManager with market data dir: {market_data_dir}")
        logger.info(f"Contract specs file: {contract_specs_file}")

        # Initialize contract specs loader
        self.contract_specs_loader = ContractSpecsLoader(contract_specs_file)

        # Dictionary to store market data loaders by symbol
        self.market_data_loaders = {}

    def get_available_symbols(self) -> List[str]:
        """Get list of available market data symbols from CSV files in directory."""
        logger.debug(f"Scanning for available symbols in {self.market_data_dir}")
        symbols = []

        if os.path.exists(self.market_data_dir):
            for filename in os.listdir(self.market_data_dir):
                if filename.endswith(".csv"):
                    # Extract symbol from filename (assuming format: {SYMBOL}_{INTERVAL}.csv)
                    if "_" in filename:
                        symbol = filename.split("_")[0]
                    else:
                        symbol = filename.replace(".csv", "")

                    symbols.append(symbol)

            logger.info(f"Found {len(symbols)} symbols in market data directory")
        else:
            logger.warning(f"Market data directory not found: {self.market_data_dir}")

        return symbols

    def get_market_data_loader(
        self, symbol: str, interval: str = "1d"
    ) -> MarketDataLoader:
        """
        Get or create a market data loader for the specified symbol and interval.

        Args:
            symbol: Trading symbol
            interval: Data interval (e.g., '1d', '1h')

        Returns:
            MarketDataLoader for the specified symbol
        """
        loader_key = f"{symbol}_{interval}"

        if loader_key not in self.market_data_loaders:
            logger.debug(f"Creating new MarketDataLoader for {loader_key}")
            file_path = os.path.join(self.market_data_dir, f"{symbol}_{interval}.csv")
            self.market_data_loaders[loader_key] = MarketDataLoader(file_path, symbol)
        else:
            logger.debug(f"Using existing MarketDataLoader for {loader_key}")

        return self.market_data_loaders[loader_key]

    def load_data_with_specs(
        self, symbol: str, interval: str = "1d"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Load market data and contract specifications for a symbol.

        Args:
            symbol: Trading symbol
            interval: Data interval

        Returns:
            Tuple of (market_data_df, contract_specs_dict)
        """
        logger.info(f"Loading data with specs for {symbol}_{interval}")

        # Get market data
        market_loader = self.get_market_data_loader(symbol, interval)
        market_data = market_loader.preprocess()

        # Get contract specs
        contract_specs = self.contract_specs_loader.get_specs(symbol)

        return market_data, contract_specs

    def load_all_symbols(self, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Load market data for all available symbols.

        Args:
            interval: Data interval

        Returns:
            Dictionary mapping symbols to their preprocessed market data
        """
        logger.info(f"Loading data for all available symbols with interval {interval}")
        all_data = {}
        for symbol in self.get_available_symbols():
            logger.debug(f"Loading data for {symbol}")
            loader = self.get_market_data_loader(symbol, interval)
            all_data[symbol] = loader.preprocess()

        logger.info(f"Loaded data for {len(all_data)} symbols")
        return all_data
