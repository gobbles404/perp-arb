# perp_arb/backtesting/data/contract_specs.py
import pandas as pd
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import the logger
from core.logger import get_logger

# Create a logger instance for contract_specs
logger = get_logger(__name__)


class ContractSpecification:
    """
    Class representing contract specifications for futures and perpetuals.
    """

    def __init__(self, specs_data: Dict[str, Any]):
        """
        Initialize contract specification from data dictionary.

        Args:
            specs_data: Dictionary containing contract specifications
        """
        self.symbol = specs_data.get("symbol", "")
        self.pair = specs_data.get("pair", "")
        self.contract_type = specs_data.get("contractType", "PERPETUAL")

        # Convert timestamps to datetime if available
        self.delivery_date = self._parse_timestamp(specs_data.get("deliveryDate", 0))
        self.onboard_date = self._parse_timestamp(specs_data.get("onboardDate", 0))

        self.status = specs_data.get("status", "TRADING")
        self.maint_margin_pct = specs_data.get("maintMarginPercent", 2.5)
        self.required_margin_pct = specs_data.get("requiredMarginPercent", 5.0)

        self.base_asset = specs_data.get("baseAsset", "")
        self.quote_asset = specs_data.get("quoteAsset", "")
        self.margin_asset = specs_data.get("marginAsset", "")

        self.price_precision = specs_data.get("pricePrecision", 2)
        self.quantity_precision = specs_data.get("quantityPrecision", 3)

        self.liquidation_fee = specs_data.get(
            "liquidationFee", 0.015
        )  # Default to 1.5%
        self.market_take_bound = specs_data.get(
            "marketTakeBound", 0.10
        )  # Default to 10%

        logger.debug(f"Created contract specification for {self.symbol}")

    def _parse_timestamp(self, timestamp: int) -> Optional[datetime]:
        """Parse timestamp to datetime if valid."""
        if timestamp and timestamp > 0:
            # Binance timestamps are in milliseconds
            return datetime.fromtimestamp(timestamp / 1000)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert contract specification to dictionary."""
        return {
            "symbol": self.symbol,
            "pair": self.pair,
            "contract_type": self.contract_type,
            "delivery_date": self.delivery_date,
            "onboard_date": self.onboard_date,
            "status": self.status,
            "maint_margin_pct": self.maint_margin_pct,
            "required_margin_pct": self.required_margin_pct,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "margin_asset": self.margin_asset,
            "price_precision": self.price_precision,
            "quantity_precision": self.quantity_precision,
            "liquidation_fee": self.liquidation_fee,
            "market_take_bound": self.market_take_bound,
        }

    def is_active(self) -> bool:
        """Check if contract is active (not expired or delisted)."""
        if self.status != "TRADING":
            logger.debug(f"{self.symbol} is not active: status is {self.status}")
            return False

        if self.delivery_date and self.delivery_date < datetime.now():
            logger.debug(
                f"{self.symbol} is expired: delivery date {self.delivery_date}"
            )
            return False

        return True

    def days_to_expiry(self) -> Optional[int]:
        """
        Calculate days to expiry for futures contracts.

        Returns:
            Number of days to expiry or None for perpetual contracts
        """
        if self.contract_type == "PERPETUAL" or not self.delivery_date:
            return None

        # Calculate days to expiry
        days = (self.delivery_date - datetime.now()).days
        days_to_expiry = max(0, days)

        logger.debug(f"{self.symbol} days to expiry: {days_to_expiry}")
        return days_to_expiry

    def get_max_leverage(self) -> float:
        """
        Calculate maximum allowed leverage based on margin requirements.

        Returns:
            Maximum leverage as a float
        """
        if self.required_margin_pct <= 0:
            logger.warning(
                f"{self.symbol} has invalid required margin: {self.required_margin_pct}, returning default max leverage"
            )
            return 100.0  # Default high value if margin requirement is invalid

        max_leverage = 100.0 / self.required_margin_pct
        logger.debug(f"{self.symbol} max leverage: {max_leverage:.2f}")
        return max_leverage


class ContractSpecificationRegistry:
    """
    Registry for contract specifications with lookup by symbol.
    """

    def __init__(self, specs_file: Optional[str] = None):
        """
        Initialize the contract specification registry.

        Args:
            specs_file: Path to CSV file containing contract specifications
        """
        self.specs_by_symbol: Dict[str, ContractSpecification] = {}

        # Load specs if file provided
        if specs_file and os.path.exists(specs_file):
            self.load_from_file(specs_file)
            logger.info(f"Loaded contract specifications from {specs_file}")
        elif specs_file:
            logger.warning(f"Contract specs file not found: {specs_file}")

    def load_from_file(self, file_path: str) -> None:
        """
        Load contract specifications from CSV file.

        Args:
            file_path: Path to CSV file
        """
        if not os.path.exists(file_path):
            logger.error(f"Contract specs file not found: {file_path}")
            raise FileNotFoundError(f"Contract specs file not found: {file_path}")

        # Load CSV file
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} contract specifications from {file_path}")

        # Convert each row to ContractSpecification
        for _, row in df.iterrows():
            spec = ContractSpecification(row.to_dict())
            self.add_specification(spec)

    def add_specification(self, spec: ContractSpecification) -> None:
        """
        Add a contract specification to the registry.

        Args:
            spec: Contract specification
        """
        self.specs_by_symbol[spec.symbol] = spec
        logger.debug(f"Added specification for {spec.symbol}")

    def get_specification(self, symbol: str) -> Optional[ContractSpecification]:
        """
        Get contract specification for a symbol.

        Args:
            symbol: Contract symbol

        Returns:
            ContractSpecification or None if not found
        """
        spec = self.specs_by_symbol.get(symbol)
        if spec is None:
            logger.debug(f"No specification found for {symbol}")
        return spec

    def list_active_contracts(self) -> List[ContractSpecification]:
        """
        List all active contracts.

        Returns:
            List of active contract specifications
        """
        active_contracts = [
            spec for spec in self.specs_by_symbol.values() if spec.is_active()
        ]
        logger.debug(f"Found {len(active_contracts)} active contracts")
        return active_contracts

    def list_perpetuals(self) -> List[ContractSpecification]:
        """
        List all perpetual contracts.

        Returns:
            List of perpetual contract specifications
        """
        perpetuals = [
            spec
            for spec in self.specs_by_symbol.values()
            if spec.contract_type == "PERPETUAL" and spec.is_active()
        ]
        logger.debug(f"Found {len(perpetuals)} active perpetual contracts")
        return perpetuals

    def list_futures(self) -> List[ContractSpecification]:
        """
        List all futures contracts.

        Returns:
            List of futures contract specifications
        """
        futures = [
            spec
            for spec in self.specs_by_symbol.values()
            if spec.contract_type != "PERPETUAL" and spec.is_active()
        ]
        logger.debug(f"Found {len(futures)} active futures contracts")
        return futures

    def list_by_underlying(self, base_asset: str) -> List[ContractSpecification]:
        """
        List all contracts for a specific underlying asset.

        Args:
            base_asset: Base asset symbol (e.g., 'BTC')

        Returns:
            List of contract specifications for the base asset
        """
        asset_contracts = [
            spec
            for spec in self.specs_by_symbol.values()
            if spec.base_asset == base_asset and spec.is_active()
        ]
        logger.debug(f"Found {len(asset_contracts)} active contracts for {base_asset}")
        return asset_contracts

    def get_margin_requirements(self, symbol: str) -> Dict[str, float]:
        """
        Get margin requirements for a symbol.

        Args:
            symbol: Contract symbol

        Returns:
            Dictionary with initial and maintenance margin percentages
        """
        spec = self.get_specification(symbol)
        if not spec:
            logger.warning(
                f"No specification found for {symbol}, using default margin requirements"
            )
            # Return default values if specification not found
            return {"initial_margin": 0.05, "maintenance_margin": 0.025}  # 5%  # 2.5%

        margins = {
            "initial_margin": spec.required_margin_pct / 100,
            "maintenance_margin": spec.maint_margin_pct / 100,
        }
        logger.debug(f"Margin requirements for {symbol}: {margins}")
        return margins

    def get_liquidation_fee(self, symbol: str) -> float:
        """
        Get liquidation fee for a symbol.

        Args:
            symbol: Contract symbol

        Returns:
            Liquidation fee as a percentage
        """
        spec = self.get_specification(symbol)
        if not spec:
            logger.warning(
                f"No specification found for {symbol}, using default liquidation fee"
            )
            return 0.015  # Default 1.5%

        logger.debug(f"Liquidation fee for {symbol}: {spec.liquidation_fee:.4f}")
        return spec.liquidation_fee
