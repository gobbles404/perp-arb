# perp_arb/backtesting/data/market_data.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any

# Import the logger
from core.logger import get_logger

# Create a logger instance for market_data
logger = get_logger(__name__)


class MarketData:
    """
    Container for market data with methods for accessing and analyzing the data.
    Provides a standardized interface for the backtesting engine.
    """

    def __init__(
        self,
        symbol: str,
        data: pd.DataFrame,
        contract_specs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the market data object.

        Args:
            symbol: Trading symbol
            data: Preprocessed market data DataFrame
            contract_specs: Contract specifications (optional)
        """
        self.symbol = symbol
        self.data = data
        self.contract_specs = contract_specs or {}

        # Ensure required columns exist
        self._validate_data()

        # Set index to timestamp if not already
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if "Timestamp" in self.data.columns:
                self.data = self.data.set_index("Timestamp")
                logger.debug(f"Set index to Timestamp for {symbol}")
            else:
                logger.error(f"Data for {symbol} must contain a 'Timestamp' column")
                raise ValueError("Data must contain a 'Timestamp' column")

        logger.info(f"Initialized market data for {symbol} with {len(data)} rows")

    def _validate_data(self) -> None:
        """Validate that the data contains required columns."""
        required_columns = [
            "spot_close",
            "perp_close",
            "spot_open",
            "perp_open",
            "spot_high",
            "perp_high",
            "spot_low",
            "perp_low",
        ]

        missing_columns = [
            col for col in required_columns if col not in self.data.columns
        ]
        if missing_columns:
            logger.error(
                f"Data for {self.symbol} is missing required columns: {missing_columns}"
            )
            raise ValueError(f"Data is missing required columns: {missing_columns}")

        logger.debug(f"Validated data columns for {self.symbol}")

    def get_close_prices(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get spot and perpetual close prices for the specified date range.

        Args:
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            DataFrame with spot and perp close prices
        """
        df = self.data.copy()

        # Filter by date range if specified
        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        logger.debug(f"Getting close prices for {self.symbol} ({len(df)} rows)")
        # Return only close prices
        return df[["spot_close", "perp_close"]]

    def get_price_data(
        self,
        instrument: str = "spot",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get OHLC price data for the specified instrument and date range.

        Args:
            instrument: 'spot', 'perp', 'prompt', or 'next'
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            DataFrame with OHLC prices
        """
        df = self.data.copy()

        # Filter by date range if specified
        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        # Select columns based on instrument
        columns = [
            f"{instrument}_{field}" for field in ["open", "high", "low", "close"]
        ]

        # Check if columns exist
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            logger.error(
                f"Data for {self.symbol} is missing columns for {instrument}: {missing_columns}"
            )
            raise ValueError(
                f"Data is missing columns for {instrument}: {missing_columns}"
            )

        logger.debug(
            f"Getting {instrument} OHLC data for {self.symbol} ({len(df)} rows)"
        )
        return df[columns]

    def get_basis(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Get basis (premium/discount) between perp and spot.

        Args:
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            Series with basis values
        """
        df = self.data.copy()

        # Filter by date range if specified
        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        # Return basis column if it exists
        if "basis" in df.columns:
            logger.debug(
                f"Getting basis data for {self.symbol} from existing column ({len(df)} rows)"
            )
            return df["basis"]

        # Otherwise calculate it
        logger.debug(f"Calculating basis data for {self.symbol} ({len(df)} rows)")
        return (df["perp_close"] / df["spot_close"] - 1) * 100

    def get_funding_rates(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Get funding rates for the specified date range.

        Args:
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            Series with funding rate values
        """
        df = self.data.copy()

        # Filter by date range if specified
        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        # Return funding rate column if it exists
        if "funding_rate" in df.columns:
            logger.debug(
                f"Getting funding rates for {self.symbol} from existing column ({len(df)} rows)"
            )
            return df["funding_rate"]

        # Otherwise return zeros
        logger.warning(f"No funding rate data found for {self.symbol}, returning zeros")
        return pd.Series(0, index=df.index)

    def calculate_rolling_zscore(self, window: int = 20) -> pd.Series:
        """
        Calculate rolling Z-score of the basis.

        Args:
            window: Rolling window size

        Returns:
            Series with Z-score values
        """
        basis = self.get_basis()

        # Calculate rolling mean and standard deviation
        rolling_mean = basis.rolling(window=window).mean()
        rolling_std = basis.rolling(window=window).std()

        # Calculate Z-score
        zscore = (basis - rolling_mean) / rolling_std

        logger.debug(f"Calculated Z-scores for {self.symbol} with window size {window}")
        return zscore

    def get_contract_details(self) -> Dict[str, Any]:
        """
        Get contract specifications for this market.

        Returns:
            Dictionary with contract specifications
        """
        if not self.contract_specs:
            logger.warning(f"No contract specifications available for {self.symbol}")
        else:
            logger.debug(f"Retrieved contract details for {self.symbol}")

        return self.contract_specs

    def get_margin_requirements(self) -> Dict[str, float]:
        """
        Get margin requirements for this market.

        Returns:
            Dictionary with initial and maintenance margin percentages
        """
        if not self.contract_specs:
            logger.warning(
                f"No contract specs found for {self.symbol}, using default margin requirements"
            )
            return {
                "initial_margin": 0.05,  # Default 5% initial margin
                "maintenance_margin": 0.025,  # Default 2.5% maintenance margin
            }

        margins = {
            "initial_margin": self.contract_specs.get("required_margin_pct", 0.05)
            / 100,
            "maintenance_margin": self.contract_specs.get("maint_margin_pct", 0.025)
            / 100,
        }

        logger.debug(f"Margin requirements for {self.symbol}: {margins}")
        return margins

    def get_liquidation_fee(self) -> float:
        """
        Get liquidation fee for this market.

        Returns:
            Liquidation fee as a percentage
        """
        if not self.contract_specs:
            logger.warning(
                f"No contract specs found for {self.symbol}, using default liquidation fee"
            )
            return 0.015  # Default 1.5% liquidation fee

        fee = self.contract_specs.get("liquidation_fee", 0.015)
        logger.debug(f"Liquidation fee for {self.symbol}: {fee:.4f}")
        return fee


class MultiMarketData:
    """
    Container for multiple market data objects for cross-market strategies.
    """

    def __init__(self, markets: Dict[str, MarketData]):
        """
        Initialize the multi-market data object.

        Args:
            markets: Dictionary mapping symbols to MarketData objects
        """
        self.markets = markets

        # Validate that all markets have data for the same time period
        self._validate_date_alignment()

        logger.info(
            f"Initialized MultiMarketData with {len(markets)} markets: {list(markets.keys())}"
        )

    def _validate_date_alignment(self) -> None:
        """Validate that all markets have data for the same time period."""
        if not self.markets:
            logger.warning("No markets provided to MultiMarketData")
            return

        # Get first market's date range as reference
        first_symbol = next(iter(self.markets))
        first_market = self.markets[first_symbol]
        reference_dates = first_market.data.index

        # Check if all other markets have the same dates
        for symbol, market in self.markets.items():
            if symbol == first_symbol:
                continue

            if not market.data.index.equals(reference_dates):
                # If dates don't match exactly, check for significant differences
                missing_dates = len(reference_dates) - len(market.data.index)
                if abs(missing_dates) > 5:  # Allow small discrepancies
                    logger.warning(
                        f"Market {symbol} has {abs(missing_dates)} more/less dates than {first_symbol}"
                    )

    def get_common_dates(self) -> pd.DatetimeIndex:
        """
        Get dates common to all markets.

        Returns:
            DatetimeIndex of common dates
        """
        if not self.markets:
            logger.warning("No markets to get common dates from")
            return pd.DatetimeIndex([])

        # Get intersection of all date indices
        common_dates = None
        for market in self.markets.values():
            if common_dates is None:
                common_dates = set(market.data.index)
            else:
                common_dates &= set(market.data.index)

        # Convert to DatetimeIndex and sort
        common_dates_index = pd.DatetimeIndex(sorted(common_dates))
        logger.debug(f"Found {len(common_dates_index)} common dates across all markets")
        return common_dates_index

    def get_close_prices_all(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get close prices for all markets.

        Args:
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            DataFrame with close prices for all markets
        """
        close_prices = {}

        for symbol, market in self.markets.items():
            market_prices = market.get_close_prices(start_date, end_date)

            # Rename columns to include symbol
            renamed_columns = {
                "spot_close": f"{symbol}_spot",
                "perp_close": f"{symbol}_perp",
            }
            market_prices = market_prices.rename(columns=renamed_columns)

            close_prices[symbol] = market_prices

        # Concatenate all price DataFrames
        if close_prices:
            result = pd.concat(close_prices.values(), axis=1)
            logger.debug(f"Retrieved close prices for all markets: {result.shape}")
            return result
        else:
            logger.warning("No price data to concatenate")
            return pd.DataFrame()

    def calculate_correlation_matrix(
        self, instrument: str = "spot", lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between markets.

        Args:
            instrument: 'spot' or 'perp'
            lookback_days: Number of days to use for correlation calculation

        Returns:
            DataFrame with correlation matrix
        """
        # Get close prices for all markets
        prices = {}

        for symbol, market in self.markets.items():
            price_column = f"{instrument}_close"
            if price_column in market.data.columns:
                prices[symbol] = market.data[price_column]
            else:
                logger.warning(f"Price column '{price_column}' not found for {symbol}")

        if not prices:
            logger.warning(f"No {instrument} prices found for correlation calculation")
            return pd.DataFrame()

        # Create DataFrame with all prices
        price_df = pd.DataFrame(prices)

        # Calculate returns
        returns_df = price_df.pct_change(fill_method=None).dropna()

        # Limit to lookback period
        if lookback_days and len(returns_df) > lookback_days:
            returns_df = returns_df.iloc[-lookback_days:]

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        logger.debug(
            f"Calculated correlation matrix for {instrument} prices with {lookback_days} day lookback"
        )
        return corr_matrix
