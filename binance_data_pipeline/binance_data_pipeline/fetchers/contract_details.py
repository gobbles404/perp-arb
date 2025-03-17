# binance_data_pipeline/fetchers/contract_details.py
from pathlib import Path
import pandas as pd
import csv

from ..core.config import config
from ..exceptions import FetcherError
from .base import BinanceFetcher


class ContractDetailsFetcher(BinanceFetcher):
    """Fetcher for contract specifications from Binance."""

    def __init__(
        self,
        contract_type="both",
        symbol=None,
        intervals=None,
        start_date=None,
        end_date=None,
    ):
        """
        Initialize with contract type.

        Args:
            contract_type (str): Type of contracts to fetch ('futures', 'spot', or 'both')
            symbol (str, optional): Not used for contract details but kept for compatibility
            intervals (list, optional): Not used for contract details but kept for compatibility
            start_date (str, optional): Not used for contract details but kept for compatibility
            end_date (str, optional): Not used for contract details but kept for compatibility
        """
        super().__init__(symbol, intervals, start_date, end_date)
        self.contract_type = contract_type.lower()

    def fetch_for_interval(self, interval):
        """
        Not applicable for contract details.

        Args:
            interval (str): Not used for contract details

        Returns:
            None: This method is not used for contract details
        """
        self.logger.warning(
            "Interval-based fetching not applicable for contract details."
        )
        return None

    def fetch_all(self):
        """
        Fetch contract details based on the specified type.

        Returns:
            dict: Dictionary containing the fetched contract data
        """
        self.logger.info(
            f"Starting contract details fetch for type: {self.contract_type}"
        )

        results = {}

        if self.contract_type in ["futures", "both"]:
            futures_specs = self._fetch_futures_contracts_info()
            results["futures"] = futures_specs

        if self.contract_type in ["spot", "both"]:
            spot_specs = self._fetch_spot_trading_pairs()
            results["spot"] = spot_specs

        self.logger.info("Completed contract details fetch")
        return results

    def _fetch_futures_contracts_info(self):
        """
        Fetch and save specifications for all futures contracts from Binance.
        Includes both delivery and perpetual futures.

        Returns:
            pandas.DataFrame: DataFrame with futures contracts information
        """
        try:
            # Fetch exchange info using Binance client
            self.logger.info("Fetching futures contracts information")
            data = self.client.futures_exchange_info()
            contracts_info = []

            for symbol_info in data.get("symbols", []):
                # Store the full contract info for all contracts, including perpetuals
                contracts_info.append(symbol_info)

            # Convert to DataFrame and save to CSV
            df = pd.DataFrame(contracts_info)

            # Ensure directory exists
            fut_specs_path = Path(config.contracts_dir) / "fut_specs.csv"

            # Save specifications
            save_path = self._save_data(df, fut_specs_path, "N/A")
            self.logger.info(
                f"Successfully saved {len(df)} futures contracts to {save_path}"
            )

            # Update fut_expirys.csv with quarterly contracts
            self._update_fut_expirys(df)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching futures contracts information: {e}")
            raise FetcherError(f"Failed to fetch futures contracts info: {e}")

    def _fetch_spot_trading_pairs(self):
        """
        Fetch and save specifications for all spot trading pairs from Binance.

        Returns:
            pandas.DataFrame: DataFrame with spot trading pairs information
        """
        try:
            # Fetch exchange information
            self.logger.info("Fetching spot trading pairs information")
            exchange_info = self.client.get_exchange_info()

            # Extract all trading pairs
            all_pairs = exchange_info["symbols"]

            # Store selected keys in CSV
            df = pd.DataFrame(
                [
                    {
                        "symbol": s["symbol"],
                        "status": s["status"],
                        "baseAsset": s["baseAsset"],
                        "baseAssetPrecision": s["baseAssetPrecision"],
                        "quoteAsset": s["quoteAsset"],
                        "quotePrecision": s["quotePrecision"],
                        "quoteAssetPrecision": s["quoteAssetPrecision"],
                        "baseCommissionPrecision": s["baseCommissionPrecision"],
                        "quoteCommissionPrecision": s["quoteCommissionPrecision"],
                        "orderTypes": ", ".join(s["orderTypes"]),
                        "icebergAllowed": s["icebergAllowed"],
                        "ocoAllowed": s["ocoAllowed"],
                        "quoteOrderQtyMarketAllowed": s["quoteOrderQtyMarketAllowed"],
                        "allowTrailingStop": s["allowTrailingStop"],
                        "cancelReplaceAllowed": s["cancelReplaceAllowed"],
                        "isSpotTradingAllowed": s["isSpotTradingAllowed"],
                        "isMarginTradingAllowed": s["isMarginTradingAllowed"],
                    }
                    for s in all_pairs
                ]
            )

            # Save to CSV
            spot_specs_path = Path(config.contracts_dir) / "spot_specs.csv"
            save_path = self._save_data(df, spot_specs_path, "N/A")
            self.logger.info(
                f"Successfully saved {len(df)} spot trading pairs to {save_path}"
            )

            return df

        except Exception as e:
            self.logger.error(f"Error fetching spot trading pairs information: {e}")
            raise FetcherError(f"Failed to fetch spot trading pairs info: {e}")

    def _update_fut_expirys(self, new_contracts_df):
        """
        Update fut_expirys.csv with new quarterly contracts.

        Args:
            new_contracts_df (pd.DataFrame): DataFrame containing the newly fetched futures contracts
        """
        try:
            expiry_file_path = Path(config.contracts_dir) / "fut_expirys.csv"

            # Filter for quarterly contracts
            quarterly_mask = new_contracts_df["contractType"].isin(
                ["CURRENT_QUARTER", "NEXT_QUARTER"]
            )
            quarterly_contracts = new_contracts_df[quarterly_mask].copy()

            if quarterly_contracts.empty:
                self.logger.info(
                    "No quarterly contracts found to add to fut_expirys.csv"
                )
                return

            self.logger.info(f"Found {len(quarterly_contracts)} quarterly contracts")

            # Check if fut_expirys.csv exists
            if not expiry_file_path.exists():
                self.logger.warning(
                    f"fut_expirys.csv not found at {expiry_file_path}, will create new file"
                )
                self._save_data(quarterly_contracts, expiry_file_path, "N/A")
                self.logger.info(
                    f"Created fut_expirys.csv with {len(quarterly_contracts)} quarterly contracts"
                )
                return

            # Read existing fut_expirys.csv
            existing_df = pd.read_csv(expiry_file_path)

            # Get symbol column names - handle potential differences in column names
            existing_symbol_col = "symbol" if "symbol" in existing_df.columns else None
            if existing_symbol_col is None:
                self.logger.error("Could not find symbol column in fut_expirys.csv")
                return

            # Find contracts to add (not already in fut_expirys.csv)
            existing_symbols = set(existing_df[existing_symbol_col].values)

            # Create a list to store new contracts
            new_contracts = []

            for _, row in quarterly_contracts.iterrows():
                if row["symbol"] not in existing_symbols:
                    new_contracts.append(row)

            # If there are new contracts, append them to fut_expirys.csv
            if new_contracts:
                new_contracts_df = pd.DataFrame(new_contracts)

                # Handle the case where column names might differ
                if set(new_contracts_df.columns) != set(existing_df.columns):
                    self.logger.warning(
                        "Column mismatch between new contracts and existing fut_expirys.csv"
                    )
                    # Only keep columns that exist in both DataFrames
                    common_columns = list(
                        set(new_contracts_df.columns).intersection(
                            set(existing_df.columns)
                        )
                    )
                    new_contracts_df = new_contracts_df[common_columns]

                # Append the new contracts to the existing CSV, line by line
                with open(expiry_file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for _, row in new_contracts_df.iterrows():
                        writer.writerow(row.tolist())
                self.logger.info(
                    f"Added {len(new_contracts)} new quarterly contracts to fut_expirys.csv"
                )
            else:
                self.logger.info("No new quarterly contracts to add to fut_expirys.csv")

        except Exception as e:
            self.logger.error(f"Error updating fut_expirys.csv: {e}")
            raise FetcherError(f"Failed to update futures expiry data: {e}")
