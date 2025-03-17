# perp_arb/backtesting/utils/helpers.py
import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime

# Import the logger
from core.logger import get_logger

# Create a logger instance for helpers
logger = get_logger(__name__)


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path: Path to check/create
    """
    if not os.path.exists(directory_path):
        logger.info(f"Creating directory: {directory_path}")
        os.makedirs(directory_path)


def get_market_data_path(
    symbol: str, interval: str = "1d", base_dir: str = None
) -> str:
    """
    Get the file path for market data based on symbol and interval.

    Args:
        symbol: Trading symbol
        interval: Data interval (e.g., '1d', '1h')
        base_dir: Base directory for market data (default: ../binance_data_pipeline/data/markets)

    Returns:
        Full path to market data file
    """
    if base_dir is None:
        base_dir = os.path.join("..", "binance_data_pipeline", "data", "markets")

    file_path = os.path.join(base_dir, f"{symbol}_{interval}.csv")
    logger.debug(f"Market data path for {symbol}_{interval}: {file_path}")
    return file_path


def get_contract_specs_path(base_dir: str = None) -> str:
    """
    Get the file path for contract specifications.

    Args:
        base_dir: Base directory for contract data (default: ../binance_data_pipeline/data/contracts)

    Returns:
        Full path to contract specifications file
    """
    if base_dir is None:
        base_dir = os.path.join("..", "binance_data_pipeline", "data", "contracts")

    file_path = os.path.join(base_dir, "fut_specs.csv")
    logger.debug(f"Contract specs path: {file_path}")
    return file_path


def calculate_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Z-score for a time series using rolling window.

    Args:
        series: Time series data
        window: Rolling window size

    Returns:
        Series of Z-scores
    """
    logger.debug(f"Calculating Z-score with window size {window}")
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    # Calculate Z-score
    z_score = (series - rolling_mean) / rolling_std

    return z_score


def calculate_funding_payment(
    position_size: float, funding_rate: float, mark_price: float
) -> float:
    """
    Calculate the funding payment for a perpetual position.

    Args:
        position_size: Size of the position (positive for long, negative for short)
        funding_rate: Current funding rate
        mark_price: Current mark price

    Returns:
        Funding payment (positive means payment received, negative means payment made)
    """
    # For longs: negative funding rate means payment received
    # For shorts: negative funding rate means payment made
    # Payment = position_size * mark_price * funding_rate
    payment = -position_size * mark_price * funding_rate
    logger.debug(
        f"Funding payment calculated: {payment} (position: {position_size}, rate: {funding_rate}, price: {mark_price})"
    )
    return payment


def save_results_to_csv(
    data: pd.DataFrame, filename: str, output_dir: str = "results"
) -> str:
    """
    Save results DataFrame to CSV file.

    Args:
        data: DataFrame to save
        filename: Output filename (without extension)
        output_dir: Output directory

    Returns:
        Full path to saved file
    """
    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # Append .csv extension if not present
    if not filename.endswith(".csv"):
        filename += ".csv"

    # Full file path
    file_path = os.path.join(output_dir, filename)

    # Save to CSV
    data.to_csv(file_path, index=True)
    logger.info(f"Results saved to CSV: {file_path}")

    return file_path


def save_results_to_json(
    data: Dict[str, Any], filename: str, output_dir: str = "results"
) -> str:
    """
    Save results dictionary to JSON file.

    Args:
        data: Dictionary to save
        filename: Output filename (without extension)
        output_dir: Output directory

    Returns:
        Full path to saved file
    """
    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # Append .json extension if not present
    if not filename.endswith(".json"):
        filename += ".json"

    # Full file path
    file_path = os.path.join(output_dir, filename)

    # Handle datetime objects
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)

    # Save to JSON
    with open(file_path, "w") as f:
        json.dump(data, f, cls=DateTimeEncoder, indent=4)

    logger.info(f"Results saved to JSON: {file_path}")
    return file_path


def get_available_symbols(market_data_dir: str) -> List[str]:
    """
    Get list of available trading symbols from market data files.

    Args:
        market_data_dir: Directory containing market data files

    Returns:
        List of symbol strings
    """
    symbols = []

    if os.path.exists(market_data_dir):
        for filename in os.listdir(market_data_dir):
            if filename.endswith(".csv"):
                # Extract symbol from filename (assuming format: {SYMBOL}_{INTERVAL}.csv)
                if "_" in filename:
                    symbol = filename.split("_")[0]
                    symbols.append(symbol)

    unique_symbols = list(set(symbols))  # Remove duplicates
    logger.debug(f"Found {len(unique_symbols)} unique symbols in {market_data_dir}")
    return unique_symbols


def match_contracts_to_market_data(
    market_data_symbols: List[str], contract_specs: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """
    Match contract specifications to market data symbols.

    Args:
        market_data_symbols: List of symbols from market data
        contract_specs: DataFrame with contract specifications

    Returns:
        Dictionary mapping market data symbols to contract specification dictionaries
    """
    matched_contracts = {}

    for symbol in market_data_symbols:
        # Look for exact match in contract specs
        contract_match = contract_specs[contract_specs["symbol"] == symbol]

        if not contract_match.empty:
            # Convert first matching row to dictionary
            matched_contracts[symbol] = contract_match.iloc[0].to_dict()
            logger.debug(f"Found exact contract match for {symbol}")
        else:
            # Check if symbol is contained within any contract symbols
            for _, contract in contract_specs.iterrows():
                if symbol in contract["symbol"]:
                    matched_contracts[symbol] = contract.to_dict()
                    logger.debug(f"Found partial contract match for {symbol}")
                    break

            if symbol not in matched_contracts:
                logger.warning(f"No contract specifications found for {symbol}")

    logger.info(
        f"Matched {len(matched_contracts)}/{len(market_data_symbols)} symbols to contract specifications"
    )
    return matched_contracts
