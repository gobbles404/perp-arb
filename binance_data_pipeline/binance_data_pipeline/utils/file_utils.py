# binance_data_pipeline/utils/file_utils.py
import os
import pandas as pd
from pathlib import Path

from ..core.logger import get_logger

logger = get_logger(__name__)


def save_to_csv(df, filepath, symbol=None, interval=None):
    """Save DataFrame to CSV file with directory creation."""
    # Convert string path to Path object if needed
    filepath = Path(filepath)

    # Create directory structure if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save the DataFrame
    df.to_csv(filepath, index=False)

    # Log success
    if symbol and interval:
        logger.info(f"Saved {len(df)} rows for {symbol} ({interval}) to {filepath}")
    else:
        logger.info(f"Saved {len(df)} rows to {filepath}")

    return filepath
