import os
from datetime import datetime, timedelta

# Global date settings
DEFAULT_START_DATE = "2023-06-01"  # contract futures data starts in June of 2023
DEFAULT_END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

# Default intervals for different data types
DEFAULT_INTERVALS = {
    "spot": ["1d", "8h", "1h"],
    "perpetuals": ["1d", "8h", "1h"],
    "futures": ["1d", "8h", "1h"],
    "funding": ["8h"],
    "premium": ["1d", "8h", "1h"],
}

# Default symbol
DEFAULT_SYMBOL = "BTCUSDT"

# Directory Structure
CSV_DIR = "data"
RAW_DIR = os.path.join(CSV_DIR, "raw")
PROCESSED_DIR = os.path.join(CSV_DIR, "processed")
MARKETS_DIR = os.path.join(CSV_DIR, "markets")
CONTRACTS_DIR = os.path.join(CSV_DIR, "contracts")


# Ensure all directories exist
def ensure_directories():
    """Create all required directories if they don't exist."""
    for directory in [CSV_DIR, RAW_DIR, PROCESSED_DIR, MARKETS_DIR, CONTRACTS_DIR]:
        os.makedirs(directory, exist_ok=True)


# Create directories when module is imported
ensure_directories()

# Other global settings
LOG_LEVEL = "INFO"  # Consider removing this once logging_config is fully implemented
