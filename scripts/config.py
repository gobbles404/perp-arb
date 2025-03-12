# scripts/config.py
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

# Other global settings
LOG_LEVEL = "INFO"
