"""
Configuration settings for the trading strategy.
"""

import os

# Default paths
DATA_DIR = os.path.join("data", "markets")
RESULTS_DIR = os.path.join("strategy", "results")

# Default strategy parameters
DEFAULT_CAPITAL = 10000.0
DEFAULT_LEVERAGE = 1.0
DEFAULT_FEE_RATE = 0.0004  # 4 basis points

# Data parameters
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "8h"

# Visualization settings
CHART_WIDTH = 14
CHART_HEIGHT = 24
DATE_FORMAT = "%Y-%m-%d"

# File naming
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"

# Market assumptions
FUNDING_PERIODS_PER_DAY = 3  # 8-hour funding periods
TRADING_DAYS_PER_YEAR = 365
