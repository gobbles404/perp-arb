import os
from datetime import datetime

# Global log settings
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log file settings
LOGS_DIR = "logs"
LOG_FILE_PREFIX = "perpetual_arbitrage"
LOG_FILE_SUFFIX = datetime.now().strftime("%Y%m%d")
LOG_FILE = f"{LOG_FILE_PREFIX}_{LOG_FILE_SUFFIX}.log"

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Module-specific log levels (optional)
MODULE_LOG_LEVELS = {
    "perpetual_arbitrage.fetchers": "INFO",
    "perpetual_arbitrage.process": "INFO",
    # Add other modules as needed
}

# Log file path
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

# Maximum log file size for rotation (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Keep up to 30 backup log files
BACKUP_COUNT = 30
