import logging
import logging.handlers
from .config import (
    LOG_LEVEL,
    LOG_FORMAT,
    DATE_FORMAT,
    LOG_FILE_PATH,
    MODULE_LOG_LEVELS,
    MAX_LOG_SIZE,
    BACKUP_COUNT,
)

# Root logger name for your application
ROOT_LOGGER_NAME = "perpetual_arbitrage"

# Configure root logger only once
_is_configured = False


def _configure_root_logger():
    """Configure the root logger for the application."""
    global _is_configured

    if _is_configured:
        return

    # Create root logger
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    root_logger.setLevel(getattr(logging, LOG_LEVEL))

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Create file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE_PATH, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Prevent propagation to Python's root logger
    root_logger.propagate = False

    _is_configured = True
    root_logger.info(f"Logging system initialized. Log file: {LOG_FILE_PATH}")


def get_logger(module_name):
    """
    Get a logger with the appropriate name and level.

    Args:
        module_name (str): Usually passed as __name__ from the calling module

    Returns:
        logging.Logger: Configured logger instance
    """
    # Configure root logger if not done yet
    if not _is_configured:
        _configure_root_logger()

    # If the module_name doesn't contain our root logger name, add it as prefix
    if not module_name.startswith(ROOT_LOGGER_NAME) and module_name != "__main__":
        # Convert modules like 'scripts.fetchers.base' to 'perpetual_arbitrage.fetchers.base'
        parts = module_name.split(".")
        if parts[0] in ["scripts", "strategy"]:
            # Replace the first component with our root logger name
            parts[0] = ROOT_LOGGER_NAME
            module_name = ".".join(parts)
        else:
            # Otherwise, just prefix with root logger name
            module_name = f"{ROOT_LOGGER_NAME}.{module_name}"

    # Get the logger
    logger = logging.getLogger(module_name)

    # Set module-specific log level if defined
    if module_name in MODULE_LOG_LEVELS:
        logger.setLevel(getattr(logging, MODULE_LOG_LEVELS[module_name]))

    return logger
