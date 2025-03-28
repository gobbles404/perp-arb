# binance_data_pipeline/core/logger.py
import logging
import os
from pathlib import Path


def get_logger(name, level=None):
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        log_level = level or os.environ.get("LOG_LEVEL", "INFO")
        logger.setLevel(getattr(logging, log_level))

        # Add console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
