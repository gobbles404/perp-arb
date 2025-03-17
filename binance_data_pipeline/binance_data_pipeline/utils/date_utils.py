# binance_data_pipeline/utils/date_utils.py
from datetime import datetime, timezone, timedelta
import re
import pandas as pd

from ..core.logger import get_logger

logger = get_logger(__name__)


def date_to_timestamp(start_date, end_date):
    """Convert date strings (YYYY-MM-DD) to Binance timestamps in UTC milliseconds."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    start_ts = int(start_dt.timestamp()) * 1000
    end_ts = int(end_dt.timestamp()) * 1000
    return start_ts, end_ts


def get_last_expected_timestamp(interval):
    """Compute the last expected timestamp for any Binance interval based on the current time."""
    now = datetime.now(timezone.utc)

    # Parse the interval string with regex
    match = re.match(r"(\d+)([mhdwM])", interval)
    if not match:
        raise ValueError(f"Unsupported interval: {interval}")

    value, unit = int(match.group(1)), match.group(2)

    # Calculate last expected timestamp based on interval type
    last_expected_ts = None

    # Handle standard time intervals (minutes, hours, days)
    if unit in ["m", "h", "d"]:
        # Convert unit to seconds
        unit_to_seconds = {"m": 60, "h": 3600, "d": 86400}
        interval_seconds = value * unit_to_seconds[unit]
        last_expected_ts = (
            int(now.timestamp()) // interval_seconds * interval_seconds
        ) * 1000

    # Handle weekly intervals (starting on Mondays)
    elif unit == "w":
        # Find the most recent Monday
        days_since_monday = now.weekday()  # Monday=0, Sunday=6
        last_monday = now - timedelta(days=days_since_monday)
        monday_midnight = datetime(
            last_monday.year, last_monday.month, last_monday.day, tzinfo=timezone.utc
        )
        last_expected_ts = int(monday_midnight.timestamp()) * 1000

    # Handle monthly intervals (starting on day 1)
    elif unit == "M":
        first_of_month = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        last_expected_ts = int(first_of_month.timestamp()) * 1000

    else:
        raise ValueError(f"Unsupported unit: {unit}")

    logger.info(
        f"Last expected timestamp for {interval}: {pd.to_datetime(last_expected_ts, unit='ms')}"
    )
    return last_expected_ts


def extract_base_symbol(full_symbol):
    """Extract base symbol (e.g., 'BTCUSDT' from 'BTCUSDT_230929')."""
    if "_" in full_symbol:
        return full_symbol.split("_")[0]
    return full_symbol
