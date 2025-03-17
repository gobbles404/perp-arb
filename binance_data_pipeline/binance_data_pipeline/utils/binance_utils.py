# binance_data_pipeline/utils/binance_utils.py
import pandas as pd
from ..core.logger import get_logger
from ..core.client import client

logger = get_logger(__name__)


def fetch_klines(symbol, interval, start_ts, end_ts, contract_type="Futures"):
    """Unified function to fetch klines data with pagination."""
    all_data = []
    current_ts = start_ts
    last_valid_ts = None
    limit = 1000

    while current_ts < end_ts:
        logger.info(f"Fetching {symbol} from {pd.to_datetime(current_ts, unit='ms')}")

        try:
            if contract_type == "Spot":
                klines = client.get_klines(
                    symbol=symbol, interval=interval, startTime=current_ts, limit=limit
                )
            else:
                klines = client.futures_klines(
                    symbol=symbol, interval=interval, startTime=current_ts, limit=limit
                )

            # Process klines and handle pagination
            # (Code from your existing utils.py fetch_klines function)

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {str(e)}")
            return None

    # Convert and return the DataFrame
    if all_data:
        df = pd.DataFrame(
            all_data, columns=["Timestamp", "Open", "High", "Low", "Close"]
        )
        return df
    else:
        logger.warning(f"No data collected for {symbol}.")
        return None
