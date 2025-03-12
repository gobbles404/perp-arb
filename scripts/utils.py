import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import re

# Create base directory structure
CSV_DIR = "data"
RAW_DIR = os.path.join(CSV_DIR, "raw")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


def get_last_expected_timestamp(interval):
    """Compute the last expected timestamp for any Binance interval based on the current time."""
    now = datetime.now(timezone.utc)

    # Parse the interval string with regex
    match = re.match(r"(\d+)([mhdwM])", interval)
    if not match:
        raise ValueError(f"Unsupported interval: {interval}")

    value, unit = int(match.group(1)), match.group(2)

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

    print(
        f"🕒 Last expected timestamp for {interval}: {pd.to_datetime(last_expected_ts, unit='ms')}"
    )
    return last_expected_ts


def extract_base_symbol(full_symbol):
    """
    Extract the base symbol (e.g., 'BTCUSDT' from 'BTCUSDT_230929').
    Trim everything starting with the '_'.
    """
    if "_" in full_symbol:
        return full_symbol.split("_")[0]
    return full_symbol  # Return as is if '_' not found


def date_to_timestamp(start_date, end_date):
    """Convert date strings (YYYY-MM-DD) to Binance timestamps in UTC milliseconds."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    start_ts = int(start_dt.timestamp()) * 1000
    end_ts = int(end_dt.timestamp()) * 1000
    return start_ts, end_ts


def save_to_csv(df, filepath, symbol=None, interval=None):
    """
    Save DataFrame to CSV file with structured directory path.

    Args:
        df (pandas.DataFrame): DataFrame to save
        filepath (str): Full relative path where the file should be saved
        symbol (str, optional): Trading pair symbol (for logging)
        interval (str, optional): Time interval (for logging)

    Returns:
        str: Full path where the file was saved
    """
    import os
    from pathlib import Path

    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct the full path
    full_path = os.path.join(project_root, filepath)

    # Create directory structure if it doesn't exist
    Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True)

    # Save the data
    df.to_csv(full_path, index=False)

    return full_path  # Return the full path for reference


def fetch_klines(client, symbol, interval, start_ts, end_ts, contract_type="Futures"):
    """Fetches historical kline data (candlestick) for any symbol and contract type with auto-pagination."""
    all_data = []
    current_ts = start_ts
    last_valid_ts = None

    while current_ts < end_ts:
        print(f"🔄 Fetching {symbol} from {pd.to_datetime(current_ts, unit='ms')}")

        try:
            if contract_type == "Spot":
                klines = client.get_klines(
                    symbol=symbol, interval=interval, startTime=current_ts, limit=1000
                )
            else:
                klines = client.futures_klines(
                    symbol=symbol, interval=interval, startTime=current_ts, limit=1000
                )

            if not klines:
                if last_valid_ts:
                    print(
                        f"✅ {symbol} contract likely expired at {pd.to_datetime(last_valid_ts, unit='ms')}"
                    )
                    break  # Stop fetching since the contract has expired
                else:
                    print(
                        f"⚠️ No data returned for {symbol} at {pd.to_datetime(current_ts, unit='ms')}"
                    )
                    return None

            df = pd.DataFrame(
                klines,
                columns=[
                    "Timestamp",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Close Time",
                    "Quote Asset Volume",
                    "Trades",
                    "Taker Buy Base",
                    "Taker Buy Quote",
                    "Ignore",
                ],
            )
            df = df[["Timestamp", "Open", "High", "Low", "Close"]]
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
            df["Close"] = df["Close"].astype(float)

            all_data.extend(df.values.tolist())  # Convert to list for performance
            last_valid_ts = int(df.iloc[-1, 0].timestamp() * 1000)

            current_ts = last_valid_ts + 1

            print(
                f"✅ Fetched {len(df)} rows. Next start: {pd.to_datetime(current_ts, unit='ms')}"
            )

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {str(e)}")
            return None

    if not all_data:
        print(f"⚠️ No data collected for {symbol}.")
        return None

    # Convert list back into DataFrame
    df = pd.DataFrame(all_data, columns=["Timestamp", "Open", "High", "Low", "Close"])

    return df
