"""
Functions for loading and preprocessing market data.
"""

import pandas as pd
import numpy as np


def load_data(file_path):
    """Load market data from CSV file."""
    print(f"Loading data from {file_path}")

    try:
        data = pd.read_csv(file_path)

        # Ensure Timestamp column is in datetime format
        if "Timestamp" in data.columns:
            data["Timestamp"] = pd.to_datetime(data["Timestamp"])

        # Clean funding rate data - replace NaN with 0 or interpolate
        if "funding_rate" in data.columns:
            # First replace extreme values that might be errors
            funding_std = data["funding_rate"].std()
            funding_mean = data["funding_rate"].mean()

            # Replace extreme outliers (> 5 std from mean) with NaN for interpolation
            data.loc[
                abs(data["funding_rate"] - funding_mean) > 5 * funding_std,
                "funding_rate",
            ] = np.nan

            # Interpolate NaN values in funding_rate
            data["funding_rate"] = data["funding_rate"].interpolate(method="linear")

            # Fill any remaining NaN at the ends of the series
            data["funding_rate"] = (
                data["funding_rate"].fillna(method="ffill").fillna(method="bfill")
            )

            # If still NaN, replace with 0
            data["funding_rate"] = data["funding_rate"].fillna(0)

        print(
            f"Loaded {len(data)} rows of data from {data['Timestamp'].min()} to {data['Timestamp'].max()}"
        )
        return data

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def load_futures_contracts(file_path="data/contracts/fut_specs.csv"):
    """Load futures contract data with contract information."""
    print(f"Loading futures contracts from {file_path}")

    try:
        contracts_df = pd.read_csv(file_path)

        # Convert timestamps to datetime
        contracts_df["deliveryDate"] = pd.to_datetime(
            contracts_df["deliveryDate"], unit="ms"
        )
        contracts_df["onboardDate"] = pd.to_datetime(
            contracts_df["onboardDate"], unit="ms"
        )

        # Calculate max leverage based on margin requirements
        contracts_df["maxLeverage"] = 100 / contracts_df["requiredMarginPercent"]

        print(f"Loaded {len(contracts_df)} futures contracts")
        return contracts_df

    except Exception as e:
        print(f"Error loading futures contracts: {e}")
        return None


def filter_date_range(data, start_date=None, end_date=None):
    """Filter data by date range."""
    df = data.copy()

    if start_date is not None:
        df = df[df["Timestamp"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["Timestamp"] <= pd.to_datetime(end_date)]

    # Reset index after filtering
    df = df.reset_index(drop=True)

    if len(df) < 2:
        print("Error: Insufficient data points after filtering")
        return None

    return df


def calculate_metrics(data):
    """Calculate additional market metrics needed for analysis."""
    df = data.copy()

    # Calculate basis percentage - handle potential zeros in spot_close
    df["basis_pct"] = [
        (p / s - 1) * 100 if s > 0 else 0
        for p, s in zip(df["perp_close"], df["spot_close"])
    ]

    # Calculate annualized funding rate
    df["funding_apr"] = df["funding_rate"] * 3 * 365 * 100  # 3 funding periods per day

    # Make sure basis_pct and funding_apr don't have NaN
    df["basis_pct"] = df["basis_pct"].fillna(0)
    df["funding_apr"] = df["funding_apr"].fillna(0)

    return df


# todo: join contract data with the price data.
def enrich_market_data(data):

    pass
