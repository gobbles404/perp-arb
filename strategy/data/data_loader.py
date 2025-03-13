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

        print(
            f"Loaded {len(data)} rows of data from {data['Timestamp'].min()} to {data['Timestamp'].max()}"
        )
        return data

    except Exception as e:
        print(f"Error loading data: {e}")
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

    # Calculate basis percentage
    df["basis_pct"] = [
        (p / s - 1) * 100 for p, s in zip(df["perp_close"], df["spot_close"])
    ]

    # Calculate annualized funding rate
    df["funding_apr"] = df["funding_rate"] * 3 * 365 * 100  # 3 funding periods per day

    return df
