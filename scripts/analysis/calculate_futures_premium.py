#!/usr/bin/env python3
"""
Futures Premium Calculator

This script calculates the premium/discount of futures contracts relative to spot prices
and the implied interest rate (APR). It takes spot and futures price data as input
and outputs a CSV file with the calculations.

Usage:
    python calculate_futures_premium.py --spot spot.csv --futures futures_index_1d.csv --output results.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime


def calculate_futures_premium(spot_file, futures_file, output_file=None):
    """
    Calculate futures premium and implied interest rates from spot and futures data.

    Parameters:
    -----------
    spot_file : str
        Path to the CSV file containing spot price data
    futures_file : str
        Path to the CSV file containing futures price data
    output_file : str, optional
        Path where to save the results CSV. If None, will use a default name.

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all calculations
    """
    print(f"Loading spot data from: {spot_file}")
    print(f"Loading futures data from: {futures_file}")

    # Load the data
    spot_df = pd.read_csv(spot_file)
    futures_df = pd.read_csv(futures_file)

    # Make sure Timestamp is in datetime format
    spot_df["Timestamp"] = pd.to_datetime(spot_df["Timestamp"])
    futures_df["Timestamp"] = pd.to_datetime(futures_df["Timestamp"])

    # Print basic info about the data
    print(
        f"Loaded {len(spot_df)} spot price records from {spot_df['Timestamp'].min()} to {spot_df['Timestamp'].max()}"
    )
    print(
        f"Loaded {len(futures_df)} futures price records from {futures_df['Timestamp'].min()} to {futures_df['Timestamp'].max()}"
    )

    # Check for required columns
    required_spot_cols = ["Timestamp", "Close"]
    required_futures_cols = ["Timestamp", "Close", "Contract"]

    missing_spot_cols = [
        col for col in required_spot_cols if col not in spot_df.columns
    ]
    missing_futures_cols = [
        col for col in required_futures_cols if col not in futures_df.columns
    ]

    if missing_spot_cols:
        raise ValueError(f"Spot data is missing required columns: {missing_spot_cols}")
    if missing_futures_cols:
        if (
            "Contract" in missing_futures_cols
            and "days_till_expiry" not in futures_df.columns
        ):
            print("Warning: 'Contract' column not found in futures data.")

            # Try to extract contract info from the filename if possible
            try:
                contract_name = (
                    os.path.basename(futures_file).split("_")[2].split(".")[0]
                )
                print(f"Using contract name from filename: {contract_name}")
                futures_df["Contract"] = contract_name
            except:
                print("Could not extract contract name from filename. Using 'UNKNOWN'")
                futures_df["Contract"] = "UNKNOWN"

    # Check if days_till_expiry exists in futures data
    if "days_till_expiry" not in futures_df.columns:
        print(
            "Warning: 'days_till_expiry' column not found. Trying to calculate from contract name..."
        )

        # Try to extract expiry date from contract name (assuming format like 'BTCUSDT_241227')
        try:
            # Extract dates from contract names (assuming format like asset_YYMMDD)
            def extract_expiry(contract):
                parts = contract.split("_")
                if len(parts) >= 2:
                    date_part = parts[-1]
                    if len(date_part) >= 6 and date_part[:6].isdigit():
                        year = int("20" + date_part[:2])
                        month = int(date_part[2:4])
                        day = int(date_part[4:6])
                        return datetime(year, month, day)
                # If we can't parse, use a default far future date
                print(f"Could not parse expiry date from contract: {contract}")
                return datetime(2099, 12, 31)

            futures_df["expiry_date"] = futures_df["Contract"].apply(extract_expiry)
            futures_df["days_till_expiry"] = (
                futures_df["expiry_date"] - futures_df["Timestamp"]
            ).dt.days
            print("Successfully calculated days_till_expiry from contract names")
        except Exception as e:
            print(f"Error calculating expiry: {e}")
            print("Using placeholder values for days_till_expiry")
            futures_df["days_till_expiry"] = (
                90  # Assuming a standard quarterly contract
            )

    # Merge spot and futures data on Timestamp
    print("Merging spot and futures data...")
    merged_df = pd.merge(
        futures_df,
        spot_df[["Timestamp", "Close"]],
        on="Timestamp",
        how="inner",
        suffixes=("_futures", "_spot"),
    )

    print(f"Merged data contains {len(merged_df)} records")

    # Calculate premium and implied rate
    print("Calculating premium and implied rate...")

    # Basic premium calculation: (futures - spot) / spot
    merged_df["premium_pct"] = (
        100
        * (merged_df["Close_futures"] - merged_df["Close_spot"])
        / merged_df["Close_spot"]
    )

    # Convert days to expiry to years for annualized rate calculation
    merged_df["time_to_expiry_years"] = merged_df["days_till_expiry"] / 365

    # Calculate implied rate using logarithmic formula: r = ln(F/S) / T
    merged_df["implied_rate_pct"] = (
        100
        * np.log(merged_df["Close_futures"] / merged_df["Close_spot"])
        / merged_df["time_to_expiry_years"]
    )

    # Clean up the columns and rename for clarity
    result_df = merged_df[
        [
            "Timestamp",
            "Contract",
            "Close_spot",
            "Close_futures",
            "days_till_expiry",
            "premium_pct",
            "implied_rate_pct",
        ]
    ].rename(columns={"Close_spot": "spot_price", "Close_futures": "futures_price"})

    # Calculate basic statistics
    avg_premium = result_df["premium_pct"].mean()
    avg_rate = result_df["implied_rate_pct"].mean()

    print(f"Average Premium: {avg_premium:.2f}%")
    print(f"Average Implied Rate (APR): {avg_rate:.2f}%")

    # Save to CSV if output file is specified
    if output_file:
        print(f"Saving results to: {output_file}")
        result_df.to_csv(output_file, index=False)
        print(f"Results saved successfully")

    return result_df


def main():
    """Parse command line arguments and run the calculation."""
    parser = argparse.ArgumentParser(
        description="Calculate futures premium and implied interest rates."
    )
    parser.add_argument("--spot", required=True, help="Path to spot price CSV file")
    parser.add_argument(
        "--futures", required=True, help="Path to futures price CSV file"
    )
    parser.add_argument("--output", default=None, help="Path to save results CSV file")

    args = parser.parse_args()

    # If output file is not specified, create a default name
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"futures_premium_analysis_{timestamp}.csv"

    # Run the calculation
    calculate_futures_premium(args.spot, args.futures, args.output)


if __name__ == "__main__":
    main()
