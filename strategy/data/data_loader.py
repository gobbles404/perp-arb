"""
Functions for loading and preprocessing market data.
"""

import pandas as pd
import os


def load_data(file_path, exclude_cols=True):
    """
    Load market data from CSV file.

    Args:
        file_path: Path to the CSV file
        exclude_cols: Whether to exclude unwanted columns

    Returns:
        Processed DataFrame
    """
    print(f"Loading data from {file_path}")

    try:
        data = pd.read_csv(file_path)

        # Ensure Timestamp column is in datetime format
        if "Timestamp" in data.columns:
            data["Timestamp"] = pd.to_datetime(data["Timestamp"])

        # Clean funding rate data - replace NaN with 0 or interpolate
        if "funding_rate" in data.columns:
            # Fill any remaining NaN at the ends of the series
            data["funding_rate"] = (
                data["funding_rate"].fillna(method="ffill").fillna(method="bfill")
            )

            # If still NaN, replace with 0
            data["funding_rate"] = data["funding_rate"].fillna(0)

        # Enrich data with contract specifications
        data = enrich_market_data(data, file_path)

        # Exclude unwanted columns if requested
        if exclude_cols:
            data = exclude_columns(data)

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


def extract_contract_data(file_path):
    """
    Extract contract data for a specific symbol from futures contracts.

    Args:
        file_path: Path to the CSV file with market data

    Returns:
        Dictionary of contract data indexed by contract type
    """
    # Load futures contracts
    contracts_df = load_futures_contracts("data/contracts/fut_specs.csv")

    # Extract symbol from file path
    file_name = os.path.basename(file_path)
    symbol_base = file_name.split("_")[0]  # e.g., "BTCUSDT" from "BTCUSDT_1d.csv"

    # Filter contracts by pair (which should match the base symbol)
    matching_contracts = contracts_df[contracts_df["pair"] == symbol_base]

    # Print the number of matching contracts
    print(f"Found {len(matching_contracts)} matching contracts for pair {symbol_base}")

    # Create a dictionary to store the relevant fields for each contract type
    contract_data = {}

    for _, contract in matching_contracts.iterrows():
        symbol = contract["symbol"]
        contract_type = contract["contractType"]

        # Extract relevant fields
        contract_data[contract_type] = {
            "symbol": symbol,
            "maintMarginPercent": contract["maintMarginPercent"],
            "requiredMarginPercent": contract["requiredMarginPercent"],
            "liquidationFee": contract["liquidationFee"],
        }

    # Print contract data for review
    print("\nContract Data:")
    for contract_type, contract_info in contract_data.items():
        print(f"  {contract_type}:")
        for field, value in contract_info.items():
            print(f"    {field}: {value}")

    return contract_data


def exclude_columns(data, columns_to_exclude=None):
    """
    Filter DataFrame to exclude specified columns.

    Args:
        data: DataFrame with all columns
        columns_to_exclude: List of column names to exclude. If None, use default exclusion list.

    Returns:
        DataFrame with unwanted columns removed
    """
    # Default list of columns to exclude if none provided
    if columns_to_exclude is None:
        # By default, exclude these columns (open/high/low prices that aren't needed)
        columns_to_exclude = [
            "spot_open",
            "spot_high",
            "spot_low",
            "perp_open",
            "perp_high",
            "perp_low",
            "prompt_open",
            "prompt_high",
            "prompt_low",
            "next_open",
            "next_high",
            "next_low",
        ]

    # Check which columns actually exist in the DataFrame to exclude
    existing_exclusions = [col for col in columns_to_exclude if col in data.columns]
    non_existent_exclusions = [
        col for col in columns_to_exclude if col not in data.columns
    ]

    if non_existent_exclusions:
        print(
            f"Note: The following columns in exclusion list don't exist in data: {non_existent_exclusions}"
        )

    # Get all columns that aren't in the exclusion list
    columns_to_keep = [col for col in data.columns if col not in existing_exclusions]

    print(
        f"Keeping {len(columns_to_keep)} columns, excluded {len(existing_exclusions)} columns"
    )

    # Return DataFrame with unwanted columns removed
    return data[columns_to_keep]


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


def enrich_market_data(data, file_path):
    """
    Enrich market data with contract specifications.

    Args:
        data: DataFrame with market data
        file_path: Path to the original data file (used to extract symbol)

    Returns:
        DataFrame with added contract specification fields
    """
    # Extract contract data
    contract_data = extract_contract_data(file_path)

    # Map contract types to column prefixes
    type_to_prefix = {
        "PERPETUAL": "perpetual",
        "CURRENT_QUARTER": "prompt",
        "NEXT_QUARTER": "next",
    }

    # For each contract type, add the specified columns to the data
    for contract_type, contract_info in contract_data.items():
        if contract_type in type_to_prefix:
            prefix = type_to_prefix[contract_type]

            # Add columns with the specified naming convention
            data[f"{prefix}_initial"] = contract_info["requiredMarginPercent"]
            data[f"{prefix}_maint"] = contract_info["maintMarginPercent"]
            data[f"{prefix}_liquidation_fee"] = contract_info["liquidationFee"]

    return data


# Test the function if this script is run directly
if __name__ == "__main__":
    # Test with a sample file path
    test_file_path = "data/markets/BTCUSDT_1d.csv"

    # Load and enrich the data (with column exclusion)
    data = load_data(test_file_path, exclude_cols=True)

    if data is not None:
        # Export a small sample to CSV for inspection
        sample_output_path = "sample_filtered_data.csv"
        data.head(20).to_csv(sample_output_path)
        print(f"\nExported filtered sample data to {sample_output_path}")

        # Show number of columns before and after filtering
        full_data = load_data(test_file_path, exclude_cols=False)
        print(f"\nNumber of columns in full dataset: {len(full_data.columns)}")
        print(f"Number of columns after exclusion: {len(data.columns)}")

        # Print the retained column names
        print("\nRetained columns:")
        for col in data.columns:
            print(f"  - {col}")
