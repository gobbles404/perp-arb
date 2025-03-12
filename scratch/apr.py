import pandas as pd
from datetime import datetime

# https://fsr-develop.com/blog-cscalp/tpost/r48x284bt1-binance-funding-rate-how-funding-works-o

# File paths for the CSV files
spot_file = "historical_spot_2024-12-01_to_2024-12-28.csv"
quarterly_file = "historical_quarterly_futures_2024-12-01_to_2024-12-28.csv"
output_file = "btc_futures_premium_analysis.csv"

# Define expiry date for the quarterly futures contract
expiry_date = "2024-12-27"
expiry_date_obj = datetime.strptime(expiry_date, "%Y-%m-%d")


def calculate_days_to_expiry(date_str):
    """Calculate the number of days between a given date and the expiry date."""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        days_diff = (expiry_date_obj - date_obj).days
        return max(days_diff, 0)  # Return 0 if date is after expiry
    except:
        return None


def main():
    """
    Calculate the historical premium/discount of BTC quarterly futures,
    expressed as an annualized percentage rate (APR).
    """
    print("Loading data files...")

    # Load CSV files
    spot_data = pd.read_csv(spot_file)
    quarterly_data = pd.read_csv(quarterly_file)

    # Ensure timestamp is treated as date
    spot_data["Timestamp"] = pd.to_datetime(spot_data["Timestamp"]).dt.strftime(
        "%Y-%m-%d"
    )
    quarterly_data["Timestamp"] = pd.to_datetime(
        quarterly_data["Timestamp"]
    ).dt.strftime("%Y-%m-%d")

    # Merge the two dataframes on Timestamp
    merged_data = pd.merge(
        spot_data, quarterly_data, on="Timestamp", suffixes=("_spot", "_quarterly")
    )

    print(f"Processing {len(merged_data)} days of data...")

    # Calculate days to expiry for each date
    merged_data["days_to_expiry"] = merged_data["Timestamp"].apply(
        calculate_days_to_expiry
    )

    # Calculate the basis (absolute difference)
    merged_data["basis"] = merged_data["Close_quarterly"] - merged_data["Close_spot"]

    # Calculate the premium percentage (not annualized)
    merged_data["premium_pct"] = (
        merged_data["Close_quarterly"] / merged_data["Close_spot"] - 1
    ) * 100

    # Calculate the annualized premium rate (APR)
    def calculate_apr(row):
        if row["days_to_expiry"] > 0:
            return (
                (row["Close_quarterly"] / row["Close_spot"] - 1)
                * (365 / row["days_to_expiry"])
                * 100
            )
        return None

    merged_data["apr"] = merged_data.apply(calculate_apr, axis=1)

    # Select and rename columns for the output
    result = merged_data[
        [
            "Timestamp",
            "Close_spot",
            "Close_quarterly",
            "days_to_expiry",
            "basis",
            "premium_pct",
            "apr",
        ]
    ]
    result.columns = [
        "date",
        "spot_price",
        "quarterly_price",
        "days_to_expiry",
        "basis",
        "premium_percentage",
        "annualized_premium_rate",
    ]

    # Add market state (contango or backwardation)
    result["market_state"] = result["premium_percentage"].apply(
        lambda x: "Contango" if x > 0 else ("Backwardation" if x < 0 else "Neutral")
    )

    # Save to CSV
    result.to_csv(output_file, index=False)

    # Print summary statistics
    valid_apr = result["annualized_premium_rate"].dropna()

    print("\nAnalysis Complete!")
    print(f"Results saved to {output_file}")
    print("\nSummary Statistics:")
    print(f"Average Premium: {result['premium_percentage'].mean():.4f}%")
    print(f"Average APR: {valid_apr.mean():.4f}%")
    print(f"Min APR: {valid_apr.min():.4f}%")
    print(f"Max APR: {valid_apr.max():.4f}%")
    print(f"Days in Contango: {(result['premium_percentage'] > 0).sum()}")
    print(f"Days in Backwardation: {(result['premium_percentage'] < 0).sum()}")


if __name__ == "__main__":
    main()
