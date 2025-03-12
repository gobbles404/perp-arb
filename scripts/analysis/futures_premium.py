import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


class FuturesPremiumAnalyzer:
    """
    A class to analyze futures premium and calculate implied interest rates
    for cryptocurrency futures contracts.
    """

    def __init__(self, base_dir=None):
        """
        Initialize the analyzer with the base directory for data files.

        Parameters:
        -----------
        base_dir : str, optional
            The base directory containing the data files. If None, uses current directory.
        """
        self.base_dir = base_dir if base_dir is not None else os.getcwd()

    def load_data(self, asset="BTCUSDT", timeframe="1d"):
        """
        Load futures and spot data for the specified asset and timeframe.

        Parameters:
        -----------
        asset : str
            The asset to analyze (e.g., 'BTCUSDT')
        timeframe : str
            The timeframe to analyze (e.g., '1d')

        Returns:
        --------
        tuple
            (futures_df, spot_df) containing the loaded DataFrames
        """
        # Define file paths based on your data structure
        spot_csv = os.path.join(self.base_dir, "data", "historical_spot_prices.csv")
        futures_csv = os.path.join(
            self.base_dir, "data", f"historical_futures_prices_{asset}_241227.csv"
        )

        # Load data
        spot_df = pd.read_csv(spot_csv, parse_dates=["Timestamp"])
        futures_df = pd.read_csv(futures_csv, parse_dates=["Timestamp"])

        # Add days_till_expiry if it doesn't exist
        if "days_till_expiry" not in futures_df.columns:
            # Calculate days until December 27, 2024 for each timestamp
            futures_df["expiry_date"] = pd.to_datetime("2024-12-27")
            futures_df["days_till_expiry"] = (
                futures_df["expiry_date"] - futures_df["Timestamp"]
            ).dt.days

        # Add Contract column if it doesn't exist
        if "Contract" not in futures_df.columns:
            futures_df["Contract"] = f"{asset}_241227"

        print(
            f"Loaded {len(futures_df)} futures records and {len(spot_df)} spot records"
        )

        return futures_df, spot_df

    def calculate_premium(self, futures_df, spot_df):
        """
        Calculate the premium/discount of futures contracts and the implied interest rate.

        Parameters:
        -----------
        futures_df : pandas.DataFrame
            DataFrame containing futures data
        spot_df : pandas.DataFrame
            DataFrame containing spot data

        Returns:
        --------
        pandas.DataFrame
            DataFrame with premium and implied rate calculations
        """
        # Make a copy of the dataframes to avoid modifying the originals
        futures_df = futures_df.copy()
        spot_df = spot_df.copy()

        # Make sure Timestamp is treated consistently
        futures_df["Timestamp"] = pd.to_datetime(futures_df["Timestamp"])
        spot_df["Timestamp"] = pd.to_datetime(spot_df["Timestamp"])

        # Merge spot and futures data on Timestamp
        merged_df = pd.merge(
            futures_df,
            spot_df[["Timestamp", "Close"]],
            on="Timestamp",
            how="inner",
            suffixes=("_futures", "_spot"),
        )

        # Calculate premium as (futures - spot) / spot
        merged_df["premium_pct"] = (
            100
            * (merged_df["Close_futures"] - merged_df["Close_spot"])
            / merged_df["Close_spot"]
        )

        # Calculate implied interest rate (annualized)
        merged_df["time_to_expiry_years"] = merged_df["days_till_expiry"] / 365
        merged_df["implied_rate_pct"] = (
            100
            * np.log(merged_df["Close_futures"] / merged_df["Close_spot"])
            / merged_df["time_to_expiry_years"]
        )

        # Rearrange and clean up columns
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

        return result_df

    def plot_premium_by_contract(self, result_df):
        """
        Plot the premium and implied rate for each contract over time.

        Parameters:
        -----------
        result_df : pandas.DataFrame
            DataFrame with premium and implied rate calculations

        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the plots
        """
        # Group by contract
        contracts = result_df["Contract"].unique()

        # Create figure with subplots
        fig, axes = plt.subplots(
            len(contracts), 1, figsize=(12, 4 * len(contracts)), sharex=True
        )

        # If there's only one contract, axes will not be an array
        if len(contracts) == 1:
            axes = [axes]

        for i, contract in enumerate(contracts):
            contract_data = result_df[result_df["Contract"] == contract].sort_values(
                "Timestamp"
            )

            ax = axes[i]
            ax2 = ax.twinx()

            # Plot premium
            ax.plot(
                contract_data["Timestamp"],
                contract_data["premium_pct"],
                "b-",
                label="Premium (%)",
            )
            ax.set_ylabel("Premium (%)", color="b")
            ax.tick_params(axis="y", labelcolor="b")

            # Plot implied rate
            ax2.plot(
                contract_data["Timestamp"],
                contract_data["implied_rate_pct"],
                "r-",
                label="Implied Rate (% APR)",
            )
            ax2.set_ylabel("Implied Rate (% APR)", color="r")
            ax2.tick_params(axis="y", labelcolor="r")

            ax.set_title(f"Contract: {contract}")

            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.tight_layout()
        plt.show()

        return fig

    def plot_premium_vs_expiry(self, result_df):
        """
        Create a scatter plot of premium vs days till expiry.

        Parameters:
        -----------
        result_df : pandas.DataFrame
            DataFrame with premium and implied rate calculations

        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create a scatter plot
        scatter = ax.scatter(
            result_df["days_till_expiry"],
            result_df["premium_pct"],
            alpha=0.6,
            c=result_df["Timestamp"].astype(np.int64)
            // 10**9,  # Convert to Unix timestamp for color
            cmap="viridis",
        )

        # Add a trend line
        z = np.polyfit(result_df["days_till_expiry"], result_df["premium_pct"], 1)
        p = np.poly1d(z)
        ax.plot(
            result_df["days_till_expiry"],
            p(result_df["days_till_expiry"]),
            "r--",
            label=f"Trend: y={z[0]:.6f}x+{z[1]:.4f}",
        )

        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Time")

        # Add axis labels and title
        ax.set_xlabel("Days Till Expiry")
        ax.set_ylabel("Premium (%)")
        ax.set_title("Futures Premium vs Days Till Expiry")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

    def plot_implied_rate_distribution(self, result_df):
        """
        Plot a histogram of implied rates.

        Parameters:
        -----------
        result_df : pandas.DataFrame
            DataFrame with premium and implied rate calculations

        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the histogram
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        ax.hist(
            result_df["implied_rate_pct"],
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )

        # Add a vertical line for the mean
        mean_rate = result_df["implied_rate_pct"].mean()
        ax.axvline(
            mean_rate,
            color="red",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean: {mean_rate:.2f}%",
        )

        # Add labels and title
        ax.set_xlabel("Implied Rate (% APR)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Implied Interest Rates")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

    def summarize_results(self, result_df):
        """
        Generate summary statistics for the premium and implied rate.

        Parameters:
        -----------
        result_df : pandas.DataFrame
            DataFrame with premium and implied rate calculations

        Returns:
        --------
        pandas.DataFrame
            Summary statistics by contract
        """
        # Group by contract and calculate statistics
        summary = result_df.groupby("Contract").agg(
            {
                "premium_pct": ["mean", "std", "min", "max", "count"],
                "implied_rate_pct": ["mean", "std", "min", "max"],
            }
        )

        # Flatten the column names
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]

        return summary

    def run_analysis(self, asset="BTCUSDT", timeframe="1d"):
        """
        Run the full analysis for a specific asset and timeframe.

        Parameters:
        -----------
        asset : str
            The asset to analyze (e.g., 'BTCUSDT')
        timeframe : str
            The timeframe to analyze (e.g., '1d')

        Returns:
        --------
        tuple
            (result_df, summary) containing the results DataFrame and summary statistics
        """
        print(f"Analyzing {asset} futures premium...")

        # Load data
        futures_df, spot_df = self.load_data(asset, timeframe)

        # Calculate premium and implied rate
        result_df = self.calculate_premium(futures_df, spot_df)

        # Generate summary statistics
        summary = self.summarize_results(result_df)

        print("\nSummary by Contract:")
        print(summary)

        # Average implied rate across all contracts
        avg_rate = result_df["implied_rate_pct"].mean()
        print(f"\nAverage Implied Interest Rate (APR): {avg_rate:.2f}%")

        # Create visualizations
        self.plot_premium_by_contract(result_df)
        self.plot_premium_vs_expiry(result_df)
        self.plot_implied_rate_distribution(result_df)

        return result_df, summary
