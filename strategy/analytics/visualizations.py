"""
Modified visualization function to fix the datetime conversion overflow error.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
from scipy import stats


def create_performance_charts(results, metrics, output_dir="strategy/results"):
    """Generate visualization charts for backtest results with improved date handling."""
    # Create directory for plots
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Ensure dates are proper datetime objects
    try:
        # Convert dates to pandas datetime if they're not already
        if isinstance(results["dates"], list):
            # Check if dates are already datetime objects
            if not all(isinstance(d, datetime) for d in results["dates"]):
                dates = pd.to_datetime(results["dates"])
            else:
                dates = results["dates"]
        else:
            dates = pd.to_datetime(results["dates"])

        # Filter out any problematic dates (too far in future/past)
        # Use a safe date range (e.g., 1970 to 2100)
        min_safe_date = pd.Timestamp("1970-01-01")
        max_safe_date = pd.Timestamp("2100-01-01")

        valid_dates_mask = (dates >= min_safe_date) & (dates <= max_safe_date)

        if not all(valid_dates_mask):
            print(
                f"Warning: Filtering out {sum(~valid_dates_mask)} invalid date entries"
            )

            # Create filtered versions of all time series data
            filtered_dates = dates[valid_dates_mask]
            filtered_equity = pd.Series(results["equity_curve"])[
                valid_dates_mask
            ].values
            filtered_spot_pnl = pd.Series(results["spot_pnl"])[valid_dates_mask].values
            filtered_perp_pnl = pd.Series(results["perp_pnl"])[valid_dates_mask].values
            filtered_funding = pd.Series(results["cumulative_funding"])[
                valid_dates_mask
            ].values
            filtered_net_market_pnl = pd.Series(results["net_market_pnl"])[
                valid_dates_mask
            ].values
            filtered_notional = pd.Series(results["notional_values"])[
                valid_dates_mask
            ].values
        else:
            # Use original data if all dates are valid
            filtered_dates = dates
            filtered_equity = results["equity_curve"]
            filtered_spot_pnl = results["spot_pnl"]
            filtered_perp_pnl = results["perp_pnl"]
            filtered_funding = results["cumulative_funding"]
            filtered_net_market_pnl = results["net_market_pnl"]
            filtered_notional = results["notional_values"]

        # Create figure with subplots
        fig, axes = plt.subplots(
            5, 1, figsize=(14, 24), gridspec_kw={"height_ratios": [2, 1, 1, 1, 1.5]}
        )

        # Plot 1: Equity curve and notional value with filtered data
        plot_equity_curve(
            axes[0],
            {
                "dates": filtered_dates,
                "equity_curve": filtered_equity,
                "notional_values": filtered_notional,
            },
            metrics,
        )

        # Plot 2: Cumulative PnL components with filtered data
        plot_pnl_components(
            axes[1],
            {
                "dates": filtered_dates,
                "spot_pnl": filtered_spot_pnl,
                "perp_pnl": filtered_perp_pnl,
                "cumulative_funding": filtered_funding,
            },
        )

        # Plot 3: Net Market PnL with filtered data
        plot_net_market_pnl(
            axes[2],
            {"dates": filtered_dates, "net_market_pnl": filtered_net_market_pnl},
            metrics,
        )

        # Plot 4: Basis and Funding Rate - ensure data["Timestamp"] is properly handled
        plot_basis_funding_time_series(axes[3], results["data"])

        # Plot 5: Basis vs Funding Rate Scatter Plot
        plot_basis_funding_correlation(axes[4], results["data"], metrics)

        # Format x-axis dates for time series plots
        for ax in axes[:-1]:  # Skip the scatter plot
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Layout and save - use tight_layout with a larger pad to avoid issues
        plt.tight_layout(pad=3.0)
        fig.autofmt_xdate(rotation=45)

        plt.savefig(f"{output_dir}/backtest_results_{timestamp}.png")
        plt.close(fig)  # Close the figure to free memory

        print(
            f"Performance charts saved to {output_dir}/backtest_results_{timestamp}.png"
        )

    except Exception as e:
        print(f"Error creating performance charts: {str(e)}")
        print("Continuing with backtesting process despite chart error.")


def plot_equity_curve(ax, results, metrics):
    """Plot equity curve and notional value."""
    ax.set_title("Long Spot, Short Perp Strategy Performance")

    # Plot equity curve
    color = "green" if metrics["total_return_pct"] > 0 else "red"
    ax.plot(
        results["dates"],
        results["equity_curve"],
        label="Portfolio Value",
        color=color,
        linewidth=2,
    )

    # Add annotations
    max_equity = max(results["equity_curve"])
    min_equity = min(results["equity_curve"])
    max_idx = (
        results["equity_curve"].index(max_equity)
        if isinstance(results["equity_curve"], list)
        else np.argmax(results["equity_curve"])
    )
    min_idx = (
        results["equity_curve"].index(min_equity)
        if isinstance(results["equity_curve"], list)
        else np.argmin(results["equity_curve"])
    )

    ax.annotate(
        f"Max: ${max_equity:.2f}",
        xy=(results["dates"][max_idx], max_equity),
        xytext=(10, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )

    ax.annotate(
        f"Min: ${min_equity:.2f}",
        xy=(results["dates"][min_idx], min_equity),
        xytext=(10, -20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )

    # Plot secondary y-axis for notional value
    ax1b = ax.twinx()
    ax1b.plot(
        results["dates"],
        results["notional_values"],
        label="Total Notional",
        color="purple",
        linestyle="--",
        alpha=0.7,
    )
    ax1b.set_ylabel("Notional Value ($)", color="purple")
    ax1b.tick_params(axis="y", labelcolor="purple")
    ax1b.yaxis.set_major_formatter("{x:,.0f}")

    # Format y-axis as currency
    ax.yaxis.set_major_formatter("{x:,.0f}")

    # Add key metrics as text
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    # Format metrics safely
    total_return_str = (
        f"{metrics['total_return_pct']:.2f}%"
        if not np.isnan(metrics["total_return_pct"])
        else "N/A"
    )
    ann_return_str = (
        f"{metrics['annualized_return']:.2f}%"
        if not np.isnan(metrics["annualized_return"])
        else "N/A"
    )
    funding_apr_str = (
        f"{metrics['funding_apr']:.2f}%"
        if not np.isnan(metrics["funding_apr"])
        else "N/A"
    )
    funding_apy_str = (
        f"{metrics['funding_apy']:.2f}%"
        if not np.isnan(metrics["funding_apy"])
        else "N/A"
    )
    sharpe_str = (
        f"{metrics['sharpe_ratio']:.2f}"
        if not np.isnan(metrics["sharpe_ratio"])
        else "N/A"
    )
    max_dd_str = (
        f"{metrics['max_drawdown']:.2f}%"
        if not np.isnan(metrics["max_drawdown"])
        else "N/A"
    )
    cap_eff_str = (
        f"{metrics['capital_efficiency']:.2f}x"
        if not np.isnan(metrics["capital_efficiency"])
        else "N/A"
    )

    textstr = "\n".join(
        (
            f"Total Return: {total_return_str}",
            f"Ann. Return: {ann_return_str}",
            f"Funding APR: {funding_apr_str}",
            f"Funding APY: {funding_apy_str}",
            f"Sharpe: {sharpe_str}",
            f"Max DD: {max_dd_str}",
            f"Capital Efficiency: {cap_eff_str}",
        )
    )

    ax.text(
        0.02,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=props,
    )

    ax.grid(True)
    ax.legend(loc="upper left")


def plot_pnl_components(ax, results):
    """Plot cumulative PnL components."""
    ax.set_title("Cumulative P&L Components")

    ax.plot(results["dates"], results["spot_pnl"], label="Spot P&L", color="blue")
    ax.plot(results["dates"], results["perp_pnl"], label="Perp P&L", color="red")
    ax.plot(
        results["dates"],
        results["cumulative_funding"],
        label="Funding Income",
        color="green",
        linewidth=2,
    )

    # Format y-axis as currency
    ax.yaxis.set_major_formatter("{x:,.0f}")

    ax.grid(True)
    ax.legend(loc="best")


def plot_net_market_pnl(ax, results, metrics):
    """Plot net market PnL (spot + perp without funding)."""
    ax.set_title("Net Market P&L (Spot + Perp without Funding)")

    # Plot net market PnL
    ax.plot(
        results["dates"],
        results["net_market_pnl"],
        label="Net Market P&L",
        color="purple",
    )
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter("{x:,.0f}")

    # Add annotations for max and min net PnL
    net_pnl_max = metrics.get("net_pnl_max", max(results["net_market_pnl"]))
    net_pnl_min = metrics.get("net_pnl_min", min(results["net_market_pnl"]))

    # Handle finding the index in a way that works for both lists and numpy arrays
    if isinstance(results["net_market_pnl"], list):
        try:
            max_idx = results["net_market_pnl"].index(net_pnl_max)
        except ValueError:
            max_idx = 0

        try:
            min_idx = results["net_market_pnl"].index(net_pnl_min)
        except ValueError:
            min_idx = 0
    else:
        max_idx = np.argmax(results["net_market_pnl"])
        min_idx = np.argmin(results["net_market_pnl"])

    if max_idx > 0:
        ax.annotate(
            f"Max: ${net_pnl_max:.2f}",
            xy=(results["dates"][max_idx], net_pnl_max),
            xytext=(10, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
        )

    if min_idx > 0:
        ax.annotate(
            f"Min: ${net_pnl_min:.2f}",
            xy=(results["dates"][min_idx], net_pnl_min),
            xytext=(10, -20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
        )

    # Add net PnL statistics in text box
    net_pnl_props = dict(boxstyle="round", facecolor="lightskyblue", alpha=0.5)
    net_pnl_text = "\n".join(
        (
            f"Net PnL Volatility: ${metrics.get('net_pnl_volatility', 0):.2f}",
            f"Net PnL Range: ${metrics.get('net_pnl_range', 0):.2f}",
        )
    )
    ax.text(
        0.02,
        0.05,
        net_pnl_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=net_pnl_props,
    )

    ax.grid(True)
    ax.legend(loc="best")


def plot_basis_funding_time_series(ax, data):
    """Plot basis and funding rate time series with inverted funding axis."""
    ax.set_title("Basis and Funding Rate (Inverted)")

    # Convert timestamps to datetime if they're not already
    if "Timestamp" in data.columns:
        # Copy to avoid modifying the original data
        data_copy = data.copy()
        # Convert Timestamp column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data_copy["Timestamp"]):
            data_copy["Timestamp"] = pd.to_datetime(data_copy["Timestamp"])

        # Filter out invalid timestamps (from far future/past)
        min_safe_date = pd.Timestamp("1970-01-01")
        max_safe_date = pd.Timestamp("2100-01-01")

        valid_dates_mask = (data_copy["Timestamp"] >= min_safe_date) & (
            data_copy["Timestamp"] <= max_safe_date
        )
        if not all(valid_dates_mask):
            print(
                f"Warning: Filtering out {sum(~valid_dates_mask)} invalid timestamps in basis/funding plot"
            )
            data_copy = data_copy[valid_dates_mask].reset_index(drop=True)
    else:
        # If there's no Timestamp column, just use the original data
        data_copy = data

    # Calculate basis (difference between perp and spot)
    basis_pct = (
        data_copy["basis_pct"]
        if "basis_pct" in data_copy.columns
        else [
            (p / s - 1) * 100
            for p, s in zip(data_copy["perp_close"], data_copy["spot_close"])
        ]
    )

    # Plot basis
    color_basis = "tab:blue"
    ax.plot(data_copy["Timestamp"], basis_pct, label="Basis %", color=color_basis)
    ax.set_ylabel("Basis (%)", color=color_basis)
    ax.tick_params(axis="y", labelcolor=color_basis)

    # Create second y-axis for funding rate
    ax4b = ax.twinx()
    color_funding = "tab:red"

    # Convert funding rate to annualized percentage
    funding_pct = (
        data_copy["funding_apr"]
        if "funding_apr" in data_copy.columns
        else data_copy["funding_rate"] * 3 * 365 * 100
    )

    ax4b.plot(
        data_copy["Timestamp"],
        funding_pct,
        label="Funding Rate (APR)",
        color=color_funding,
        alpha=0.7,
    )
    ax4b.set_ylabel("Funding Rate APR (%) - Inverted", color=color_funding)
    ax4b.tick_params(axis="y", labelcolor=color_funding)

    # Invert the funding rate axis
    ax4b.invert_yaxis()

    # Add horizontal line at zero
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Add explanation of the inverted axis
    props = dict(boxstyle="round", facecolor="lightpink", alpha=0.5)
    axis_explanation = "\n".join(
        (
            "Funding Rate Axis Inverted:",
            "HIGHER on chart = NEGATIVE funding (cost)",
            "LOWER on chart = POSITIVE funding (income)",
        )
    )
    ax.text(
        0.02,
        0.95,
        axis_explanation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax.grid(True)


def plot_basis_funding_correlation(ax, data, metrics):
    """Plot basis vs funding rate scatter plot with correlation."""
    ax.set_title("Basis vs Funding Rate Correlation")

    # Get basis and funding data
    basis_pct = (
        data["basis_pct"]
        if "basis_pct" in data.columns
        else [(p / s - 1) * 100 for p, s in zip(data["perp_close"], data["spot_close"])]
    )
    funding_pct = (
        data["funding_apr"]
        if "funding_apr" in data.columns
        else data["funding_rate"] * 3 * 365 * 100
    )

    # Clean data for scatter plot
    valid_indices = ~(np.isnan(basis_pct) | np.isnan(funding_pct))

    # Convert to numpy arrays for safety and easier filtering
    basis_pct_np = np.array(basis_pct)
    funding_pct_np = np.array(funding_pct)

    valid_basis = basis_pct_np[valid_indices]
    valid_funding = funding_pct_np[valid_indices]

    # Create scatter plot
    scatter = ax.scatter(valid_basis, valid_funding, alpha=0.5, c="blue")
    ax.set_xlabel("Basis (%)")
    ax.set_ylabel("Funding Rate APR (%)")

    # Add regression line
    if len(valid_basis) > 1 and len(valid_funding) > 1:
        # Use scipy.stats.linregress to get regression line
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_basis, valid_funding
            )

            # Plot regression line
            x_range = np.linspace(min(valid_basis), max(valid_basis), 100)
            ax.plot(
                x_range,
                intercept + slope * x_range,
                "r-",
                linewidth=2,
                label=f"y = {slope:.4f}x + {intercept:.4f}",
            )

            # Add text box with correlation info
            corr_props = dict(boxstyle="round", facecolor="lightgreen", alpha=0.5)
            corr_text = "\n".join(
                (
                    f"Correlation: {metrics.get('basis_funding_correlation', r_value):.4f}",
                    f"R²: {r_value**2:.4f}",
                    f"p-value: {metrics.get('basis_funding_p_value', p_value):.4f}",
                    f"Regression: y = {slope:.4f}x + {intercept:.4f}",
                )
            )
            ax.text(
                0.02,
                0.95,
                corr_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=corr_props,
            )
        except Exception as e:
            print(f"Warning: Could not calculate regression for scatter plot: {e}")

    # Add grid and zero lines
    ax.grid(True)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

    # Annotate quadrants
    # Top-right: Positive basis, Negative funding (costs you)
    ax.text(
        0.95,
        0.05,
        "Premium\nCosts Funding",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
    )

    # Bottom-right: Positive basis, Positive funding (pays you)
    ax.text(
        0.95,
        0.95,
        "Premium\nPays Funding",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    # Top-left: Negative basis, Negative funding (costs you)
    ax.text(
        0.05,
        0.05,
        "Discount\nCosts Funding",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="salmon", alpha=0.5),
    )

    # Bottom-left: Negative basis, Positive funding (pays you)
    ax.text(
        0.05,
        0.95,
        "Discount\nPays Funding",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    ax.legend(loc="center right")
