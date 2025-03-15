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
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    try:
        dates = pd.to_datetime(results["dates"])

        filtered_dates = dates
        filtered_equity = results["equity_curve"]
        filtered_spot_pnl = results["spot_pnl"]
        filtered_perp_pnl = results["perp_pnl"]
        filtered_funding = results["cumulative_funding"]
        filtered_net_market_pnl = results["net_market_pnl"]
        filtered_notional = results["notional_values"]

        fig, axes = plt.subplots(
            4, 1, figsize=(14, 20), gridspec_kw={"height_ratios": [2, 1, 1, 1]}
        )

        plot_equity_curve(
            axes[0],
            {
                "dates": filtered_dates,
                "equity_curve": filtered_equity,
                "notional_values": filtered_notional,
            },
            metrics,
        )

        plot_pnl_components(
            axes[1],
            {
                "dates": filtered_dates,
                "spot_pnl": filtered_spot_pnl,
                "perp_pnl": filtered_perp_pnl,
                "cumulative_funding": filtered_funding,
            },
        )

        plot_net_market_pnl(
            axes[2],
            {"dates": filtered_dates, "net_market_pnl": filtered_net_market_pnl},
            metrics,
        )

        plot_basis_funding_time_series(axes[3], results["data"])

        plt.savefig(f"{output_dir}/backtest_results_{timestamp}.png")
        plt.close(fig)

        print(
            f"Performance charts saved to {output_dir}/backtest_results_{timestamp}.png"
        )
    except Exception as e:
        print(f"Error creating performance charts: {str(e)}")


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


# Add to analytics/visualizations.py


def plot_health_factor_over_time(ax, health_factors, dates):
    """
    Plot health factor evolution over the backtest period.

    Args:
        ax: Matplotlib axes object
        health_factors: List of health factor values
        dates: List of corresponding dates
    """
    # Create color mapping
    colors = []
    for hf in health_factors:
        if hf is None:
            colors.append("gray")
        elif hf >= 3:
            colors.append("green")
        elif hf >= 1.5:
            colors.append("yellow")
        elif hf >= 1:
            colors.append("orange")
        else:
            colors.append("red")

    # Plot health factor
    ax.scatter(dates, health_factors, c=colors, alpha=0.7)
    ax.plot(dates, health_factors, "k-", alpha=0.3)

    # Add threshold lines
    ax.axhline(y=3, color="green", linestyle="--", alpha=0.5, label="Very Safe (3.0+)")
    ax.axhline(
        y=1.5,
        color="yellow",
        linestyle="--",
        alpha=0.5,
        label="Moderate Risk (1.5-3.0)",
    )
    ax.axhline(
        y=1, color="red", linestyle="--", alpha=0.5, label="Liquidation Threshold (1.0)"
    )

    # Set labels and title
    ax.set_title("Position Health Factor Over Time")
    ax.set_ylabel("Health Factor")
    ax.set_ylim(bottom=0)  # Start y-axis at 0
    ax.grid(True)
    ax.legend()


def plot_liquidation_buffer(ax, prices, liquidation_prices, dates):
    """
    Plot price and liquidation threshold over time.

    Args:
        ax: Matplotlib axes object
        prices: List of perp prices
        liquidation_prices: List of calculated liquidation prices
        dates: List of corresponding dates
    """
    # Plot price and liquidation price
    ax.plot(dates, prices, "b-", label="Perp Price")
    ax.plot(dates, liquidation_prices, "r-", alpha=0.7, label="Liquidation Threshold")

    # Fill the buffer area
    ax.fill_between(
        dates,
        prices,
        liquidation_prices,
        where=(prices < liquidation_prices),
        color="green",
        alpha=0.2,
        label="Safety Buffer",
    )

    # Highlight danger zones
    buffer_pcts = [
        (liq - price) / price * 100 if price > 0 and liq is not None else None
        for price, liq in zip(prices, liquidation_prices)
    ]

    danger_dates = []
    danger_prices = []

    for i, pct in enumerate(buffer_pcts):
        if pct is not None and pct < 10:  # Less than 10% buffer
            danger_dates.append(dates[i])
            danger_prices.append(prices[i])

    if danger_dates:
        ax.scatter(
            danger_dates,
            danger_prices,
            color="red",
            s=50,
            alpha=0.7,
            label="Danger Zone (<10% Buffer)",
        )

    # Set labels and title
    ax.set_title("Price vs Liquidation Threshold")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()


def create_risk_dashboard(results, metrics, output_dir="strategy/results"):
    """
    Generate a comprehensive risk dashboard.

    Args:
        results: Backtest results dictionary
        metrics: Calculated performance metrics
        output_dir: Output directory for the dashboard
    """
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    # Skip if no risk metrics available
    if "health_factors" not in results:
        print("Risk metrics not available - cannot create risk dashboard")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Prepare data
    dates = results["dates"]
    health_factors = results.get("health_factors", [])

    # Get perp prices either from data or from the perp_close list if available
    if "data" in results and "perp_close" in results["data"]:
        perp_prices = results["data"]["perp_close"].values
    else:
        # Try to reconstruct from entry/exit prices
        perp_entry = results.get("perp_entry", 0)
        perp_exit = results.get("perp_exit", 0)
        # Create a linear interpolation between entry and exit prices
        perp_prices = np.linspace(perp_entry, perp_exit, len(dates))

    liquidation_prices = results.get("liquidation_prices", [])

    # Create figure with multiple subplots
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 18), gridspec_kw={"height_ratios": [2, 1.5, 1.5]}
    )

    # Plot main equity curve
    plot_equity_curve(axes[0], results, metrics)

    # Plot health factor over time
    if health_factors:
        plot_health_factor_over_time(axes[1], health_factors, dates)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Health factor data not available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[1].transAxes,
        )

    # Plot liquidation buffer
    if liquidation_prices and isinstance(perp_prices, (list, np.ndarray)):
        plot_liquidation_buffer(axes[2], perp_prices, liquidation_prices, dates)
    else:
        axes[2].text(
            0.5,
            0.5,
            "Liquidation price data not available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[2].transAxes,
        )

    # Add a title for the entire dashboard
    plt.suptitle("Funding Arbitrage Risk Dashboard", fontsize=16)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
    risk_dashboard_path = f"{output_dir}/risk_dashboard_{timestamp}.png"
    plt.savefig(risk_dashboard_path)
    plt.close(fig)

    print(f"Risk dashboard saved to {risk_dashboard_path}")


def print_risk_summary(results):
    """
    Print a summary of risk metrics.

    Args:
        results: Backtest results dictionary
    """
    print("\n=== RISK MANAGEMENT SUMMARY ===")

    # Get latest risk data
    latest_health = results.get("latest_health_factor", None)
    liquidation_price = results.get("latest_liquidation_price", None)
    current_price = (
        results.get("data", {}).get("perp_close", {}).iloc[-1]
        if "data" in results
        else None
    )

    # Health factor summary
    if latest_health:
        print(f"Final Health Factor: {latest_health:.2f}")

        if latest_health >= 3:
            print("Risk Level: Very Safe (Health Factor >= 3)")
        elif latest_health >= 1.5:
            print("Risk Level: Moderate Risk (1.5 <= Health Factor < 3)")
        elif latest_health >= 1:
            print("Risk Level: High Risk (1 <= Health Factor < 1.5)")
        else:
            print("Risk Level: DANGER - Below Liquidation Threshold!")

    # Liquidation buffer
    if liquidation_price and current_price:
        buffer_pct = (liquidation_price - current_price) / current_price * 100
        print(f"Current Perp Price: ${current_price:.2f}")
        print(f"Liquidation Threshold: ${liquidation_price:.2f}")
        print(f"Buffer to Liquidation: {buffer_pct:.2f}%")

        if buffer_pct >= 50:
            print("Buffer Status: Very Large Buffer (>50%)")
        elif buffer_pct >= 25:
            print("Buffer Status: Healthy Buffer (25-50%)")
        elif buffer_pct >= 10:
            print("Buffer Status: Adequate Buffer (10-25%)")
        elif buffer_pct > 0:
            print("Buffer Status: CAUTION - Small Buffer (<10%)")
        else:
            print("Buffer Status: CRITICAL - No Buffer!")

    # Risk events during backtest
    close_calls = results.get("close_calls", 0)
    if close_calls:
        print(f"\nRisk Events During Backtest:")
        print(f"  Near-Liquidation Events (Health Factor < 1.2): {close_calls}")

    # Stress test summary
    stress_test = results.get("latest_stress_test", None)
    if stress_test:
        print("\nStress Test Summary:")

        # Price increase scenarios
        up_scenarios = stress_test.get("upside_scenarios", [])
        if up_scenarios:
            max_safe_increase = None
            for scenario in up_scenarios:
                if not scenario.get("liquidation", False):
                    max_safe_increase = scenario.get("price_change_pct", None)

            if max_safe_increase is not None:
                print(f"  Can withstand price increase of: {max_safe_increase}%")
            else:
                print("  Cannot withstand any significant price increase")

        # Basis expansion scenarios
        basis_scenarios = stress_test.get("basis_expansion_scenarios", [])
        if basis_scenarios:
            max_safe_basis = None
            for scenario in basis_scenarios:
                if not scenario.get("liquidation", False):
                    max_safe_basis = scenario.get("basis_change_pct", None)

            if max_safe_basis is not None:
                print(f"  Can withstand basis expansion of: {max_safe_basis}%")
            else:
                print("  Cannot withstand any significant basis expansion")

    print("\nRecommendations:")
    if latest_health and latest_health < 1.5:
        print("  - URGENT: Consider reducing leverage or position size")
        print("  - Monitor position closely for adverse movements")
    elif latest_health and latest_health < 3:
        print("  - Consider setting stop-loss orders to protect capital")
        print("  - Be prepared to exit if funding rates turn negative")
    else:
        print("  - Position appears safe under normal market conditions")
        print("  - Continue to monitor for changes in market conditions")
