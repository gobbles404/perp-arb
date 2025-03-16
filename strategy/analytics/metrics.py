"""
Performance metrics calculation for trading strategies.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime


def calculate_performance_metrics(results):
    """Calculate comprehensive performance metrics from backtest results."""
    # Basic variables
    initial_capital = results["initial_capital"]
    final_capital = results["final_capital"]
    entry_date = results["entry_date"]
    exit_date = results["exit_date"]
    days_held = (exit_date - entry_date).days if isinstance(entry_date, datetime) else 0

    # Final PnL values - with NaN handling
    try:
        spot_pnl = results["spot_pnl"][-1] if len(results["spot_pnl"]) > 0 else 0
        if np.isnan(spot_pnl):
            spot_pnl = 0
    except:
        spot_pnl = 0

    try:
        perp_pnl = results["perp_pnl"][-1] if len(results["perp_pnl"]) > 0 else 0
        if np.isnan(perp_pnl):
            perp_pnl = 0
    except:
        perp_pnl = 0

    try:
        funding_pnl = (
            results["cumulative_funding"][-1]
            if len(results["cumulative_funding"]) > 0
            else 0
        )
        if np.isnan(funding_pnl):
            funding_pnl = 0
    except:
        funding_pnl = 0

    net_market_pnl = spot_pnl + perp_pnl
    total_pnl = final_capital - initial_capital
    fees_paid = results["entry_fee"] + results["exit_fee"]

    # Return metrics
    total_return_pct = (
        (final_capital / initial_capital - 1) * 100 if initial_capital > 0 else 0
    )

    # Initialize time-based metrics
    annualized_return = 0.0
    funding_apr = 0.0
    funding_apy = 0.0
    avg_funding_rate = 0.0
    avg_funding_apr = 0.0

    # Calculate time-based metrics if we have sufficient data
    if days_held > 0:
        # Annualized return
        ann_factor = 365 / days_held
        annualized_return = ((1 + total_return_pct / 100) ** ann_factor - 1) * 100

        # Funding metrics
        funding_apr = (
            (funding_pnl / initial_capital) * (365 / days_held) * 100
            if initial_capital > 0
            else 0
        )

        # APY (with compounding)
        daily_funding_rate = (
            funding_pnl / (initial_capital * days_held)
            if initial_capital > 0 and days_held > 0
            else 0
        )
        funding_apy = ((1 + daily_funding_rate) ** 365 - 1) * 100

        # Average funding rate - with NaN handling
        valid_funding_rates = [r for r in results["funding_rates"] if not np.isnan(r)]
        avg_funding_rate = np.mean(valid_funding_rates) if valid_funding_rates else 0
        avg_funding_apr = avg_funding_rate * 3 * 365 * 100  # 3 funding periods per day

    # Risk metrics
    equity_series = pd.Series(results["equity_curve"])

    # Calculate drawdown
    rolling_max = equity_series.cummax()
    drawdown = (equity_series / rolling_max - 1) * 100
    max_drawdown = drawdown.min() if not np.isnan(drawdown.min()) else 0

    # Calculate Sharpe
    if len(equity_series) > 1:
        daily_returns = equity_series.pct_change().dropna()
        # Remove infinite values that might appear from division by zero
        daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()

        sharpe_ratio = (
            np.sqrt(365) * daily_returns.mean() / daily_returns.std()
            if daily_returns.std() > 0 and not np.isnan(daily_returns.mean())
            else 0
        )
    else:
        sharpe_ratio = 0

    # Net PnL statistics
    net_pnl_series = pd.Series(results["net_market_pnl"])
    net_pnl_series = net_pnl_series.replace([np.inf, -np.inf], np.nan).dropna()

    if len(net_pnl_series) > 1:
        net_pnl_volatility = (
            net_pnl_series.std() if not np.isnan(net_pnl_series.std()) else 0
        )
        net_pnl_max = net_pnl_series.max() if not np.isnan(net_pnl_series.max()) else 0
        net_pnl_min = net_pnl_series.min() if not np.isnan(net_pnl_series.min()) else 0
        net_pnl_range = net_pnl_max - net_pnl_min
    else:
        net_pnl_volatility = 0
        net_pnl_max = 0
        net_pnl_min = 0
        net_pnl_range = 0

    # Calculate basis-funding correlation
    data = results["data"]

    # Get basis and funding values
    basis_values = (
        data["basis_pct"]
        if "basis_pct" in data.columns
        else [(p / s - 1) * 100 for p, s in zip(data["perp_close"], data["spot_close"])]
    )
    funding_values = (
        data["funding_apr"]
        if "funding_apr" in data.columns
        else data["funding_rate"] * 3 * 365 * 100
    )

    correlation = 0
    p_value = 1

    if len(basis_values) > 1 and len(funding_values) > 1:
        # Clean data - replace NaN and infinite values
        basis_array = np.array(basis_values)
        funding_array = np.array(funding_values)

        # Create mask for invalid values in either array
        mask = ~(
            np.isnan(basis_array)
            | np.isnan(funding_array)
            | np.isinf(basis_array)
            | np.isinf(funding_array)
        )

        # Extract valid values using mask
        valid_basis = basis_array[mask]
        valid_funding = funding_array[mask]

        # Calculate correlation if we have enough valid data points
        if len(valid_basis) > 1 and len(valid_funding) > 1:
            correlation, p_value = stats.pearsonr(valid_basis, valid_funding)
            # Handle potential NaN in correlation
            if np.isnan(correlation):
                correlation = 0
            if np.isnan(p_value):
                p_value = 1

    # Create metrics dictionary with existing metrics
    metrics = {
        "days_held": days_held,
        "total_pnl": total_pnl,
        "spot_pnl": spot_pnl,
        "perp_pnl": perp_pnl,
        "funding_pnl": funding_pnl,
        "net_market_pnl": net_market_pnl,
        "fees_paid": fees_paid,
        "total_return_pct": total_return_pct,
        "annualized_return": annualized_return,
        "funding_apr": funding_apr,
        "funding_apy": funding_apy,
        "avg_funding_rate": avg_funding_rate,
        "avg_funding_apr": avg_funding_apr,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "net_pnl_volatility": net_pnl_volatility,
        "net_pnl_max": net_pnl_max,
        "net_pnl_min": net_pnl_min,
        "net_pnl_range": net_pnl_range,
        "capital_efficiency": (
            results["final_notional"] / initial_capital if initial_capital > 0 else 0
        ),
        "basis_funding_correlation": correlation,
        "basis_funding_p_value": p_value,
    }

    # Add risk metrics

    # Health factor metrics
    if "health_factors" in results and results["health_factors"]:
        # Filter out None values
        valid_health_factors = [
            hf for hf in results["health_factors"] if hf is not None
        ]

        if valid_health_factors:
            metrics["min_health_factor"] = min(valid_health_factors)
            metrics["avg_health_factor"] = sum(valid_health_factors) / len(
                valid_health_factors
            )
            metrics["final_health_factor"] = valid_health_factors[-1]

            # Calculate time spent in different risk zones
            total_periods = len(valid_health_factors)
            high_risk_periods = sum(1 for hf in valid_health_factors if hf < 1.5)
            safe_periods = sum(1 for hf in valid_health_factors if hf >= 3)

            metrics["pct_time_high_risk"] = (
                (high_risk_periods / total_periods) * 100 if total_periods > 0 else 0
            )
            metrics["pct_time_safe"] = (
                (safe_periods / total_periods) * 100 if total_periods > 0 else 0
            )

    # Liquidation buffer metrics
    if "buffer_percentages" in results and results["buffer_percentages"]:
        # Filter out None values
        valid_buffers = [b for b in results["buffer_percentages"] if b is not None]

        if valid_buffers:
            metrics["min_liquidation_buffer"] = min(valid_buffers)
            metrics["avg_liquidation_buffer"] = sum(valid_buffers) / len(valid_buffers)
            metrics["final_liquidation_buffer"] = valid_buffers[-1]

            # Calculate time spent with dangerously low buffer
            danger_threshold = 10  # Less than 10% buffer is dangerous
            total_buffer_periods = len(valid_buffers)
            dangerous_periods = sum(1 for b in valid_buffers if b < danger_threshold)

            metrics["pct_time_danger_zone"] = (
                (dangerous_periods / total_buffer_periods) * 100
                if total_buffer_periods > 0
                else 0
            )

    # Add close calls to metrics
    metrics["liquidation_close_calls"] = results.get("close_calls", 0)

    # Add margin requirements
    metrics["maintenance_margin_pct"] = results.get("maintenance_margin_pct", 0)
    metrics["initial_margin_pct"] = results.get("initial_margin_pct", 0)

    # Capital allocation metrics
    metrics["spot_allocation"] = results.get("spot_allocation", 0)
    metrics["perp_allocation"] = results.get("perp_allocation", 0)

    return metrics


def print_performance_summary(results, metrics):
    """Print a summary of backtest performance metrics."""
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"Position held for {metrics['days_held']} days")
    print(f"Initial capital: ${results['initial_capital']:.2f}")
    print(f"Final capital: ${results['final_capital']:.2f}")
    print(f"Total P&L: ${metrics['total_pnl']:.2f}")
    print(f"  Spot P&L: ${metrics['spot_pnl']:.2f}")
    print(f"  Perp P&L: ${metrics['perp_pnl']:.2f}")
    print(f"  Net Market P&L (Spot+Perp): ${metrics['net_market_pnl']:.2f}")
    print(f"  Funding income: ${metrics['funding_pnl']:.2f}")
    print(f"  Fees paid: ${metrics['fees_paid']:.2f}")

    print(f"\nNet Market P&L Statistics:")
    print(f"  Volatility: ${metrics['net_pnl_volatility']:.2f}")
    print(f"  Max: ${metrics['net_pnl_max']:.2f}")
    print(f"  Min: ${metrics['net_pnl_min']:.2f}")
    print(f"  Range: ${metrics['net_pnl_range']:.2f}")

    print(f"\nBasis-Funding Relationship:")
    print(f"  Correlation: {metrics['basis_funding_correlation']:.4f}")
    print(f"  p-value: {metrics['basis_funding_p_value']:.4f}")

    print(f"\nReturn Metrics:")
    print(f"  Total return: {metrics['total_return_pct']:.2f}%")
    print(f"  Annualized return: {metrics['annualized_return']:.2f}%")
    print(f"  Funding APR: {metrics['funding_apr']:.2f}%")
    print(f"  Funding APY: {metrics['funding_apy']:.2f}%")
    print(
        f"  Avg funding rate per period: {metrics['avg_funding_rate']:.6f} ({metrics['avg_funding_apr']:.2f}% APR)"
    )
    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max drawdown: {metrics['max_drawdown']:.2f}%")

    print(f"\nCapital Efficiency:")
    print(f"  Initial notional: ${results['initial_notional']:.2f}")
    print(f"  Final notional: ${results['final_notional']:.2f}")
    print(f"  Capital efficiency ratio: {metrics['capital_efficiency']:.2f}x")

    # Print risk metrics summary
    print("\n=== RISK METRICS ===")

    # Health factor metrics
    if "min_health_factor" in metrics:
        print(f"  Minimum Health Factor: {metrics['min_health_factor']:.2f}")
        print(f"  Average Health Factor: {metrics['avg_health_factor']:.2f}")
        print(f"  Final Health Factor: {metrics['final_health_factor']:.2f}")
        print(f"  Time in High Risk Zone (<1.5): {metrics['pct_time_high_risk']:.2f}%")
        print(f"  Time in Safe Zone (>3.0): {metrics['pct_time_safe']:.2f}%")

    # Liquidation buffer metrics
    if "min_liquidation_buffer" in metrics:
        print(
            f"\n  Minimum Liquidation Buffer: {metrics['min_liquidation_buffer']:.2f}%"
        )
        print(f"  Average Liquidation Buffer: {metrics['avg_liquidation_buffer']:.2f}%")
        print(f"  Final Liquidation Buffer: {metrics['final_liquidation_buffer']:.2f}%")
        print(
            f"  Time in Danger Zone (<10% buffer): {metrics['pct_time_danger_zone']:.2f}%"
        )

    # Close calls
    if "liquidation_close_calls" in metrics:
        print(f"\n  Near-Liquidation Events: {metrics['liquidation_close_calls']}")

    # Capital allocation
    if "spot_allocation" in metrics and "perp_allocation" in metrics:
        initial_capital = results["initial_capital"]
        print(f"\n  Capital Allocation:")
        print(
            f"    Spot: ${metrics['spot_allocation']:.2f} ({metrics['spot_allocation']/initial_capital*100:.2f}% of capital)"
        )
        print(
            f"    Perp: ${metrics['perp_allocation']:.2f} ({metrics['perp_allocation']/initial_capital*100:.2f}% of capital)"
        )

    # Risk assessment
    if "final_health_factor" in metrics:
        print("\n  Risk Assessment:")
        if metrics["final_health_factor"] >= 3:
            print("    Position Status: VERY SAFE")
        elif metrics["final_health_factor"] >= 1.5:
            print("    Position Status: MODERATE RISK")
        elif metrics["final_health_factor"] >= 1:
            print("    Position Status: HIGH RISK")
        else:
            print("    Position Status: DANGER - LIQUIDATION IMMINENT")
