"""
Utility functions for working with vectorbt in the trading strategy framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vectorbt as vbt


def calculate_vectorbt_metrics(portfolio):
    """
    Calculate additional performance metrics using vectorbt.

    Parameters:
    ----------
    portfolio : vbt.Portfolio
        VectorBT portfolio object

    Returns:
    -------
    dict
        Dictionary of performance metrics
    """
    # Get basic stats
    stats = portfolio.stats()

    # Calculate more advanced metrics
    metrics = {
        "total_return": float(stats["total_return"]),
        "total_return_pct": float(stats["total_return"]) * 100,
        "annualized_return": float(stats["total_return_ann"]) * 100,
        "max_drawdown": float(stats["max_dd"]) * 100,
        "sharpe_ratio": float(stats["sharpe_ratio"]),
        "sortino_ratio": float(stats["sortino_ratio"]),
        "calmar_ratio": float(stats["calmar_ratio"]),
        "volatility_ann": float(stats["volatility_ann"]) * 100,
        "avg_win": float(stats.get("avg_win", 0)),
        "avg_loss": float(stats.get("avg_loss", 0)),
        "win_rate": float(stats.get("win_rate", 0)),
        "profit_factor": float(stats.get("profit_factor", 0)),
    }

    return metrics


def create_vectorbt_performance_plots(
    portfolio, title="Strategy Performance", filename=None
):
    """
    Create performance plots using vectorbt.

    Parameters:
    ----------
    portfolio : vbt.Portfolio
        VectorBT portfolio object
    title : str
        Title for the plots
    filename : str, optional
        If provided, save the plots to this file

    Returns:
    -------
    plt.Figure
        Matplotlib figure with plots
    """
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 20))

    # Plot 1: Cumulative returns
    portfolio.plot_cum_returns(title=f"{title} - Cumulative Returns", ax=axs[0])

    # Plot 2: Drawdowns
    portfolio.plot_drawdowns(title=f"{title} - Drawdowns", ax=axs[1])

    # Plot 3: Monthly returns
    if len(portfolio.returns) > 30:  # Only if we have enough data
        portfolio.plot_monthly_returns(title=f"{title} - Monthly Returns", ax=axs[2])
    else:
        # If not enough data, plot daily returns
        portfolio.plot_returns(title=f"{title} - Daily Returns", ax=axs[2])

    plt.tight_layout()

    # Save figure if filename is provided
    if filename:
        plt.savefig(filename)

    return fig


def analyze_funding_rate_impact(results):
    """
    Analyze the impact of funding rate on strategy performance.

    Parameters:
    ----------
    results : dict
        Results dictionary from the backtester

    Returns:
    -------
    dict
        Dictionary of funding rate analysis
    """
    # Convert to pandas Series for easier analysis
    funding_rates = pd.Series(results["funding_rates"])
    equity_curve = pd.Series(results["equity_curve"])
    funding_income = pd.Series(results["funding_income"])
    cumulative_funding = pd.Series(results["cumulative_funding"])

    # Calculate equity returns
    equity_returns = equity_curve.pct_change().fillna(0)

    # Calculate correlation between funding rate and returns
    funding_return_corr = funding_rates.corr(equity_returns)

    # Calculate percentage of funding income in total profit
    total_profit = equity_curve.iloc[-1] - equity_curve.iloc[0]
    total_funding = cumulative_funding.iloc[-1]
    funding_contribution = (
        (total_funding / total_profit) * 100 if total_profit != 0 else 0
    )

    # Calculate average funding rate during profitable vs. unprofitable days
    profitable_days = equity_returns > 0
    unprofitable_days = equity_returns < 0

    avg_funding_profitable = (
        funding_rates[profitable_days].mean()
        if len(funding_rates[profitable_days]) > 0
        else 0
    )
    avg_funding_unprofitable = (
        funding_rates[unprofitable_days].mean()
        if len(funding_rates[unprofitable_days]) > 0
        else 0
    )

    # Calculate annualized funding rates
    avg_funding_profitable_ann = (
        avg_funding_profitable * 3 * 365 * 100 if avg_funding_profitable else 0
    )
    avg_funding_unprofitable_ann = (
        avg_funding_unprofitable * 3 * 365 * 100 if avg_funding_unprofitable else 0
    )

    analysis = {
        "funding_return_correlation": funding_return_corr,
        "funding_contribution_pct": funding_contribution,
        "avg_funding_profitable_days": avg_funding_profitable,
        "avg_funding_unprofitable_days": avg_funding_unprofitable,
        "avg_funding_profitable_ann": avg_funding_profitable_ann,
        "avg_funding_unprofitable_ann": avg_funding_unprofitable_ann,
        "positive_funding_days": (funding_rates > 0).sum(),
        "negative_funding_days": (funding_rates < 0).sum(),
        "zero_funding_days": (funding_rates == 0).sum(),
        "max_funding_rate": funding_rates.max(),
        "min_funding_rate": funding_rates.min(),
        "max_funding_income": funding_income.max(),
        "min_funding_income": funding_income.min(),
        "total_funding_income": total_funding,
    }

    return analysis


def analyze_market_conditions(results):
    """
    Analyze strategy performance under different market conditions.

    Parameters:
    ----------
    results : dict
        Results dictionary from the backtester

    Returns:
    -------
    dict
        Dictionary of market condition analysis
    """
    # Create DataFrame for analysis
    df = pd.DataFrame(
        {
            "equity": results["equity_curve"],
            "spot_price": results["data"]["spot_close"],
            "perp_price": results["data"]["perp_close"],
            "funding_rate": results["funding_rates"],
            "date": results["dates"],
        }
    )

    # Set date as index
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Calculate daily market returns
    df["market_return"] = df["spot_price"].pct_change().fillna(0)
    df["strategy_return"] = pd.Series(results["equity_curve"]).pct_change().fillna(0)

    # Classify market conditions
    df["market_condition"] = "neutral"
    df.loc[df["market_return"] > 0.01, "market_condition"] = "bull"
    df.loc[df["market_return"] < -0.01, "market_condition"] = "bear"

    # Calculate metrics by market condition
    metrics_by_condition = {}
    for condition in ["bull", "bear", "neutral"]:
        condition_mask = df["market_condition"] == condition
        if condition_mask.sum() > 0:
            condition_returns = df.loc[condition_mask, "strategy_return"]
            condition_funding = df.loc[condition_mask, "funding_rate"]

            metrics_by_condition[condition] = {
                "days": int(condition_mask.sum()),
                "avg_return": float(condition_returns.mean() * 100),
                "total_return": float(((1 + condition_returns).prod() - 1) * 100),
                "volatility": float(condition_returns.std() * 100),
                "max_daily_gain": float(condition_returns.max() * 100),
                "max_daily_loss": float(condition_returns.min() * 100),
                "avg_funding_rate": float(condition_funding.mean()),
                "avg_funding_ann": float(condition_funding.mean() * 3 * 365 * 100),
            }

    return metrics_by_condition
