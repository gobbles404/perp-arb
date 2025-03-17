import numpy as np


def generate_signals(df, z_threshold=1.5):
    """
    Generate trading signals based on the funding-prompt spread z-score

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data and metrics
    z_threshold : float
        Z-score threshold for signal generation

    Returns:
    --------
    DataFrame with signal columns added
    """
    # Create a copy of the dataframe to avoid modifying the original
    strategy_df = df.copy()

    # Initialize signal columns
    strategy_df["signal"] = (
        0  # 1 for long perp/short prompt, -1 for short perp/long prompt, 0 for no position
    )
    strategy_df["position_perp"] = 0
    strategy_df["position_prompt"] = 0

    # Generate signals based on z-score thresholds
    # When funding rate is significantly higher than prompt APR (positive z-score)
    # Short perp (to receive funding) and long prompt (for delta neutrality)
    strategy_df.loc[
        strategy_df["funding_prompt_spread_zscore"] > z_threshold, "signal"
    ] = -1

    # When funding rate is significantly lower than prompt APR (negative z-score)
    # Long perp (to pay less funding) and short prompt (for delta neutrality)
    strategy_df.loc[
        strategy_df["funding_prompt_spread_zscore"] < -z_threshold, "signal"
    ] = 1

    return strategy_df


def calculate_positions(
    strategy_df, initial_capital=100000, leverage=3, max_allocation=0.8
):
    """
    Calculate position sizes based on signals

    Parameters:
    -----------
    strategy_df : pandas DataFrame
        DataFrame with signals
    initial_capital : float
        Initial capital in USD
    leverage : float
        Leverage multiplier
    max_allocation : float
        Maximum percentage of capital to allocate to the strategy

    Returns:
    --------
    DataFrame with position columns added
    """
    pos_df = strategy_df.copy()

    pos_df["position_perp"] = pos_df["position_perp"].astype(float)
    pos_df["position_prompt"] = pos_df["position_prompt"].astype(float)

    # Calculate available capital
    available_capital = initial_capital * max_allocation

    # Calculate position sizes (in USD notional)
    for i in range(1, len(pos_df)):
        # Default is to maintain previous day's position unless there's a signal change
        pos_df.loc[pos_df.index[i], "position_perp"] = pos_df.loc[
            pos_df.index[i - 1], "position_perp"
        ]
        pos_df.loc[pos_df.index[i], "position_prompt"] = pos_df.loc[
            pos_df.index[i - 1], "position_prompt"
        ]

        # If signal changes, update positions
        if (
            pos_df.loc[pos_df.index[i], "signal"]
            != pos_df.loc[pos_df.index[i - 1], "signal"]
        ):
            current_signal = pos_df.loc[pos_df.index[i], "signal"]

            if current_signal == 0:
                # Close positions
                pos_df.loc[pos_df.index[i], "position_perp"] = 0.0
                pos_df.loc[pos_df.index[i], "position_prompt"] = 0.0
            else:
                # Calculate position size in BTC
                perp_price = pos_df.loc[pos_df.index[i], "perp_close"]
                prompt_price = pos_df.loc[pos_df.index[i], "prompt_close"]

                # Use a position size that's proportional to capital and leverage
                # while maintaining delta neutrality
                position_size_usd = available_capital * leverage
                position_size_btc = position_size_usd / perp_price

                # Set position with sign based on the signal
                pos_df.loc[pos_df.index[i], "position_perp"] = (
                    -current_signal * position_size_btc
                )
                pos_df.loc[pos_df.index[i], "position_prompt"] = (
                    current_signal * position_size_btc
                )

    # Calculate position values in USD
    pos_df["position_perp_usd"] = pos_df["position_perp"] * pos_df["perp_close"]
    pos_df["position_prompt_usd"] = pos_df["position_prompt"] * pos_df["prompt_close"]
    pos_df["net_position_usd"] = (
        pos_df["position_perp_usd"] + pos_df["position_prompt_usd"]
    )

    return pos_df


def calculate_pnl(pos_df, trading_fee_pct=0.04, funding_interval=8):
    """
    Calculate P&L for the strategy

    Parameters:
    -----------
    pos_df : pandas DataFrame
        DataFrame with positions
    trading_fee_pct : float
        Trading fee as a percentage of position value
    funding_interval : int
        Number of hours between funding payments (8 for Binance)

    Returns:
    --------
    DataFrame with P&L columns added
    """
    pnl_df = pos_df.copy()

    # Initialize P&L columns
    pnl_df["trading_pnl_perp"] = 0.0
    pnl_df["trading_pnl_prompt"] = 0.0
    pnl_df["funding_pnl"] = 0.0
    pnl_df["fee_cost"] = 0.0
    pnl_df["daily_pnl"] = 0.0
    pnl_df["cumulative_pnl"] = 0.0

    # Calculate P&L components
    for i in range(1, len(pnl_df)):
        prev_idx = pnl_df.index[i - 1]
        curr_idx = pnl_df.index[i]

        # Trading P&L: price change * position
        prev_perp_pos = pnl_df.loc[prev_idx, "position_perp"]
        curr_perp_pos = pnl_df.loc[curr_idx, "position_perp"]
        prev_prompt_pos = pnl_df.loc[prev_idx, "position_prompt"]
        curr_prompt_pos = pnl_df.loc[curr_idx, "position_prompt"]

        # P&L from holding positions
        price_change_perp = (
            pnl_df.loc[curr_idx, "perp_close"] - pnl_df.loc[prev_idx, "perp_close"]
        )
        price_change_prompt = (
            pnl_df.loc[curr_idx, "prompt_close"] - pnl_df.loc[prev_idx, "prompt_close"]
        )

        pnl_df.loc[curr_idx, "trading_pnl_perp"] = prev_perp_pos * price_change_perp
        pnl_df.loc[curr_idx, "trading_pnl_prompt"] = (
            prev_prompt_pos * price_change_prompt
        )

        # Funding P&L (only for perpetual futures)
        # Simplified: position * funding rate * price * days
        # Negative position (short) + positive funding rate = receive funding
        funding_rate = pnl_df.loc[prev_idx, "funding_rate"]
        funding_days = 1  # Daily data
        funding_payments_per_day = (
            24 / funding_interval
        )  # Typically 3 payments per day on Binance

        pnl_df.loc[curr_idx, "funding_pnl"] = (
            -prev_perp_pos
            * funding_rate
            * pnl_df.loc[prev_idx, "perp_close"]
            * funding_days
            * funding_payments_per_day
        )

        # Trading fees: only calculated when position changes
        perp_pos_change = abs(curr_perp_pos - prev_perp_pos)
        prompt_pos_change = abs(curr_prompt_pos - prev_prompt_pos)

        perp_trade_value = perp_pos_change * pnl_df.loc[curr_idx, "perp_close"]
        prompt_trade_value = prompt_pos_change * pnl_df.loc[curr_idx, "prompt_close"]

        pnl_df.loc[curr_idx, "fee_cost"] = (perp_trade_value + prompt_trade_value) * (
            trading_fee_pct / 100
        )

        # Daily total P&L
        pnl_df.loc[curr_idx, "daily_pnl"] = (
            pnl_df.loc[curr_idx, "trading_pnl_perp"]
            + pnl_df.loc[curr_idx, "trading_pnl_prompt"]
            + pnl_df.loc[curr_idx, "funding_pnl"]
            - pnl_df.loc[curr_idx, "fee_cost"]
        )

        # Cumulative P&L
        pnl_df.loc[curr_idx, "cumulative_pnl"] = (
            pnl_df.loc[prev_idx, "cumulative_pnl"] + pnl_df.loc[curr_idx, "daily_pnl"]
        )

    return pnl_df


def calculate_performance_metrics(pnl_df, risk_free_rate=0.03):
    """
    Calculate performance metrics for the strategy

    Parameters:
    -----------
    pnl_df : pandas DataFrame
        DataFrame with P&L data
    risk_free_rate : float
        Annual risk-free rate

    Returns:
    --------
    dict with performance metrics
    """
    # Filter to get days where we had a position
    active_days = pnl_df[
        (pnl_df["position_perp"] != 0) | (pnl_df["position_prompt"] != 0)
    ].copy()

    if len(active_days) == 0:
        return {"error": "No active trading days found"}

    # Calculate daily returns (as percentage of the net position value)
    active_days["daily_return_pct"] = (
        100 * active_days["daily_pnl"] / active_days["net_position_usd"].abs()
    )

    # Calculate performance metrics
    total_days = len(active_days)
    total_pnl = pnl_df["cumulative_pnl"].iloc[-1]
    max_drawdown_pct = (
        (pnl_df["cumulative_pnl"].cummax() - pnl_df["cumulative_pnl"]).max()
        / pnl_df["cumulative_pnl"].cummax().max()
        * 100
        if pnl_df["cumulative_pnl"].cummax().max() > 0
        else 0
    )

    # Annualized metrics
    daily_returns = active_days["daily_return_pct"]
    annual_return = daily_returns.mean() * 252  # Annualized return
    annual_volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = (
        (annual_return - risk_free_rate) / annual_volatility
        if annual_volatility > 0
        else 0
    )

    # Profit factor
    winning_days = active_days[active_days["daily_pnl"] > 0]
    losing_days = active_days[active_days["daily_pnl"] < 0]
    profit_factor = (
        abs(winning_days["daily_pnl"].sum() / losing_days["daily_pnl"].sum())
        if losing_days["daily_pnl"].sum() != 0
        else float("inf")
    )

    # Win rate
    win_rate = len(winning_days) / total_days if total_days > 0 else 0

    # Source breakdown
    total_trading_pnl = (
        active_days["trading_pnl_perp"].sum() + active_days["trading_pnl_prompt"].sum()
    )
    total_funding_pnl = active_days["funding_pnl"].sum()
    total_fee_cost = active_days["fee_cost"].sum()

    # Return metrics dictionary
    return {
        "total_trading_days": total_days,
        "total_pnl": total_pnl,
        "annual_return_pct": annual_return,
        "annual_volatility_pct": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "trading_pnl": total_trading_pnl,
        "funding_pnl": total_funding_pnl,
        "fee_cost": total_fee_cost,
    }
