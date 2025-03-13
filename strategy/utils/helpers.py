"""
Utility functions for the trading strategy.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import json


def ensure_dir_exists(directory):
    """Ensure the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory


def get_timestamp():
    """Get current timestamp in the standard format."""
    from config import TIMESTAMP_FORMAT

    return datetime.now().strftime(TIMESTAMP_FORMAT)


def save_results_to_json(results, metrics, output_dir, prefix="backtest"):
    """Save backtest results and metrics to JSON files."""
    # Ensure directory exists
    ensure_dir_exists(output_dir)

    # Create timestamp for filenames
    timestamp = get_timestamp()

    # Save metrics (convert any numpy types to Python native types)
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            serializable_metrics[k] = v.item()
        elif isinstance(v, np.ndarray):
            serializable_metrics[k] = v.tolist()
        else:
            serializable_metrics[k] = v

    metrics_file = os.path.join(output_dir, f"{prefix}_metrics_{timestamp}.json")
    with open(metrics_file, "w") as f:
        json.dump(serializable_metrics, f, indent=4)

    # Save summary results (exclude large data arrays)
    summary_results = {
        "entry_date": (
            results["entry_date"].strftime("%Y-%m-%d")
            if hasattr(results["entry_date"], "strftime")
            else results["entry_date"]
        ),
        "exit_date": (
            results["exit_date"].strftime("%Y-%m-%d")
            if hasattr(results["exit_date"], "strftime")
            else results["exit_date"]
        ),
        "initial_capital": results["initial_capital"],
        "final_capital": results["final_capital"],
        "entry_fee": results["entry_fee"],
        "exit_fee": results["exit_fee"],
        "spot_quantity": results["spot_quantity"],
        "perp_quantity": results["perp_quantity"],
        "initial_notional": results["initial_notional"],
        "final_notional": results["final_notional"],
    }

    summary_file = os.path.join(output_dir, f"{prefix}_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary_results, f, indent=4)

    print(f"Results saved to:\n  - {metrics_file}\n  - {summary_file}")

    return metrics_file, summary_file


def annualize_rate(rate, periods_per_year):
    """Convert a periodic rate to an annual rate."""
    return rate * periods_per_year


def compound_rate(daily_rate, days_per_year=365):
    """Calculate compounded annual rate from a daily rate."""
    return ((1 + daily_rate) ** days_per_year) - 1


def clean_nan_values(data):
    """Replace NaN values with None for JSON serialization."""
    if isinstance(data, dict):
        return {k: clean_nan_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_nan_values(x) for x in data]
    elif isinstance(data, float) and np.isnan(data):
        return None
    else:
        return data
