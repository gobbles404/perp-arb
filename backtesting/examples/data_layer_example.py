# perp_arb/backtesting/examples/data_layer_example.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

# Import data layer components using relative imports
from ..data.loaders import DataManager
from ..data.market_data import MarketData
from ..data.contract_specs import ContractSpecificationRegistry
from ..utils.helpers import (
    get_market_data_path,
    get_contract_specs_path,
    calculate_zscore,
    save_results_to_csv,
)
from ..logger import get_logger

# Set up logger
logger = get_logger(__name__)


def main():
    """
    Example script demonstrating usage of the data layer components.
    """
    # Set up paths - using relative paths from the examples directory
    market_data_dir = os.path.join(
        "..", "..", "binance_data_pipeline", "data", "markets"
    )
    contract_specs_file = os.path.join(
        "..", "..", "binance_data_pipeline", "data", "contracts", "fut_specs.csv"
    )

    logger.info(f"Loading data from: {market_data_dir}")
    logger.info(f"Loading contract specs from: {contract_specs_file}")

    # Initialize data manager
    data_manager = DataManager(market_data_dir, contract_specs_file)

    # Get available symbols
    symbols = data_manager.get_available_symbols()
    logger.info(f"Available symbols: {symbols}")

    if not symbols:
        logger.error("No symbols found. Please check the market data directory.")
        return

    # Use first symbol as example
    symbol = symbols[0]
    logger.info(f"\nAnalyzing symbol: {symbol}")

    # Load data with contract specs
    market_data, contract_specs = data_manager.load_data_with_specs(symbol)

    # Log market data summary
    logger.info("\nMarket Data Summary:")
    logger.info(f"Time period: {market_data.index.min()} to {market_data.index.max()}")
    logger.info(f"Number of data points: {len(market_data)}")
    logger.info(f"Columns: {market_data.columns.tolist()}")

    # Log contract specs
    logger.info("\nContract Specifications:")
    if contract_specs:
        for key, value in contract_specs.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.warning("  No contract specifications found for this symbol")

    # Create MarketData object
    market = MarketData(symbol, market_data, contract_specs)

    # Calculate Z-score
    window_size = 20
    market_data["zscore"] = market.calculate_rolling_zscore(window=window_size)

    # Get margin requirements
    margin_reqs = market.get_margin_requirements()
    logger.info(f"\nMargin Requirements:")
    logger.info(f"  Initial Margin: {margin_reqs['initial_margin'] * 100:.2f}%")
    logger.info(f"  Maintenance Margin: {margin_reqs['maintenance_margin'] * 100:.2f}%")

    # Generate a simple plot of basis and Z-score
    logger.info("Generating plots...")
    plt.figure(figsize=(12, 8))

    # Plot basis
    ax1 = plt.subplot(211)
    ax1.plot(market_data.index, market_data["basis"], label="Basis (%)")
    ax1.set_title(f"{symbol} Basis (Perp Premium/Discount)")
    ax1.set_ylabel("Basis (%)")
    ax1.legend()
    ax1.grid(True)

    # Plot Z-score
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(
        market_data.index, market_data["zscore"], label=f"Z-score ({window_size}-day)"
    )
    ax2.axhline(y=2, color="r", linestyle="--", alpha=0.5)
    ax2.axhline(y=-2, color="r", linestyle="--", alpha=0.5)
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.2)
    ax2.set_title(f"{symbol} Basis Z-score")
    ax2.set_ylabel("Z-score")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"{symbol}_basis_analysis.png")
    logger.info(f"Saved plot to {symbol}_basis_analysis.png")

    # Save processed data to CSV
    output_file = save_results_to_csv(
        market_data, f"{symbol}_processed_data", "results"
    )
    logger.info(f"Saved processed data to {output_file}")


if __name__ == "__main__":
    main()
