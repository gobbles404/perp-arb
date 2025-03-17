# perp_arb/backtesting/test_data_layer.py
import os
import sys
import pandas as pd

# Add the parent directory to the path so we can use absolute imports
sys.path.insert(0, os.path.abspath(".."))

# Use absolute imports
from backtesting.data.loaders import (
    MarketDataLoader,
    ContractSpecsLoader,
    DataManager,
)
from backtesting.data.market_data import MarketData
from backtesting.data.contract_specs import ContractSpecificationRegistry
from backtesting.utils.helpers import get_available_symbols
from core.logger import get_logger

# Set up logger
logger = get_logger(__name__)


def test_market_data_loader():
    """Test the MarketDataLoader class."""
    # Define paths - using relative paths from the backtesting directory
    market_data_dir = "../binance_data_pipeline/data/markets"

    # Check directory exists
    if not os.path.exists(market_data_dir):
        logger.error(f"Market data directory not found: {market_data_dir}")
        return False

    # Find a CSV file to test with
    csv_files = [f for f in os.listdir(market_data_dir) if f.endswith(".csv")]
    if not csv_files:
        logger.error(f"No CSV files found in {market_data_dir}")
        return False

    # Use first CSV file
    test_file = os.path.join(market_data_dir, csv_files[0])
    symbol = csv_files[0].split("_")[0]

    logger.info(f"Testing MarketDataLoader with {test_file}")

    try:
        # Create loader
        loader = MarketDataLoader(test_file, symbol)

        # Load data
        data = loader.load()
        logger.info(f"Successfully loaded data with {len(data)} rows")

        # Preprocess data
        processed_data = loader.preprocess()
        logger.info(
            f"Successfully preprocessed data with columns: {processed_data.columns.tolist()}"
        )

        # Get events
        events = loader.get_events()
        logger.info(f"Successfully converted to {len(events)} events")

        return True
    except Exception as e:
        logger.error(f"Error testing MarketDataLoader: {str(e)}")
        return False


def test_contract_specs_loader():
    """Test the ContractSpecsLoader class."""
    # Define path - using relative path from the backtesting directory
    contract_specs_file = "../binance_data_pipeline/data/contracts/fut_specs.csv"

    # Check file exists
    if not os.path.exists(contract_specs_file):
        logger.error(f"Contract specs file not found: {contract_specs_file}")
        return False

    logger.info(f"Testing ContractSpecsLoader with {contract_specs_file}")

    try:
        # Create loader
        loader = ContractSpecsLoader(contract_specs_file)

        # Load data
        data = loader.load()
        logger.info(f"Successfully loaded contract specs with {len(data)} rows")

        # Get specs by symbol
        specs_by_symbol = loader.get_specs_by_symbol()
        logger.info(f"Successfully converted to {len(specs_by_symbol)} specs by symbol")

        # Get first symbol
        if specs_by_symbol:
            first_symbol = next(iter(specs_by_symbol))
            specs = loader.get_specs(first_symbol)
            logger.info(f"Specs for {first_symbol}: {specs}")

        return True
    except Exception as e:
        logger.error(f"Error testing ContractSpecsLoader: {str(e)}")
        return False


def test_contract_specification_registry():
    """Test the ContractSpecificationRegistry class."""
    # Define path - using relative path from the backtesting directory
    contract_specs_file = "../binance_data_pipeline/data/contracts/fut_specs.csv"

    # Check file exists
    if not os.path.exists(contract_specs_file):
        logger.error(f"Contract specs file not found: {contract_specs_file}")
        return False

    logger.info(f"Testing ContractSpecificationRegistry with {contract_specs_file}")

    try:
        # Create registry
        registry = ContractSpecificationRegistry(contract_specs_file)

        # List active contracts
        active_contracts = registry.list_active_contracts()
        logger.info(f"Found {len(active_contracts)} active contracts")

        # List perpetuals
        perpetuals = registry.list_perpetuals()
        logger.info(f"Found {len(perpetuals)} perpetual contracts")

        # List futures
        futures = registry.list_futures()
        logger.info(f"Found {len(futures)} futures contracts")

        # If any contracts found, test getting margin requirements
        if active_contracts:
            first_symbol = active_contracts[0].symbol
            margin_reqs = registry.get_margin_requirements(first_symbol)
            logger.info(f"Margin requirements for {first_symbol}: {margin_reqs}")

            liquidation_fee = registry.get_liquidation_fee(first_symbol)
            logger.info(f"Liquidation fee for {first_symbol}: {liquidation_fee:.2%}")

        return True
    except Exception as e:
        logger.error(f"Error testing ContractSpecificationRegistry: {str(e)}")
        return False


def test_data_manager():
    """Test the DataManager class."""
    # Define paths - using relative paths from the backtesting directory
    market_data_dir = "../binance_data_pipeline/data/markets"
    contract_specs_file = "../binance_data_pipeline/data/contracts/fut_specs.csv"

    # Check paths exist
    if not os.path.exists(market_data_dir):
        logger.error(f"Market data directory not found: {market_data_dir}")
        return False
    if not os.path.exists(contract_specs_file):
        logger.error(f"Contract specs file not found: {contract_specs_file}")
        return False

    logger.info(f"Testing DataManager with {market_data_dir} and {contract_specs_file}")

    try:
        # Create data manager
        manager = DataManager(market_data_dir, contract_specs_file)

        # Get available symbols
        symbols = manager.get_available_symbols()
        visible_symbols = symbols[:5] if len(symbols) > 5 else symbols
        logger.info(f"Found {len(symbols)} available symbols: {visible_symbols}...")

        # If any symbols found, test loading data
        if symbols:
            test_symbol = symbols[0]

            # Get market data loader
            loader = manager.get_market_data_loader(test_symbol)
            logger.info(f"Successfully created market data loader for {test_symbol}")

            # Load data with specs
            market_data, contract_specs = manager.load_data_with_specs(test_symbol)
            logger.info(
                f"Successfully loaded data for {test_symbol} with {len(market_data)} rows"
            )

            if contract_specs:
                logger.info(f"Contract specs found for {test_symbol}")
            else:
                logger.warning(f"No contract specs found for {test_symbol}")

            # Test loading all symbols
            all_data = manager.load_all_symbols()
            logger.info(f"Successfully loaded data for all {len(all_data)} symbols")

        return True
    except Exception as e:
        logger.error(f"Error testing DataManager: {str(e)}")
        return False


def test_market_data_class():
    """Test the MarketData class."""
    # Define paths - using relative paths from the backtesting directory
    market_data_dir = "../binance_data_pipeline/data/markets"
    contract_specs_file = "../binance_data_pipeline/data/contracts/fut_specs.csv"

    # Create data manager
    manager = DataManager(market_data_dir, contract_specs_file)

    # Get available symbols
    symbols = manager.get_available_symbols()
    if not symbols:
        logger.error("No symbols found")
        return False

    test_symbol = symbols[0]
    logger.info(f"Testing MarketData class with {test_symbol}")

    try:
        # Load data with specs
        market_data, contract_specs = manager.load_data_with_specs(test_symbol)

        # Create MarketData object
        market = MarketData(test_symbol, market_data, contract_specs)

        # Test methods
        close_prices = market.get_close_prices()
        logger.info(f"Successfully got close prices with shape {close_prices.shape}")

        spot_data = market.get_price_data("spot")
        logger.info(f"Successfully got spot OHLC data with shape {spot_data.shape}")

        basis = market.get_basis()
        logger.info(f"Successfully got basis with length {len(basis)}")

        funding_rates = market.get_funding_rates()
        logger.info(f"Successfully got funding rates with length {len(funding_rates)}")

        z_score = market.calculate_rolling_zscore(window=20)
        logger.info(f"Successfully calculated Z-score with length {len(z_score)}")

        margin_reqs = market.get_margin_requirements()
        logger.info(f"Margin requirements: {margin_reqs}")

        return True
    except Exception as e:
        logger.error(f"Error testing MarketData class: {str(e)}")
        return False


def main():
    """Run all tests."""
    logger.info("==== Testing Data Layer Components ====\n")

    # Run tests
    tests = [
        ("MarketDataLoader", test_market_data_loader),
        ("ContractSpecsLoader", test_contract_specs_loader),
        ("ContractSpecificationRegistry", test_contract_specification_registry),
        ("DataManager", test_data_manager),
        ("MarketData", test_market_data_class),
    ]

    results = {}
    for name, test_func in tests:
        logger.info(f"\n--- Testing {name} ---")
        result = test_func()
        results[name] = result
        logger.info(f"--- {name} test {'PASSED' if result else 'FAILED'} ---")

    # Print summary
    logger.info("\n==== Test Summary ====")
    all_passed = True
    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        if not result:
            all_passed = False
        logger.info(f"{name}: {status}")

    logger.info(f"\nOverall: {'PASSED' if all_passed else 'FAILED'}")


if __name__ == "__main__":
    main()
