# Funding Arbitrage Strategy Framework

This project implements a modular framework for backtesting funding arbitrage strategies on cryptocurrency perpetual and spot markets.

## Project Structure

```
strategy/
│
├── data/
│   └── data_loader.py        # Functions for loading and preprocessing data
│
├── models/
│   └── funding_arb.py        # Core strategy class implementation
│
├── analytics/
│   ├── metrics.py            # Performance metric calculations
│   └── visualizations.py     # Plotting and visualization functions
│
├── backtesting/
│   └── backtester.py         # Backtesting engine
│
├── utils/
│   ├── config.py             # Configuration settings
│   └── helpers.py            # Utility functions
│
└── scripts/
    └── run_backtest.py       # Main runner script
```

## How It Works

The framework follows a modular design pattern that separates concerns:

1. **Data Handling**: `data_loader.py` contains functions for loading, filtering, and preprocessing market data.

2. **Strategy Logic**: `funding_arb.py` implements the core strategy logic through a class that handles position initialization, PnL calculation, and position closing.

3. **Backtesting**: `backtester.py` provides a simulation engine that runs the strategy on historical data and records results.

4. **Analytics**:

   - `metrics.py` calculates performance metrics from backtest results.
   - `visualizations.py` generates charts and visual analysis of the strategy.

5. **Utilities**:

   - `config.py` centralizes configuration settings.
   - `helpers.py` provides common utility functions.

6. **Execution**: `run_backtest.py` ties everything together and provides a command-line interface.

## Running the Backtest

To run a backtest:

```bash
python strategy/scripts/run_backtest.py
```

Optional parameters:

```bash
python strategy/scripts/run_backtest.py --data-path data/markets/BTCUSDT_8h.csv --capital 50000 --leverage 3.0 --start-date 2023-01-01 --end-date 2023-06-30
```

## Benefits of Modular Design

This modular architecture provides several advantages:

1. **Separation of Concerns**: Each module has a specific responsibility, making the code more organized and easier to maintain.

2. **Reusability**: Components can be reused in different contexts. For example, the same visualization functions can be used for various strategy types.

3. **Testability**: Isolated modules are easier to test individually.

4. **Extensibility**: New strategies, metrics, or visualizations can be added without modifying existing code.

5. **Collaboration**: Team members can work on different components simultaneously with minimal conflicts.

## Extending the Framework

To add a new strategy:

1. Create a new strategy class in the `models` directory.
2. Implement the required methods: `initialize_position`, `calculate_pnl`, and `close_position`.
3. Update the runner script to use your new strategy.

To add new visualizations:

1. Add new plot functions to `visualizations.py`.
2. Update the `create_performance_charts` function to include your new plots.

## Next Steps

Potential enhancements to this framework include:

1. Support for dynamic position sizing and multiple entry/exit points
2. Portfolio of multiple strategies
3. Risk management modules
4. Integration with live trading APIs
5. Optimization capabilities for strategy parameters
