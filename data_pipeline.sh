#!/bin/bash
# data_pipeline.sh - Runs the complete data pipeline

# Optional parameters with defaults
SYMBOL=${1:-BTCUSDT}
INTERVALS=${2:-"1d 8h 1h"}

echo "Running data pipeline for $SYMBOL"

# Step 1: Build futures curve
echo "Building futures curve..."
# Convert space-separated intervals to comma-separated for Python script
COMMA_INTERVALS=$(echo $INTERVALS | tr ' ' ',')
python scripts/process/build_futures_curve.py --symbol $SYMBOL --intervals $COMMA_INTERVALS

# Step 2: Build market data for each interval
echo "Building market data..."
for interval in $INTERVALS; do
    echo "Processing interval: $interval"
    python scripts/process/build_market.py --symbol $SYMBOL --interval $interval
done

echo "Pipeline complete!"