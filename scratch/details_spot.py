from binance_client import client  # ✅ Use shared client
import pprint

# Define symbol
symbol = "BTCUSDT"


def fetch_spot_details(symbol):
    """Fetch detailed price data for BTC Spot Market and dump full JSON."""

    # Get Last Price & Bid/Ask Prices
    ticker_data = client.get_ticker(symbol=symbol)

    # Get Order Book (Depth Snapshot)
    # order_book_data = client.get_order_book(symbol=symbol, limit=5)

    # Get Recent Trades
    # trades_data = client.get_recent_trades(symbol=symbol, limit=5)

    # Get Kline Data (Last 10 Candles)
    # klines_data = client.get_klines(symbol=symbol, interval="1h", limit=10)

    # Get Weighted Average Price
    avg_price_data = client.get_avg_price(symbol=symbol)

    # Print all data in full JSON format
    print("\n--- Ticker Data (Last Price, High/Low, 24h Stats) ---")
    pprint.pprint(ticker_data)

    # print("\n--- Order Book Data (Top 5 Levels) ---")
    # pprint.pprint(order_book_data)

    # print("\n--- Recent Trades (Last 5) ---")
    # pprint.pprint(trades_data)

    # print("\n--- Kline Data (Last 10 Candles) ---")
    # pprint.pprint(klines_data)

    print("\n--- Weighted Average Price ---")
    pprint.pprint(avg_price_data)


if __name__ == "__main__":
    fetch_spot_details(symbol)
