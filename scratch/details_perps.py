from binance_client import client  # ✅ Use shared client
import pprint

# Define symbol
symbol = "BTCUSDT"


def fetch_perp_details(symbol):
    """Fetch detailed price data for BTC Perpetual Futures and dump full JSON."""

    # Get Mark Price (Mark & Index Prices)
    mark_price_data = client.futures_mark_price(symbol=symbol)

    # demo = client.fundingInfo(symbol=symbol)
    # print(demo)

    # Get Funding Rate & Next Funding Time
    # funding_data = client.futures_funding_rate(symbol=symbol, limit=10)

    # Get Open Interest
    open_interest_data = client.futures_open_interest(symbol=symbol)

    # Get Ticker Price (Last Price & Bid/Ask)
    ticker_data = client.futures_symbol_ticker(symbol=symbol)

    # Get Order Book (Depth Snapshot)
    order_book_data = client.futures_order_book(symbol=symbol, limit=5)

    # Get Premium Index Kline Data
    premium_index_klines_data = client.futures_premium_index_klines(
        symbol=symbol, interval="1h", limit=4
    )

    # Get Funding Info (V1)
    funding_info_data = client.futures_v1_get_funding_info(symbol=symbol)
    # Get COIN-M Funding Rate (Only valid for COIN-M contracts)
    # coin_funding_rate = None
    # try:
    # coin_funding_rate = client.futures_coin_funding_rate(symbol=symbol, limit=10)
    # except Exception as e:
    # print(f"\n⚠️ Error fetching COIN-M Funding Rate: {e}")

    # Print all data in full JSON format
    print("\n--- Mark Price Data ---")
    pprint.pprint(mark_price_data)

    # print("\n--- Funding Rate Data ---")
    # pprint.pprint(funding_data)

    print("\n--- Open Interest Data ---")
    pprint.pprint(open_interest_data)

    print("\n--- Ticker Price Data ---")
    pprint.pprint(ticker_data)

    print("\n--- Order Book Data (Top 5 Levels) ---")
    pprint.pprint(order_book_data)

    print("\n--- Premium Index Kline Data ---")
    pprint.pprint(premium_index_klines_data)

    # print("\n--- Funding Info (V1) ---")
    # pprint.pprint(funding_info_data)

    # if coin_funding_rate:
    #    print("\n--- COIN-M Funding Rate ---")
    # pprint.pprint(coin_funding_rate)


if __name__ == "__main__":
    fetch_perp_details(symbol)
