import os
import requests
import time
import hmac
import hashlib

# Load API keys from environment variables
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

BASE_URL = "https://api.binance.com"


def generate_signature(params, secret_key):
    """Generate HMAC SHA256 signature."""
    query_string = "&".join([f"{key}={value}" for key, value in params.items()])
    return hmac.new(
        secret_key.encode(), query_string.encode(), hashlib.sha256
    ).hexdigest()


def fetch_usdt_interest_rate():
    """Fetch historical USDT interest rate from Binance."""
    endpoint = "/sapi/v1/margin/interestRateHistory"
    params = {
        "asset": "USDT",
        "startTime": int(time.time() * 1000) - (7 * 24 * 60 * 60 * 1000),  # Last 7 days
        "endTime": int(time.time() * 1000),
        "limit": 100,  # Max records
        "timestamp": int(time.time() * 1000),
    }

    # Generate signature
    params["signature"] = generate_signature(params, SECRET_KEY)

    headers = {"X-MBX-APIKEY": API_KEY}
    response = requests.get(BASE_URL + endpoint, params=params, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


# Fetch and print interest rate history
data = fetch_usdt_interest_rate()
if data:
    for entry in data:
        print(f"Time: {entry['timestamp']}, Rate: {entry['dailyInterestRate']}")
