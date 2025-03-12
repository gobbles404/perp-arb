import os
from dotenv import load_dotenv
from binance.client import Client

# Load API keys from .env
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# Initialize Binance client
client = Client(API_KEY, SECRET_KEY)
