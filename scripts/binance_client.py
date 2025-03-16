import os
import sys
from dotenv import load_dotenv
from binance.client import Client

# Add path for logging import if this is run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_config import get_logger

logger = get_logger(__name__)

# Load API keys from .env
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# Initialize Binance client
try:
    client = Client(API_KEY, SECRET_KEY)
    logger.info("Binance API client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Binance API client: {e}")
    raise
