# binance_data_pipeline/core/client.py
import os
from binance.client import Client
from dotenv import load_dotenv
from .logger import get_logger

logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()


class BinanceClient:
    """Singleton manager for Binance API client."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BinanceClient, cls).__new__(cls)
            cls._initialize()
        return cls._instance

    @classmethod
    def _initialize(cls):
        """Initialize the Binance client with API credentials."""
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_SECRET_KEY")

        if not api_key or not api_secret:
            logger.warning("Missing API credentials. Check your .env file.")

        try:
            cls.client = Client(api_key, api_secret)
            logger.info("Binance API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Binance API client: {e}")
            raise


# Create global client instance
binance_client = BinanceClient()
client = binance_client.client
