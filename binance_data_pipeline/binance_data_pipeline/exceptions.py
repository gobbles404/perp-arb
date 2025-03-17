# binance_data_pipeline/exceptions.py
class BinanceDataPipelineError(Exception):
    """Base exception for all pipeline errors."""

    pass


class FetcherError(BinanceDataPipelineError):
    """Exception raised for errors in the data fetching process."""

    pass


class ProcessorError(BinanceDataPipelineError):
    """Exception raised for errors in the data processing."""

    pass


class ConfigurationError(BinanceDataPipelineError):
    """Exception raised for configuration errors."""

    pass
