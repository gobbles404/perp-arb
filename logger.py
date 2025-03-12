import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Create a log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d")
log_file = os.path.join(logs_dir, f"app_{timestamp}.log")

# Configure the logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(log_file),  # Save to a timestamped file
    ],
)

# Create a logger you can import
log = logging.getLogger("my_app")

# Different log levels you can use:
# log.debug("Detailed information, typically of interest only when diagnosing problems.")
# log.info("Confirmation that things are working as expected.")
# log.warning("An indication that something unexpected happened, or may happen in the near future.")
# log.error("Due to a more serious problem, the software has not been able to perform some function.")
# log.critical("A serious error, indicating that the program itself may be unable to continue running.")

# Example usage within logger.py
if __name__ == "__main__":
    log.info("Logger initialized")
    log.debug("This is a debug message")
    log.warning("This is a warning message")
    log.error("This is an error message")
