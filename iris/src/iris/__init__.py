import pathlib
import os

from iris.helpers import logger as log_handler
from iris.helpers import config as config_handler

LOG_PATH = os.getenv("LOG_PATH", "")

# Initiate logger
logger = log_handler.LogHandler(file_path=pathlib.Path(LOG_PATH) if LOG_PATH else None)

# Initiate config handler
config = config_handler.Storage(logger=logger)
