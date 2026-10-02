import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{datetime.now():%Y_%m_%d_%H_%M_%S}.log"

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(module)s:%(lineno)d - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE)),
        logging.StreamHandler(),
    ],
)