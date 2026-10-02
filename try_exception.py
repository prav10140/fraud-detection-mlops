import sys
from fraud.exception import FraudException
from fraud.logger import logging

logging.info("Logger works")
try:
    1 / 0
except Exception as e:
    print(FraudException(e, sys))