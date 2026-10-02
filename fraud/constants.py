"""One place for names and settings."""
TARGET = "Class"          # 1 = fraud, 0 = normal
RANDOM_STATE = 42

# Folders
RAW_CSV = "data/raw/creditcard.csv"
INGESTED_DIR = "artifacts/ingested"
TRANSFORMED_DIR = "artifacts/transformed"
MODEL_DIR = "artifacts/model"
REPORT_DIR = "reports"

# MongoDB names
DB_NAME = "fraud_db"
COLLECTION = "transactions"
INCOMING_COLLECTION = "incoming"

# Columns of the Kaggle credit-card dataset
EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
SCALE_COLUMNS = ["Time", "Amount"]

# Cost (in money) of a human checking ONE flagged transaction
REVIEW_COST = 5.0