import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from fraud.constants import (COLLECTION, DB_NAME, INCOMING_COLLECTION,
                             INGESTED_DIR, RANDOM_STATE, TARGET)
from fraud.exception import FraudException
from fraud.logger import logging
from fraud.utils.mongo import read_collection

def split_and_save(df, out_dir=INGESTED_DIR):
    """70% train, 15% validation, 15% test.
    stratify = every part gets the same tiny share of fraud."""
    os.makedirs(out_dir, exist_ok=True)
    train, rest = train_test_split(
        df, test_size=0.30, stratify=df[TARGET], random_state=RANDOM_STATE)
    val, test = train_test_split(
        rest, test_size=0.50, stratify=rest[TARGET], random_state=RANDOM_STATE)
    paths = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        paths[name] = os.path.join(out_dir, f"{name}.csv")
        part.to_csv(paths[name], index=False)
        logging.info("%s: %d rows, fraud rate %.4f%%",
                     name, len(part), part[TARGET].mean() * 100)
    return paths

def run_ingestion():
    try:
        df = read_collection(DB_NAME, COLLECTION)
        if os.getenv("INCLUDE_INCOMING") == "true":  # used by auto-retraining
            new = read_collection(DB_NAME, INCOMING_COLLECTION)
            df = pd.concat([df, new], ignore_index=True)
        logging.info("Read %d rows from MongoDB", len(df))
        return split_and_save(df)
    except Exception as e:
        raise FraudException(e, sys)