import os
import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

def get_client():
    url = os.getenv("MONGO_DB_URL")
    if not url:
        raise ValueError("MONGO_DB_URL is missing. Put it in your .env file.")
    return MongoClient(url, tlsCAFile=certifi.where())

def push_dataframe(df, db_name, collection, chunk=10000):
    """Load step of ETL: DataFrame -> MongoDB collection."""
    coll = get_client()[db_name][collection]
    coll.delete_many({})  # re-running will not create duplicates
    records = df.to_dict(orient="records")
    for i in range(0, len(records), chunk):
        coll.insert_many(records[i:i + chunk])
    return len(records)

def read_collection(db_name, collection):
    """Extract step: MongoDB collection -> DataFrame."""
    docs = get_client()[db_name][collection].find({}, {"_id": 0})
    return pd.DataFrame(list(docs))