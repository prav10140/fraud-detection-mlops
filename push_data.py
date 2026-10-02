"""Run ONCE: python push_data.py (CSV -> MongoDB)"""
import pandas as pd
from fraud.constants import COLLECTION, DB_NAME, RAW_CSV
from fraud.utils.mongo import push_dataframe

if __name__ == "__main__":
    df = pd.read_csv(RAW_CSV)
    n = push_dataframe(df, DB_NAME, COLLECTION)
    print(f"Pushed {n} rows to {DB_NAME}.{COLLECTION}")