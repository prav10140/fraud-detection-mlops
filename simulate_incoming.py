import argparse

import pandas as pd

from fraud.constants import DB_NAME, INCOMING_COLLECTION, INGESTED_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--drift", action="store_true", help="make the data different")
parser.add_argument("--push", action="store_true", help="also upload to MongoDB 'incoming'")
parser.add_argument("--n", type=int, default=5000)
args = parser.parse_args()

df = pd.read_csv(f"{INGESTED_DIR}/test.csv").sample(args.n, random_state=1)

if args.drift:
    df["Amount"] = df["Amount"] * 3      # people spend more
    df["V14"] = df["V14"] - 2            # shifted behaviour
    df["V10"] = df["V10"] + 1.5
    df["V4"] = df["V4"] * 1.5
    print("Created DRIFTED data")
else:
    print("Created NORMAL data (no drift)")

df.to_csv("artifacts/incoming.csv", index=False)
print("Saved artifacts/incoming.csv with", len(df), "rows")

if args.push:
    from fraud.utils.mongo import push_dataframe
    n = push_dataframe(df, DB_NAME, INCOMING_COLLECTION)
    print(f"Pushed {n} rows to {DB_NAME}.{INCOMING_COLLECTION}")