import pandas as pd
import requests

from fraud.constants import EXPECTED_COLUMNS, INGESTED_DIR, TARGET

URL = "http://16.176.167.174:5000"
FEATURES = [c for c in EXPECTED_COLUMNS if c != TARGET]

df = pd.read_csv(f"{INGESTED_DIR}/test.csv")
fraud = df[df[TARGET] == 1][FEATURES].iloc[0].to_dict()
normal = df[df[TARGET] == 0][FEATURES].iloc[0].to_dict()


def show(name, r):
    print(f"\n{name}: HTTP {r.status_code}")
    print("  ", r.text.strip()[:300])


show("1 health", requests.get(f"{URL}/health"))
show("2 real fraud (expect REVIEW)", requests.post(f"{URL}/predict", json=fraud))
show("3 real normal (expect ALLOW)", requests.post(f"{URL}/predict", json=normal))

missing = {k: v for k, v in normal.items() if k != "V1"}
show("4 missing V1 (expect 400)", requests.post(f"{URL}/predict", json=missing))

bad = dict(normal, Amount="abc")
show("5 Amount is text (expect 400)", requests.post(f"{URL}/predict", json=bad))

show("6 not JSON (expect 400)", requests.post(f"{URL}/predict", data="hello"))
show("7 batch of 2 (expect 2 results)", requests.post(f"{URL}/predict", json=[fraud, normal]))