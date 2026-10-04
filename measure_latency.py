import statistics
import time

import pandas as pd
import requests

from fraud.constants import EXPECTED_COLUMNS, INGESTED_DIR, TARGET

URL = "http://127.0.0.1:5000/predict"
FEATURES = [c for c in EXPECTED_COLUMNS if c != TARGET]
df = pd.read_csv(f"{INGESTED_DIR}/test.csv")
rows = {
    "normal (ALLOW)": df[df[TARGET] == 0][FEATURES].iloc[0].to_dict(),
    "fraud (REVIEW + SHAP)": df[df[TARGET] == 1][FEATURES].iloc[0].to_dict(),
}

s = requests.Session()
for name, row in rows.items():
    s.post(URL, json=row)                      # warm-up, not counted
    times = []
    for _ in range(100):
        t = time.perf_counter()
        s.post(URL, json=row)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    print(f"{name}: median {statistics.median(times):.1f} ms, p95 {times[94]:.1f} ms")