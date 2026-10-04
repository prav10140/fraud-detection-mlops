import pandas as pd

from fraud.constants import INGESTED_DIR, TARGET
from fraud.pipeline.predict_pipeline import FEATURES, Predictor

df = pd.read_csv(f"{INGESTED_DIR}/test.csv")
p = Predictor()

fraud = df[df[TARGET] == 1].head(3)
normal = df[df[TARGET] == 0].head(3)

for name, part in [("REAL FRAUD", fraud), ("REAL NORMAL", normal)]:
    print(f"\n=== {name} ===")
    results = p.predict(part[FEATURES].to_dict(orient="records"))
    for r in results:
        print(r["decision"], r["score"], [x["feature"] for x in r["reasons"]])