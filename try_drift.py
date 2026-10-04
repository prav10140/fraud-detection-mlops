import json

import pandas as pd

from fraud.components.drift import make_reference, run_drift

make_reference()
new = pd.read_csv("artifacts/incoming.csv")
report = run_drift(new)
print(json.dumps(report, indent=2))