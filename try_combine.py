import json

import joblib

from fraud.components.combine_threshold import run_combination
from fraud.components.data_transformation import run_transformation
from fraud.constants import MODEL_DIR

d = run_transformation()
xgb = joblib.load(f"{MODEL_DIR}/xgb_class_weight.joblib")   # the winner from Module 5
iforest = joblib.load(f"{MODEL_DIR}/iforest.joblib")

config, report = run_combination(d, xgb, iforest)
print("CONFIG:", config)
print(json.dumps(report, indent=2))