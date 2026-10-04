import json

import joblib
import numpy as np
import pandas as pd

from fraud.components.combine_threshold import scale_anomaly
from fraud.components.explainability import build_explainer, explain_rows
from fraud.components.unsupervised_model import anomaly_score
from fraud.constants import EXPECTED_COLUMNS, MODEL_DIR, TARGET

FEATURES = [c for c in EXPECTED_COLUMNS if c != TARGET]


class Predictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.xgb = joblib.load(f"{model_dir}/xgb_best.joblib")
        self.iforest = joblib.load(f"{model_dir}/iforest.joblib")
        self.pre = joblib.load(f"{model_dir}/preprocessor.joblib")
        with open(f"{model_dir}/config.json") as f:
            self.cfg = json.load(f)
        self.explainer = build_explainer(self.xgb)

    def predict(self, rows, explain=True):
        """rows = list of dicts, each with Time, V1..V28, Amount."""
        raw = pd.DataFrame(rows)[FEATURES]
        X = self.pre.transform(raw)
        px = self.xgb.predict_proba(X)[:, 1]
        pa = scale_anomaly(anomaly_score(self.iforest, X),
                           self.cfg["anomaly_lo"], self.cfg["anomaly_hi"])
        w = self.cfg["weight_xgb"]
        score = w * px + (1 - w) * pa
        flag = score >= self.cfg["threshold"]
        reasons = explain_rows(self.explainer, X, raw) if explain else [None] * len(raw)
        return [
            {
                "decision": "REVIEW" if flag[i] else "ALLOW",
                "score": round(float(score[i]), 4),
                "xgb_probability": round(float(px[i]), 4),
                "anomaly_score": round(float(pa[i]), 4),
                "reasons": reasons[i] if flag[i] else [],
            }
            for i in range(len(raw))
        ]