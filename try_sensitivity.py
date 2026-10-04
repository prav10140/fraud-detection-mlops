import joblib

from fraud.components.combine_threshold import best_threshold
from fraud.components.data_transformation import run_transformation
from fraud.constants import MODEL_DIR

d = run_transformation()
xgb = joblib.load(f"{MODEL_DIR}/xgb_class_weight.joblib")
s = xgb.predict_proba(d["X_val"])[:, 1]

for rc in [0.5, 5, 25, 100]:
    thr, cost, _ = best_threshold(s, d["y_val"], d["amount_val"], review_cost=rc)
    flagged = int((s >= thr).sum())
    print(f"review cost {rc:>5}: best threshold {thr:.2f}, flagged {flagged}, total cost {cost:.2f}")