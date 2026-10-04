import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score

from fraud.components.unsupervised_model import anomaly_score
from fraud.constants import MODEL_DIR, REPORT_DIR, REVIEW_COST
from fraud.logger import logging

GRID = np.arange(0.01, 1.0, 0.01)


def scale_anomaly(s, lo, hi):
    """Squeeze anomaly scores into 0..1 using limits learned from TRAIN."""
    return np.clip((s - lo) / (hi - lo), 0, 1)


def total_cost(scores, y, amount, thr, review_cost=REVIEW_COST):
    flag = scores >= thr
    review = flag.sum() * review_cost
    missed = amount[(y == 1) & (~flag)].sum()
    return float(review + missed)


def best_threshold(scores, y, amount, review_cost=REVIEW_COST):
    costs = np.array([total_cost(scores, y, amount, t, review_cost) for t in GRID])
    i = int(costs.argmin())
    return float(GRID[i]), float(costs[i]), costs


def describe(scores, y, amount, thr):
    flag = scores >= thr
    return {
        "threshold": round(float(thr), 3),
        "flagged": int(flag.sum()),
        "precision": round(float(precision_score(y, flag, zero_division=0)), 4),
        "recall": round(float(recall_score(y, flag, zero_division=0)), 4),
        "cost": round(total_cost(scores, y, amount, thr), 2),
    }


def run_combination(d, xgb, iforest):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1) squeeze limits from TRAIN only
    lo, hi = np.percentile(anomaly_score(iforest, d["X_train"]), [1, 99.9])

    # 2) validation scores from both models
    val_x = xgb.predict_proba(d["X_val"])[:, 1]
    val_a = scale_anomaly(anomaly_score(iforest, d["X_val"]), lo, hi)

    # 3) try several weights, keep the cheapest on VALIDATION
    rows = []
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        s = w * val_x + (1 - w) * val_a
        thr, cost, _ = best_threshold(s, d["y_val"], d["amount_val"])
        rows.append((float(w), thr, cost))
        logging.info("weight %.1f -> best threshold %.2f, val cost %.2f", w, thr, cost)
    w, thr, val_cost = min(rows, key=lambda r: (r[2], -r[0]))

    # 4) cost curve chart for the chosen weight (validation)
    s_val = w * val_x + (1 - w) * val_a
    _, _, costs = best_threshold(s_val, d["y_val"], d["amount_val"])
    fig, ax = plt.subplots()
    ax.plot(GRID, costs)
    ax.axvline(thr, color="red", linestyle="--", label=f"chosen {thr:.2f}")
    ax.set_xlabel("Alert threshold")
    ax.set_ylabel("Total cost on validation (money)")
    ax.legend()
    fig.savefig(f"{REPORT_DIR}/cost_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 5) TEST: look once
    te_x = xgb.predict_proba(d["X_test"])[:, 1]
    te_a = scale_anomaly(anomaly_score(iforest, d["X_test"]), lo, hi)
    te_s = w * te_x + (1 - w) * te_a
    y_te, amt_te = d["y_test"], d["amount_test"]

    report = {
        "weight_xgb": w,
        "val_cost_chosen": round(val_cost, 2),
        "no_model_cost_test": round(float(amt_te[y_te == 1].sum()), 2),
        "xgb_alone_at_0.5_test": describe(te_x, y_te, amt_te, 0.5),
        "chosen_test": describe(te_s, y_te, amt_te, thr),
    }
    with open(f"{REPORT_DIR}/threshold_report.json", "w") as f:
        json.dump(report, f, indent=2)

    config = {"weight_xgb": w, "threshold": thr, "anomaly_lo": float(lo),
              "anomaly_hi": float(hi), "review_cost": REVIEW_COST}
    with open(f"{MODEL_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    return config, report