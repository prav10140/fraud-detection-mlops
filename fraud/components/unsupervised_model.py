import os

import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud.constants import MODEL_DIR, RANDOM_STATE
from fraud.logger import logging


def train_iforest(X_train, y_train):
    X_normal = X_train[y_train == 0]  # learn only what normal looks like
    model = IsolationForest(
        n_estimators=200,
        max_samples=256,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_normal)
    return model


def anomaly_score(model, X):
    """Higher = stranger."""
    return -model.score_samples(X)


def run_unsupervised(d):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model = train_iforest(d["X_train"], d["y_train"])
    scores = anomaly_score(model, d["X_val"])
    result = {
        "pr_auc": float(average_precision_score(d["y_val"], scores)),
        "roc_auc": float(roc_auc_score(d["y_val"], scores)),
    }
    joblib.dump(model, f"{MODEL_DIR}/iforest.joblib")
    logging.info("IsolationForest on validation: %s", result)
    return model, result