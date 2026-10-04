import os

import joblib
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from xgboost import XGBClassifier

from fraud.constants import MODEL_DIR, RANDOM_STATE
from fraud.logger import logging


def train_xgb(X, y, use_class_weight=False):
    params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if use_class_weight:
        # normal rows per fraud row (about 577 here)
        params["scale_pos_weight"] = float((y == 0).sum() / (y == 1).sum())
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model


def evaluate(model, X, y, threshold=0.5):
    scores = model.predict_proba(X)[:, 1]
    pred = (scores >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision@0.5": float(precision_score(y, pred, zero_division=0)),
        "recall@0.5": float(recall_score(y, pred, zero_division=0)),
    }


def run_supervised(d):
    """d = the dictionary returned by run_transformation()."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    variants = {
        "smote": (d["X_train_sm"], d["y_train_sm"], False),
        "class_weight": (d["X_train"], d["y_train"], True),
    }
    models, results = {}, {}
    for name, (X, y, cw) in variants.items():
        model = train_xgb(X, y, use_class_weight=cw)
        models[name] = model
        results[name] = evaluate(model, d["X_val"], d["y_val"])
        joblib.dump(model, f"{MODEL_DIR}/xgb_{name}.joblib")
        logging.info("XGBoost %s on validation: %s", name, results[name])

    best = max(results, key=lambda k: results[k]["pr_auc"])
    logging.info("Winner: %s", best)
    return models[best], best, results