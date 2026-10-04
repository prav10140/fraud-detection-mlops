import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from fraud.components.drift import MIN_ROWS, check_drift, make_reference
from fraud.constants import (DB_NAME, INCOMING_COLLECTION, INGESTED_DIR,
                             MODEL_DIR, REPORT_DIR, REVIEW_COST, TARGET)
from fraud.logger import logging
from fraud.pipeline.predict_pipeline import FEATURES, Predictor
from fraud.utils.mongo import read_collection
from sklearn.model_selection import train_test_split

INCOMING_TRAIN_CSV = "artifacts/incoming_train.csv"
MIN_HOLDOUT_FRAUDS = 20
OLD_DIR = "artifacts/model_old"


def set_output(key, value):
    """Lets the GitHub workflow read the result."""
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")


def finish(report):
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(f"{REPORT_DIR}/retrain_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    set_output("retrained", str(report["retrained"]).lower())
    set_output("promoted", str(report["promoted"]).lower())

def money_cost(model_dir, df):
    """Total cost of a saved model on a labelled dataframe, in money."""
    p = Predictor(model_dir)
    res = p.predict(df[FEATURES].to_dict(orient="records"), explain=False)
    flag = np.array([r["decision"] == "REVIEW" for r in res])
    y = df[TARGET].to_numpy()
    amt = df["Amount"].to_numpy()
    missed = amt[(y == 1) & (~flag)].sum()
    return {
        "cost": round(float(flag.sum() * REVIEW_COST + missed), 2),
        "flagged": int(flag.sum()),
        "frauds_caught": int(((y == 1) & flag).sum()),
        "frauds_total": int((y == 1).sum()),
    }


def main(force=False):
    report = {"retrained": False, "promoted": False}

    incoming = read_collection(DB_NAME, INCOMING_COLLECTION)
    report["incoming_rows"] = len(incoming)
    if len(incoming) < MIN_ROWS:
        report["note"] = f"Need at least {MIN_ROWS} incoming rows. Model kept."
        return finish(report)

    drift = check_drift(incoming)
    report["drift"] = {k: drift[k] for k in
                       ("drift_detected", "major_drift_features",
                        "moderate_drift_features") if k in drift}
    if not drift["drift_detected"] and not force:
        report["note"] = "No drift. Model kept."
        return finish(report)

    # 60% of incoming is used for retraining, 40% stays hidden to judge both models
    strat = incoming[TARGET] if incoming[TARGET].sum() >= 10 else None
    inc_train, inc_hold = train_test_split(
        incoming, test_size=0.4, stratify=strat, random_state=42)
    os.makedirs("artifacts", exist_ok=True)
    inc_train.to_csv(INCOMING_TRAIN_CSV, index=False)
    report["holdout_rows"] = len(inc_hold)
    report["holdout_frauds"] = int(inc_hold[TARGET].sum())

    shutil.rmtree(OLD_DIR, ignore_errors=True)
    shutil.copytree(MODEL_DIR, OLD_DIR)
    os.environ["INCLUDE_INCOMING"] = "true"
    os.environ["INCOMING_TRAIN_CSV"] = INCOMING_TRAIN_CSV

    from fraud.pipeline.training_pipeline import run_training
    run_training()
    report["retrained"] = True

    old = money_cost(OLD_DIR, inc_hold)
    new = money_cost(MODEL_DIR, inc_hold)
    report["old_model"], report["new_model"] = old, new

    if report["holdout_frauds"] < MIN_HOLDOUT_FRAUDS:
        keep_new = False
        report["note"] = (f"Hidden data has only {report['holdout_frauds']} frauds, "
                          f"too few to judge fairly. Old model restored.")
    elif new["cost"] <= old["cost"]:
        keep_new = True
        report["note"] = "New model is not worse in money on hidden new data. Promoted."
    else:
        keep_new = False
        report["note"] = "New model cost more on hidden new data. Old model restored."

    if keep_new:
        report["promoted"] = True
        make_reference()
    else:
        shutil.rmtree(MODEL_DIR)
        shutil.copytree(OLD_DIR, MODEL_DIR)

    shutil.rmtree(OLD_DIR, ignore_errors=True)
    logging.info("Retrain result: %s", report["note"])
    finish(report)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="retrain even when no drift is found")
    main(force=parser.parse_args().force)