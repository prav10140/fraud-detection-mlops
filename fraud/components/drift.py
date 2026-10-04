import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from fraud.constants import EXPECTED_COLUMNS, INGESTED_DIR, MODEL_DIR, REPORT_DIR, TARGET
from fraud.logger import logging

CHECK_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in (TARGET, "Time")]
REFERENCE_FILE = f"{MODEL_DIR}/reference_sample.csv"
PSI_MAJOR = 0.25
PSI_MODERATE = 0.10
MIN_FEATURES_DRIFTED = 2
MIN_ROWS = 500


def make_reference(n=10000, in_dir=INGESTED_DIR):
    """Save a small sample of the TRAINING data as the 'normal' reference."""
    df = pd.read_csv(f"{in_dir}/train.csv")
    os.makedirs(MODEL_DIR, exist_ok=True)
    df[CHECK_COLUMNS].sample(n, random_state=42).to_csv(REFERENCE_FILE, index=False)
    logging.info("Saved reference sample to %s", REFERENCE_FILE)


def psi(ref, new, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    n = np.histogram(new, edges)[0] / len(new)
    r, n = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


def check_drift(new_df, reference=None):
    if reference is None:
        reference = pd.read_csv(REFERENCE_FILE)
    if len(new_df) < MIN_ROWS:
        return {"drift_detected": False,
                "note": f"Only {len(new_df)} rows; need {MIN_ROWS} for a reliable check"}

    rows = []
    for c in CHECK_COLUMNS:
        rows.append({
            "feature": c,
            "psi": round(psi(reference[c].to_numpy(), new_df[c].to_numpy()), 4),
            "ks": round(float(ks_2samp(reference[c], new_df[c]).statistic), 4),
        })
    rows.sort(key=lambda r: -r["psi"])
    major = [r["feature"] for r in rows if r["psi"] >= PSI_MAJOR]
    moderate = [r["feature"] for r in rows
                if PSI_MODERATE <= r["psi"] < PSI_MAJOR]

    report = {
        "drift_detected": len(major) >= MIN_FEATURES_DRIFTED,
        "rows_checked": len(new_df),
        "major_drift_features": major,
        "moderate_drift_features": moderate,
        "top_features": rows[:8],
    }
    if TARGET in new_df.columns:
        report["fraud_rate_new"] = round(float(new_df[TARGET].mean()), 5)
    return report


def run_drift(new_df):
    report = check_drift(new_df)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(f"{REPORT_DIR}/drift_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Drift report: drift_detected=%s", report["drift_detected"])
    return report