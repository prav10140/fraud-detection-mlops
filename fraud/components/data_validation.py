import json
import os

import pandas as pd

from fraud.constants import EXPECTED_COLUMNS, INGESTED_DIR, REPORT_DIR, TARGET
from fraud.logger import logging


def validate(df):
    """Check the data BEFORE using it. Returns a small report (a dict)."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        return {"is_valid": False, "problems": [f"Missing columns: {missing}"]}

    data = df[EXPECTED_COLUMNS]
    not_numeric = [c for c in EXPECTED_COLUMNS
                   if not pd.api.types.is_numeric_dtype(data[c])]
    if not_numeric:
        return {"is_valid": False, "problems": [f"Non-numeric columns: {not_numeric}"]}

    problems = []
    if data.isnull().any().any():
        problems.append("There are missing values")
    if not set(data[TARGET].unique()) <= {0, 1}:
        problems.append("Class must contain only 0 and 1")
    if (data["Amount"] < 0).any():
        problems.append("Negative Amount found")
    fraud_rate = float(data[TARGET].mean())
    if not 0.0005 <= fraud_rate <= 0.05:
        problems.append(f"Fraud rate {fraud_rate:.4%} looks unusual")

    return {"is_valid": not problems, "problems": problems,
            "rows": len(data), "fraud_rate": fraud_rate}


def run_validation(in_dir=INGESTED_DIR):
    df = pd.read_csv(os.path.join(in_dir, "train.csv"))
    report = validate(df)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Validation report: %s", report)
    return report