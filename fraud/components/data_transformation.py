import os

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from fraud.constants import (INGESTED_DIR, RANDOM_STATE, SCALE_COLUMNS, TARGET,
                             TRANSFORMED_DIR)
from fraud.logger import logging


def build_preprocessor():
    """Scale Time and Amount. V1..V28 are already scaled, so pass them through."""
    return ColumnTransformer(
        [("scale", RobustScaler(), SCALE_COLUMNS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")  # keep column names (needed for SHAP)


def split_xy(df):
    return df.drop(columns=[TARGET]), df[TARGET].to_numpy()


def run_transformation(in_dir=INGESTED_DIR, out_dir=TRANSFORMED_DIR):
    os.makedirs(out_dir, exist_ok=True)
    X_train, y_train = split_xy(pd.read_csv(f"{in_dir}/train.csv"))
    X_val, y_val = split_xy(pd.read_csv(f"{in_dir}/val.csv"))
    X_test, y_test = split_xy(pd.read_csv(f"{in_dir}/test.csv"))

    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)   # learn scaling from TRAIN only
    X_val_t = pre.transform(X_val)           # only apply to val and test
    X_test_t = pre.transform(X_test)

    # SMOTE: make extra "look-alike" fraud rows, ONLY for training data
    X_sm, y_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_t, y_train)
    logging.info("Train size %d -> %d after SMOTE", len(X_train_t), len(X_sm))

    joblib.dump(pre, f"{out_dir}/preprocessor.joblib")

    return {
        "preprocessor": pre,
        "raw_columns": list(X_train.columns),
        "X_train": X_train_t, "y_train": y_train,
        "X_val": X_val_t, "y_val": y_val,
        "X_test": X_test_t, "y_test": y_test,
        "X_train_sm": X_sm, "y_train_sm": y_sm,
        "amount_val": X_val["Amount"].to_numpy(),    # real money values, for cost
        "amount_test": X_test["Amount"].to_numpy(),
    }