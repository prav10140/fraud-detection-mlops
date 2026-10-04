import os
import sys

import joblib
import mlflow
from dotenv import load_dotenv

from fraud.components.combine_threshold import run_combination
from fraud.components.data_ingestion import run_ingestion
from fraud.components.data_transformation import run_transformation
from fraud.components.data_validation import run_validation
from fraud.components.supervised_model import run_supervised
from fraud.components.unsupervised_model import run_unsupervised
from fraud.constants import MODEL_DIR, REPORT_DIR
from fraud.exception import FraudException
from fraud.logger import logging

load_dotenv()


def setup_mlflow():
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)  # DagsHub (username/token come from .env)
        logging.info("MLflow tracking at %s", uri)
    else:
        logging.info("MLFLOW_TRACKING_URI not set: saving runs locally in mlruns/")
    mlflow.set_experiment("fraud-detection")


def run_training():
    try:
        setup_mlflow()
        with mlflow.start_run():
            run_ingestion()

            report = run_validation()
            if not report["is_valid"]:
                raise ValueError(f"Data validation failed: {report['problems']}")

            d = run_transformation()
            xgb, best, xgb_results = run_supervised(d)
            iforest, if_result = run_unsupervised(d)

            # files the API will need later (this folder goes in Git)
            joblib.dump(xgb, f"{MODEL_DIR}/xgb_best.joblib")
            joblib.dump(d["preprocessor"], f"{MODEL_DIR}/preprocessor.joblib")

            config, rep = run_combination(d, xgb, iforest)

            # ---- log to MLflow ----
            mlflow.log_params({
                "imbalance_method": best,
                "weight_xgb": config["weight_xgb"],
                "threshold": config["threshold"],
                "review_cost": config["review_cost"],
                "train_rows": report["rows"],
            })
            for name, r in xgb_results.items():
                mlflow.log_metric(f"xgb_{name}_pr_auc", r["pr_auc"])
                mlflow.log_metric(f"xgb_{name}_roc_auc", r["roc_auc"])
            mlflow.log_metric("iforest_pr_auc", if_result["pr_auc"])
            mlflow.log_metric("iforest_roc_auc", if_result["roc_auc"])

            chosen = rep["chosen_test"]
            mlflow.log_metric("test_cost", chosen["cost"])
            mlflow.log_metric("test_precision", chosen["precision"])
            mlflow.log_metric("test_recall", chosen["recall"])
            mlflow.log_metric("test_flagged", chosen["flagged"])
            mlflow.log_metric("no_model_cost_test", rep["no_model_cost_test"])
            mlflow.log_metric("money_saved_test",
                              rep["no_model_cost_test"] - chosen["cost"])

            mlflow.log_artifacts(REPORT_DIR)
            mlflow.log_artifact(f"{MODEL_DIR}/config.json")
            mlflow.log_artifact(f"{MODEL_DIR}/xgb_best.joblib")

            logging.info("Training finished. Config: %s", config)
            return config, rep
    except Exception as e:
        raise FraudException(e, sys)


if __name__ == "__main__":
    config, rep = run_training()
    print("CONFIG:", config)
    print("TEST RESULT:", rep["chosen_test"])