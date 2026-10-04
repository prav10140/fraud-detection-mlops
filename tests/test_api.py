import pandas as pd

from app import app
from fraud.constants import MODEL_DIR

REF = pd.read_csv(f"{MODEL_DIR}/reference_sample.csv")


def good_row():
    row = REF.iloc[0].to_dict()
    row["Time"] = 0.0
    return row


def test_health():
    r = app.test_client().get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_predict_one_payment():
    r = app.test_client().post("/predict", json=good_row())
    assert r.status_code == 200
    assert r.get_json()["decision"] in ("ALLOW", "REVIEW")


def test_missing_field_is_400():
    row = good_row()
    del row["V1"]
    assert app.test_client().post("/predict", json=row).status_code == 400


def test_text_amount_is_400():
    row = good_row()
    row["Amount"] = "abc"
    assert app.test_client().post("/predict", json=row).status_code == 400


def test_negative_amount_is_400():
    row = good_row()
    row["Amount"] = -5
    assert app.test_client().post("/predict", json=row).status_code == 400


def test_batch_of_two():
    r = app.test_client().post("/predict", json=[good_row(), good_row()])
    assert r.status_code == 200
    assert len(r.get_json()["results"]) == 2