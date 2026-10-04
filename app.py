import math

from flask import Flask, jsonify, request

from fraud.logger import logging
from fraud.pipeline.predict_pipeline import FEATURES, Predictor

app = Flask(__name__)
predictor = Predictor()          # loaded ONCE, when the server starts
MAX_BATCH = 1000


def check_row(row, i):
    """Return an error message, or None if the payment is fine."""
    if not isinstance(row, dict):
        return f"payment {i}: must be a JSON object"
    missing = [c for c in FEATURES if c not in row]
    if missing:
        return f"payment {i}: missing fields {missing}"
    for c in FEATURES:
        v = row[c]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return f"payment {i}: {c} must be a finite number"
    if row["Amount"] < 0:
        return f"payment {i}: Amount cannot be negative"
    return None


@app.get("/health")
def health():
    return jsonify({"status": "ok", "threshold": predictor.cfg["threshold"]})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Send a JSON body."}), 400

    single = isinstance(data, dict)
    rows = [data] if single else data
    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "Send one payment (object) or a non-empty list."}), 400
    if len(rows) > MAX_BATCH:
        return jsonify({"error": f"At most {MAX_BATCH} payments per request."}), 400

    for i, row in enumerate(rows):
        err = check_row(row, i)
        if err:
            return jsonify({"error": err}), 400

    try:
        results = predictor.predict(rows)
    except Exception:
        logging.exception("Prediction failed")
        return jsonify({"error": "Internal error while scoring."}), 500

    return jsonify(results[0] if single else {"results": results})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)