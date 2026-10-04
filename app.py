import math
from flask import Response
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

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Fraud check</title>
<style>
body{font-family:sans-serif;max-width:640px;margin:30px auto;padding:0 12px}
label{display:block;margin:8px 0 2px}input{width:100%;padding:6px;box-sizing:border-box}
button{margin-top:14px;padding:10px 18px;font-size:16px}
#out{margin-top:20px;padding:14px;border-radius:6px;background:#eee;white-space:pre-wrap}
.review{background:#fdd!important}.allow{background:#dfd!important}
</style></head><body>
<h2>Payment fraud check</h2>
<p>Enter a few values. Every field you leave empty is set to 0, an average payment.</p>
<label>Amount</label><input id="Amount" value="50">
<label>V14</label><input id="V14" placeholder="e.g. -5.2">
<label>V10</label><input id="V10" placeholder="e.g. -3.2">
<label>V4</label><input id="V4" placeholder="e.g. 3">
<label>V12</label><input id="V12" placeholder="e.g. -4">
<label>V17</label><input id="V17" placeholder="e.g. -3">
<button onclick="go()">Check payment</button>
<div id="out">Result will appear here.</div>
<script>
async function go(){
  const p={Time:0};
  for(let i=1;i<=28;i++)p["V"+i]=0;
  p.Amount=0;
  for(const k of ["Amount","V14","V10","V4","V12","V17"]){
    const v=document.getElementById(k).value.trim();
    if(v!=="")p[k]=Number(v);
  }
  const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)});
  const d=await r.json(),o=document.getElementById("out");
  if(!r.ok){o.className="";o.textContent="Error: "+d.error;return;}
  o.className=d.decision==="REVIEW"?"review":"allow";
  let t="Decision: "+d.decision+"\\nScore: "+d.score+" (alert level 0.22)\\n";
  if(d.reasons.length){t+="\\nWhy flagged:\\n";
    d.reasons.forEach(x=>t+="  "+x.feature+" = "+x.value+" -> "+x.direction+"\\n");}
  o.textContent=t;
}
</script></body></html>"""


@app.get("/")
def home():
    return Response(PAGE, mimetype="text/html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)