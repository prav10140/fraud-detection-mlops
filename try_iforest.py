from fraud.components.data_transformation import run_transformation
from fraud.components.unsupervised_model import anomaly_score, run_unsupervised

d = run_transformation()
model, result = run_unsupervised(d)
print({k: round(v, 4) for k, v in result.items()})

s = anomaly_score(model, d["X_val"])
print("avg anomaly score, normal:", round(s[d["y_val"] == 0].mean(), 4))
print("avg anomaly score, fraud :", round(s[d["y_val"] == 1].mean(), 4))