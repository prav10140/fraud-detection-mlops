import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from fraud.components.explainability import build_explainer, explain_rows
from fraud.constants import INGESTED_DIR, MODEL_DIR, REPORT_DIR, TARGET

model = joblib.load(f"{MODEL_DIR}/xgb_best.joblib")
pre = joblib.load(f"{MODEL_DIR}/preprocessor.joblib")

df = pd.read_csv(f"{INGESTED_DIR}/test.csv")
raw = df.drop(columns=[TARGET])
y = df[TARGET].to_numpy()
X = pre.transform(raw)

score = model.predict_proba(X)[:, 1]
top = score.argsort()[::-1][:5]          # the 5 riskiest payments
explainer = build_explainer(model)
reasons = explain_rows(explainer, X.iloc[top], raw.iloc[top])

for n, i in enumerate(top):
    print(f"\nPayment #{n+1}: risk {score[i]:.3f}, really fraud: {bool(y[i])}")
    for r in reasons[n]:
        print(f"   {r['feature']} = {r['value']}  ->  {r['direction']} ({r['impact']:+})")

# Global picture: which features matter most overall
os.makedirs(REPORT_DIR, exist_ok=True)
sample = X.sample(2000, random_state=42)
sv = explainer.shap_values(sample)
shap.summary_plot(sv, sample, plot_type="bar", show=False)
plt.savefig(f"{REPORT_DIR}/shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved reports/shap_summary.png")