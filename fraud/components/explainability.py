import numpy as np
import shap


def build_explainer(model):
    return shap.TreeExplainer(model)


def explain_rows(explainer, X, raw, k=3):
    """X = preprocessed rows (with column names), raw = same rows before scaling.
    Returns the top k reasons for each row."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):      # some versions return one array per class
        sv = sv[1]
    cols = list(X.columns)
    out = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(sv[i]))[:k]
        out.append([
            {
                "feature": cols[j],
                "value": round(float(raw.iloc[i][cols[j]]), 3),
                "impact": round(float(sv[i][j]), 3),
                "direction": "raises risk" if sv[i][j] > 0 else "lowers risk",
            }
            for j in order
        ])
    return out