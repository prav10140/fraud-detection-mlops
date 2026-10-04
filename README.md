# Payment Fraud Detection (MLOps)

An end-to-end fraud scoring service for card payments. It combines a supervised model (XGBoost) with an unsupervised one (Isolation Forest), chooses its alert threshold by **money** instead of accuracy, explains every flagged payment with SHAP, and monitors for data drift with a retraining workflow that only replaces the live model if the new one is no worse in money on unseen data.

It is a learning project built on a public dataset. It is not a production system.

**Code:** https://github.com/prav10140/fraud-detection-mlops
**Experiments (MLflow on DagsHub):** https://dagshub.com/prav10140/fraud-detection-mlops

## Problem

Fraud is rare: about 0.17% of payments in this dataset. A model that says "everything is normal" scores about 99.8% accuracy and catches no fraud, so accuracy is not used. Models are compared with PR-AUC, and the final alert level is chosen by total cost in money.

## Data

Kaggle "Credit Card Fraud Detection" (ULB / MLG): 284,807 transactions, 492 of them fraud. Check the dataset page for its terms of use. The CSV is not stored in Git (about 150 MB). Download it from Kaggle and place it at `data/raw/creditcard.csv`.

**Limitation:** V1 to V28 are anonymous PCA outputs. Explanations therefore say "V14 raised the risk", not a real-world reason. A company would have readable features such as the amount compared with the customer's usual spending.

## Results

Split: stratified 70 / 15 / 15 (train 199,364 rows, validation 42,721, test 42,722). Each split keeps about 0.17% fraud. The test set has **74 fraud payments**.

Weights and the threshold were chosen on the validation set only. The test set was used once, at the end.

**Cost** = 5.0 for every flagged payment (an assumed cost of a human review) + the full `Amount` of every fraud that was missed.

| Setup (test set) | Flagged | Precision | Recall | Frauds caught | Total cost |
|---|---|---|---|---|---|
| No model | 0 | - | 0% | 0 of 74 | 8,483.36 |
| XGBoost alone, default threshold 0.5 | 69 | 0.870 | 0.811 | 60 of 74 | 4,833.40 |
| **Combined, threshold 0.22** | 76 | 0.816 | 0.838 | 62 of 74 | **4,617.08** |

- Versus no model: 3,866 saved, about **46%** of the fraud loss on the test set.
- Versus XGBoost at 0.5: 216 saved (about 4.5%), which is 2 more frauds caught. This is a small difference on 74 frauds and should not be read as proof that the combination is better (see Limitations).

**Combined score:** 0.8 x XGBoost probability + 0.2 x scaled Isolation Forest anomaly score. The weight was picked on validation (validation cost 2,525.52).

**Model comparison on validation:**

| Model | PR-AUC | ROC-AUC | Precision @ 0.5 | Recall @ 0.5 |
|---|---|---|---|---|
| XGBoost with SMOTE | 0.827 | 0.980 | 0.720 | 0.797 |
| XGBoost with class weights (winner) | 0.841 | 0.978 | 0.905 | 0.770 |
| Isolation Forest (no labels used) | 0.113 | 0.946 | - | - |

Class weights beat SMOTE slightly, so imbalance handling was compared, not assumed. Isolation Forest has a low PR-AUC alone (many false alarms) but separates fraud from normal without ever seeing a label: average anomaly score 0.577 for fraud versus 0.404 for normal.

## Pipeline

```
Raw CSV -> MongoDB Atlas -> ingestion (stratified split) -> validation
 -> scaling + SMOTE (train only) -> XGBoost + Isolation Forest
 -> combined score -> cost-based threshold -> MLflow / DagsHub + DVC
 -> Flask API with SHAP reasons -> Docker -> GitHub Actions -> AWS ECR -> EC2
 -> drift monitor -> retraining workflow
```

**Leakage control:** the scaler is fit on training data only and only applied to validation and test. SMOTE is applied to training data only (199,364 rows became 398,040), never to validation or test. The Isolation Forest is trained on normal training rows only.

## Explanations

SHAP explains the XGBoost part of the score only (80% of it), not the Isolation Forest part. The top reasons are the three features that pushed a payment's risk up the most. The most common top reasons are V14, V10 and V4, which agrees with the correlations found in the exploratory analysis. Reasons show why the model scored a payment high. They are not proof of fraud.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status and the current alert threshold |
| `POST /predict` | One payment (JSON object) or up to 1,000 (list). Returns `ALLOW` or `REVIEW`, the score, and reasons for flagged payments |
| `GET /` | A small test page to type values and see the decision |

Input is validated: missing fields, non-numeric values and negative amounts return HTTP 400 with a clear message. The model is loaded once at startup.

**Latency (measured):**

- Scoring one payment in Python on my laptop: about 92 ms (before any optimisation).
- Through the live API on AWS EC2 (Sydney) called from India: median about 640 ms, p95 about 680 ms. This includes the network trip across regions.
- Normal and fraud payments took the same time, which suggests SHAP runs for every payment even though reasons are only shown for flagged ones. Running SHAP only for flagged payments is a known improvement that I have not applied.

## Deployment

- 6 automated tests (pytest) cover health, prediction, bad input and batches.
- GitHub Actions runs the tests, builds the Docker image, starts the container and checks `/health` before pushing the image to AWS ECR. This ran green.
- The image was pulled and run on an AWS EC2 server, and the same 7 API checks (health, fraud, normal, three bad inputs, batch) passed against the live server.
- The server is stopped when not in use to save credits. The AWS account is on the Free Plan, which ends after about 6 months, so the live service will not stay online permanently.
- Docker was never installed on my own computer. The image was built in the cloud by GitHub Actions.

## Drift monitoring

For each feature the new data is compared with a 10,000-row sample of training data using the Population Stability Index (PSI) and the KS statistic. `Time` is excluded because it always grows. Drift is declared when **at least 2 features have PSI of 0.25 or more** and the batch has **at least 500 rows**. The KS p-value is not used, because with thousands of rows it is tiny even for harmless differences.

| Test | Result |
|---|---|
| Normal data (5,000 rows) | No drift. Highest PSI 0.007 (V8) |
| Simulated drift (5,000 rows) | **Drift detected.** V14 PSI 4.26, V10 3.01, Amount 0.40. V4 moderate at 0.20 |

The drift is **simulated by hand** (Amount x 3, V14 minus 2, V10 plus 1.5, V4 x 1.5), so this shows that the monitor works, not what real-world drift looks like.

## Automatic retraining

A GitHub Actions workflow runs every Monday at 03:00 UTC (and can be started by hand):

1. Read new payments from the MongoDB `incoming` collection.
2. Run the drift check. Stop if there is no drift.
3. Retrain on the original data plus 60% of the incoming data.
4. Compare the old and new models, in money, on the other 40% of the incoming data, which neither model saw.
5. Replace the model only if the new one costs no more. Otherwise keep the old model and make no change.
6. If promoted, commit the new model and trigger the image build.

The comparison needs at least 20 frauds in the held-out data, otherwise the old model is kept.

**Result of my test** (40,000 simulated incoming rows, 16,000 held out containing 26 frauds, run on GitHub Actions in about 2 minutes):

| | Flagged | Frauds caught | Cost |
|---|---|---|---|
| Old model | 39 | 26 of 26 | **195.00** |
| New model | 24 | 21 of 26 | 940.17 |

The new model was rejected and the old one kept. This shows that retraining, the money-based comparison and the rollback work. It does not show that retraining is bad: my simulated drift pushes every row the same way, which makes fraud easier for the old model to spot, and the new model trained on many shifted normal rows that weakened the fraud signal.

My first version of this comparison used a re-split test file that contained rows the old model had trained on, which favoured the old model unfairly. I replaced it with the held-out design above.

## Limitations

- **Only 74 fraud payments in the test set.** One payment can move the numbers. The 46% saving versus no model is the solid result. The gain over XGBoost alone is not statistically solid.
- **Isolation Forest's value is not proven.** I compared the combination with XGBoost at its default 0.5 threshold, not with XGBoost at its own tuned threshold, so part of the gain may come from the threshold. During retraining the weight search was almost flat from 0.5 to 1.0.
- **The review cost of 5.0 is an assumption.** The best threshold moves with it. I did not run a sensitivity test.
- **Anonymous features** limit the explanations.
- **Drift and retraining are demonstrated on simulated data.** The incoming rows were copied from existing test rows and altered, so some near-duplicates of the original data remain in training.
- **Real fraud labels arrive late.** Retraining needs labels, so a real system would wait for them.
- **One train / validation / test split**, no cross-validation.
- **No authentication, rate limiting or HTTPS** on the API.
- **The weekly retrain job logs to local files on the GitHub runner**, not to DagsHub, so its metrics are not kept.

## Run it

```
conda create -n fraud python=3.11 -y
conda activate fraud
pip install -r requirements.txt
pip install -e .
```

1. Download `creditcard.csv` from Kaggle into `data/raw/`.
2. Create a MongoDB Atlas free cluster and a database user. Copy `.env.example` to `.env` and set `MONGO_DB_URL`. MLflow and DagsHub values are optional.
3. Load the data and train everything:

```
python push_data.py
python -m fraud.pipeline.training_pipeline
```

4. Start the API and open http://127.0.0.1:5000/ :

```
python app.py
```

5. Run the tests:

```
python -m pytest -q
```

Drift and retraining experiments:

```
python simulate_incoming.py --push                    # normal data
python simulate_incoming.py --drift --push --n 40000  # simulated drift
python -m fraud.pipeline.retrain_pipeline
```

## Project layout

```
fraud/components/   ingestion, validation, transformation, models,
                    threshold, explainability, drift
fraud/pipeline/     training, prediction, retraining
fraud/utils/        MongoDB helpers
artifacts/model/    deployed model files and config
tests/              API tests
.github/workflows/  main.yml (test, build, push), retrain.yml
app.py              Flask API and test page
Dockerfile          container for the API
```

## Tech

Python 3.11, pandas, scikit-learn, imbalanced-learn, XGBoost, SHAP, SciPy, MongoDB Atlas, MLflow, DagsHub, DVC, Flask, Gunicorn, Docker, GitHub Actions, AWS (ECR and EC2).
