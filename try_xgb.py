from fraud.components.data_transformation import run_transformation
from fraud.components.supervised_model import run_supervised

d = run_transformation()
model, best, results = run_supervised(d)

for name, r in results.items():
    print(name, {k: round(v, 4) for k, v in r.items()})
print("WINNER:", best)