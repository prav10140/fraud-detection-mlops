import os
for d in ["fraud/components", "fraud/pipeline", "fraud/utils", "tests",
          "data/raw", "artifacts/model", "reports", ".github/workflows"]:
    os.makedirs(d, exist_ok=True)
for d in ["fraud", "fraud/components", "fraud/pipeline", "fraud/utils"]:
    open(f"{d}/__init__.py", "a").close()