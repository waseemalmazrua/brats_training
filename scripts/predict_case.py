import os
import argparse
from dotenv import load_dotenv
import json
from pathlib import Path

load_dotenv()

import mlflow.pyfunc

# ── Args ───────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--case_id", required=True, help="e.g. BraTS20_Training_001")
args = parser.parse_args()

# ── Model config from .env ─────────────────────
model_name  = os.getenv("REGISTERED_MODEL_NAME", "BraTS_UNet")
model_stage = os.getenv("MODEL_STAGE","Production")

# ── Load model ─────────────────────────────────
model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")

# ── Paths from .env ────────────────────────────
data_root = os.getenv("DATA_ROOT")
base_path = os.path.join(data_root, args.case_id, args.case_id)

# ── Predict ────────────────────────────────────
result = model.predict({
    "image": [
        f"{base_path}_t1.nii",
        f"{base_path}_t1ce.nii",
        f"{base_path}_t2.nii",
        f"{base_path}_flair.nii",
    ]
})

print(result["report"])
seg = result["segmentation"]


# ── Save JSON report ───────────────────────────
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

json_path = output_dir / f"{args.case_id}_report.json"

with open(json_path, "w") as f:
    json.dump(result["report"], f, indent=4)

print("JSON report saved to:", json_path)