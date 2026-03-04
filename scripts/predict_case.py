import os
import argparse
from dotenv import load_dotenv

load_dotenv()

import mlflow.pyfunc

# ── Args ───────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--case_id", required=True, help="e.g. BraTS20_Training_001")
args = parser.parse_args()

# ── Model config from .env ─────────────────────
model_name  = os.getenv("REGISTERED_MODEL_NAME", "BraTS_UNet")
model_stage = os.getenv("MODEL_STAGE",           "Production")

# ── Load model ─────────────────────────────────
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")

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