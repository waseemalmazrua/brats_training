# =====================================================
# MONAI 3D UNet — BraTS Segmentation
# Target  : Ubuntu GPU VM (DigitalOcean / any cloud VM)
# Covers  : Training + MLflow PyFunc + Production Inference
#           Physician uploads 4 NIfTI files → gets back
#           segmentation mask + tumor report (size, grade)
# =====================================================

# =====================================================
# 1️⃣  Standard Libraries
# =====================================================

import os
import sys
import glob
import random
import logging
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path

# =====================================================
# 2️⃣  MONAI Imports
# =====================================================

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    Spacingd,
    Orientationd,
    Activations,
    AsDiscrete,
    MapTransform,
)
from monai.inferers import sliding_window_inference

# =====================================================
# 3️⃣  Logging
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =====================================================
# 4️⃣  Config — all overridable via env vars
# =====================================================

DATA_DIR        = Path(os.environ.get("DATA_ROOT",
                        "/workspace/data/BraTS20"))
CHECKPOINT_DIR  = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints"))

MAX_EPOCHS      = int(os.environ.get("MAX_EPOCHS",    "100"))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE",    "2"))
LEARNING_RATE   = float(os.environ.get("LEARNING_RATE","1e-4"))
PATIENCE        = int(os.environ.get("PATIENCE",      "15"))
MIN_DELTA       = float(os.environ.get("MIN_DELTA",   "0.001"))
VAL_INTERVAL    = int(os.environ.get("VAL_INTERVAL",  "1"))

_cpu_cap        = max(1, (os.cpu_count() or 2) - 1)
NUM_WORKERS     = int(os.environ.get("NUM_WORKERS", str(min(4, _cpu_cap))))
REQUIRE_GPU     = os.environ.get("REQUIRE_GPU", "true").lower() != "false"

MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
MODEL_NAME      = os.environ.get("MODEL_NAME", "brats_unet_v1")

# Voxel spacing used during training — needed for mm³ volume calculation
VOXEL_SPACING_MM = (1.0, 1.0, 1.0)   # after Spacingd resampling

# =====================================================
# 5️⃣  MLflow — fully optional
# =====================================================

mlflow        = None
mlflow_pyfunc = None
MlflowClient  = None

if MLFLOW_URI:
    try:
        import mlflow        as _mlflow
        import mlflow.pyfunc as _mlflow_pyfunc
        from mlflow.tracking import MlflowClient as _MlflowClient

        _mlflow.set_tracking_uri(MLFLOW_URI)
        _mlflow.set_experiment("BraTS_UNet_Production")

        mlflow        = _mlflow
        mlflow_pyfunc = _mlflow_pyfunc
        MlflowClient  = _MlflowClient
        logger.info(f"MLflow tracking enabled → {MLFLOW_URI}")
    except ImportError:
        logger.warning("mlflow not installed — tracking disabled.")
else:
    logger.info("MLFLOW_TRACKING_URI not set — MLflow tracking disabled.")

# =====================================================
# 6️⃣  GPU Validation
# =====================================================

def validate_gpu() -> torch.device:
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"✓ GPU detected : {name}  |  VRAM : {vram:.1f} GB")
    else:
        logger.warning("No CUDA GPU detected — running on CPU.")
        if REQUIRE_GPU:
            raise RuntimeError(
                "REQUIRE_GPU=true but no CUDA GPU found.\n"
                "  • Run: nvidia-smi\n"
                "  • Set REQUIRE_GPU=false to allow CPU-only training."
            )
    return torch.device("cuda" if has_gpu else "cpu")

# =====================================================
# 7️⃣  Safe Glob Helper
# =====================================================

def safe_glob_one(pattern: str, description: str) -> str:
    """
    Glob and return first match.
    Raises FileNotFoundError clearly instead of IndexError.
    Handles both .nii and .nii.gz automatically.
    """
    # Try the given pattern first, then fallback without .gz
    for pat in [pattern, pattern.replace(".nii.gz", ".nii")]:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Missing file — {description}\n"
        f"  Pattern tried : {pattern}\n"
        f"  Also tried    : {pattern.replace('.nii.gz', '.nii')}\n"
        f"  Tip           : Check DATA_ROOT and BraTS folder layout."
    )

# =====================================================
# 8️⃣  Dataset Validation
# =====================================================

def validate_dataset(data_dir: Path) -> list:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {data_dir}\n"
            f"  Set DATA_ROOT env var to the correct path.\n"
            f"  Expected layout:\n"
            f"    <DATA_ROOT>/<case_id>/*_t1.nii(.gz)\n"
            f"    <DATA_ROOT>/<case_id>/*_t1ce.nii(.gz)\n"
            f"    <DATA_ROOT>/<case_id>/*_t2.nii(.gz)\n"
            f"    <DATA_ROOT>/<case_id>/*_flair.nii(.gz)\n"
            f"    <DATA_ROOT>/<case_id>/*_seg.nii(.gz)"
        )

    cases = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not cases:
        raise FileNotFoundError(
            f"No case sub-directories found in: {data_dir}"
        )

    data, skipped = [], []

    for case_path in cases:
        cp = str(case_path)
        try:
            entry = {
                "image": [
                    # Order is FIXED — must match inference order
                    safe_glob_one(os.path.join(cp, "*_t1.nii.gz"),    f"{case_path.name} T1"),
                    safe_glob_one(os.path.join(cp, "*_t1ce.nii.gz"),  f"{case_path.name} T1ce"),
                    safe_glob_one(os.path.join(cp, "*_t2.nii.gz"),    f"{case_path.name} T2"),
                    safe_glob_one(os.path.join(cp, "*_flair.nii.gz"), f"{case_path.name} FLAIR"),
                ],
                "label": safe_glob_one(
                    os.path.join(cp, "*_seg.nii.gz"), f"{case_path.name} seg"
                ),
            }
            data.append(entry)
        except FileNotFoundError as e:
            logger.warning(f"Skipping [{case_path.name}]: {e}")
            skipped.append(case_path.name)

    if not data:
        raise RuntimeError(
            f"All {len(cases)} cases skipped — check dataset structure."
        )

    logger.info(f"✓ Dataset OK — {len(data)} valid cases ({len(skipped)} skipped).")
    return data

# =====================================================
# 9️⃣  BraTS Label Remap  (4 → 3 for training)
# =====================================================

class RemapBraTSLabels(MapTransform):
    """
    BraTS uses labels: 0=background, 1=NCR, 2=ED, 4=ET
    We remap 4→3 so labels are contiguous: 0,1,2,3
    We reverse this (3→4) after inference.
    """
    def __call__(self, data):
        d = dict(data)
        label = d["label"]
        label[label == 4] = 3
        d["label"] = label
        return d

# =====================================================
# 🔟  Transforms
# =====================================================

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    RemapBraTSLabels(keys=["label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1, 1, 1),
             mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1, neg=1, num_samples=4,
    ),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    RemapBraTSLabels(keys=["label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1, 1, 1),
             mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
])

# Inference-only transforms (no label, no augmentation)
infer_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1, 1, 1), mode="bilinear"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
])

# =====================================================
# 1️⃣1️⃣  Tumor Report Generator
# =====================================================

# BraTS label definitions (after remapping 3→4 restored)
TUMOR_LABELS = {
    1: "NCR — Necrotic Core",
    2: "ED  — Peritumoral Edema",
    4: "ET  — Enhancing Tumor",
}

# WHO grade heuristic based on region presence
# Real grading requires histopathology — this is a decision-support hint only
def classify_tumor_grade(has_et: bool, has_ncr: bool, has_ed: bool) -> str:
    """
    Simplified WHO-based heuristic:
    - ET present           → likely High-Grade (WHO III–IV, Glioblastoma)
    - NCR + ED, no ET      → possibly High-Grade (WHO III)
    - ED only, no ET/NCR   → possibly Low-Grade (WHO II)
    - None present         → No tumor detected
    """
    if has_et:
        return "High-Grade Glioma (WHO Grade III–IV) — Enhancing tumor present ⚠️"
    elif has_ncr and has_ed:
        return "Likely High-Grade Glioma (WHO Grade III) — No enhancement detected"
    elif has_ed:
        return "Possibly Low-Grade Glioma (WHO Grade II) — Edema only"
    else:
        return "No significant tumor regions detected"

def generate_tumor_report(seg: np.ndarray,
                           voxel_spacing_mm: tuple = (1.0, 1.0, 1.0)) -> dict:
    """
    Given a segmentation mask (H, W, D) with BraTS labels (0,1,2,4),
    compute per-region volume in mm³ and cm³, total tumor volume,
    and a WHO grade heuristic.

    Returns a dict suitable for logging or displaying to a physician.
    """
    voxel_vol_mm3 = voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]

    report = {}
    region_volumes = {}

    for label_id, label_name in TUMOR_LABELS.items():
        voxel_count = int(np.sum(seg == label_id))
        vol_mm3     = voxel_count * voxel_vol_mm3
        vol_cm3     = vol_mm3 / 1000.0
        region_volumes[label_id] = vol_mm3
        report[label_name] = {
            "voxel_count": voxel_count,
            "volume_mm3":  round(vol_mm3,  2),
            "volume_cm3":  round(vol_cm3,  4),
        }

    # Whole Tumor = NCR + ED + ET  (labels 1, 2, 4)
    wt_voxels   = int(np.sum(seg > 0))
    wt_vol_mm3  = wt_voxels * voxel_vol_mm3
    report["Whole Tumor"] = {
        "voxel_count": wt_voxels,
        "volume_mm3":  round(wt_vol_mm3, 2),
        "volume_cm3":  round(wt_vol_mm3 / 1000.0, 4),
    }

    # Tumor Core = NCR + ET  (labels 1, 4)
    tc_voxels  = int(np.sum((seg == 1) | (seg == 4)))
    tc_vol_mm3 = tc_voxels * voxel_vol_mm3
    report["Tumor Core"] = {
        "voxel_count": tc_voxels,
        "volume_mm3":  round(tc_vol_mm3, 2),
        "volume_cm3":  round(tc_vol_mm3 / 1000.0, 4),
    }

    # Grade heuristic
    report["WHO Grade Heuristic"] = classify_tumor_grade(
        has_et  = region_volumes.get(4, 0) > 0,
        has_ncr = region_volumes.get(1, 0) > 0,
        has_ed  = region_volumes.get(2, 0) > 0,
    )

    report["⚠️ Clinical Disclaimer"] = (
        "This is a decision-support tool only. "
        "All findings must be confirmed by a qualified radiologist/oncologist."
    )

    return report

def print_tumor_report(report: dict):
    logger.info("=" * 55)
    logger.info("  🧠  TUMOR SEGMENTATION REPORT")
    logger.info("=" * 55)
    for key, val in report.items():
        if isinstance(val, dict):
            logger.info(f"  {key}:")
            logger.info(f"    Voxels   : {val['voxel_count']:,}")
            logger.info(f"    Volume   : {val['volume_mm3']:,.1f} mm³  "
                        f"({val['volume_cm3']:.2f} cm³)")
        else:
            logger.info(f"  {key}: {val}")
    logger.info("=" * 55)

# =====================================================
# 1️⃣2️⃣  PyFunc Production Model (saved to MLflow)
# =====================================================

if mlflow_pyfunc is not None:

    class BraTS_UNet_v1_PyFunc(mlflow_pyfunc.PythonModel):
        """
        Production MLflow PyFunc model.

        Input  : {"image": [t1_path, t1ce_path, t2_path, flair_path]}
                 All paths must be .nii or .nii.gz files on the same machine.
                 ORDER IS FIXED: t1 → t1ce → t2 → flair

        Output : dict with keys:
                   "segmentation" → np.ndarray (H, W, D) uint8, BraTS labels
                   "report"       → dict with volumes + grade heuristic
        """

        def load_context(self, context):
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.model = UNet(
                spatial_dims=3,
                in_channels=4,
                out_channels=4,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ).to(self.device)

            self.model.load_state_dict(
                torch.load(
                    context.artifacts["model_path"],
                    map_location=self.device,
                )
            )
            self.model.eval()
            logger.info(f"✓ Model loaded on {self.device}")

            self.preprocess = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1, 1, 1), mode="bilinear"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ])

        def predict(self, context, model_input: dict) -> dict:
            """
            Parameters
            ----------
            model_input : dict
                {
                  "image": [
                      "/path/BraTS20_001_t1.nii",      # channel 0
                      "/path/BraTS20_001_t1ce.nii",    # channel 1
                      "/path/BraTS20_001_t2.nii",      # channel 2
                      "/path/BraTS20_001_flair.nii",   # channel 3
                  ]
                }

            Returns
            -------
            dict
                {
                  "segmentation": np.ndarray (H, W, D) uint8,
                  "report":       dict  ← volumes + WHO grade heuristic
                }
            """
            # ── 1. Validate input ──────────────────
            if "image" not in model_input:
                raise ValueError(
                    "model_input must contain key 'image' with a list of 4 NIfTI paths.\n"
                    "  Order: [t1, t1ce, t2, flair]"
                )

            paths = model_input["image"]

            if len(paths) != 4:
                raise ValueError(
                    f"Expected exactly 4 NIfTI files [t1, t1ce, t2, flair], "
                    f"got {len(paths)}."
                )

            for i, p in enumerate(paths):
                modality = ["t1", "t1ce", "t2", "flair"][i]
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"NIfTI file not found for {modality} (channel {i}): {p}"
                    )

            # ── 2. Preprocess ──────────────────────
            logger.info("Running inference...")
            data  = self.preprocess({"image": paths})
            image = data["image"].unsqueeze(0).to(self.device)
            # shape → (1, 4, H, W, D)

            # ── 3. Sliding window inference ────────
            with torch.no_grad():
                logits = sliding_window_inference(
                    inputs      = image,
                    roi_size    = (96, 96, 96),
                    sw_batch_size = 1,
                    predictor   = self.model,
                    overlap     = 0.5,        # 0.5 = better quality than 0.25
                )
            # logits shape → (1, 4, H, W, D)

            # ── 4. Postprocess → class map ─────────
            probs  = torch.softmax(logits, dim=1)       # (1, 4, H, W, D)
            seg    = torch.argmax(probs, dim=1)[0]      # (H, W, D)
            seg_np = seg.cpu().numpy().astype("uint8")

            # ── 5. Restore original BraTS labels ───
            # Training remapped 4→3; restore 3→4
            seg_np[seg_np == 3] = 4

            # ── 6. Generate tumor report ───────────
            report = generate_tumor_report(seg_np, voxel_spacing_mm=VOXEL_SPACING_MM)
            print_tumor_report(report)

            return {
                "segmentation": seg_np,   # (H, W, D) uint8, labels: 0,1,2,4
                "report":       report,
            }

# =====================================================
# 1️⃣3️⃣  Standalone Inference Function
#         (use this when MLflow is NOT available)
# =====================================================

def run_inference(
    model_path: str,
    t1_path:    str,
    t1ce_path:  str,
    t2_path:    str,
    flair_path: str,
    output_path: str = "./prediction_seg.nii.gz",
) -> dict:
    """
    Standalone inference — no MLflow needed.
    Loads best_model.pth, runs prediction, saves NIfTI + returns report.

    Usage:
        result = run_inference(
            model_path  = "./checkpoints/best_model.pth",
            t1_path     = "/data/case001/BraTS20_001_t1.nii",
            t1ce_path   = "/data/case001/BraTS20_001_t1ce.nii",
            t2_path     = "/data/case001/BraTS20_001_t2.nii",
            flair_path  = "/data/case001/BraTS20_001_flair.nii",
            output_path = "./case001_seg.nii.gz",
        )
        print(result["report"])
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required to save NIfTI output.\n"
            "  Install: pip install nibabel"
        )

    device = validate_gpu()

    # ── Validate inputs ────────────────────────────
    paths = {
        "t1":    t1_path,
        "t1ce":  t1ce_path,
        "t2":    t2_path,
        "flair": flair_path,
    }
    for mod, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {mod} file: {p}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # ── Load model ─────────────────────────────────
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()
    logger.info(f"✓ Model loaded from {model_path}")

    # ── Preprocess ─────────────────────────────────
    image_paths = [t1_path, t1ce_path, t2_path, flair_path]
    data  = infer_transforms({"image": image_paths})
    image = data["image"].unsqueeze(0).to(device)

    # ── Inference ──────────────────────────────────
    logger.info("Running sliding window inference...")
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs        = image,
            roi_size      = (96, 96, 96),
            sw_batch_size = 1,
            predictor     = model,
            overlap       = 0.5,
        )

    probs  = torch.softmax(logits, dim=1)
    seg    = torch.argmax(probs, dim=1)[0]
    seg_np = seg.cpu().numpy().astype("uint8")
    seg_np[seg_np == 3] = 4          # restore BraTS label 4

    # ── Save NIfTI ─────────────────────────────────
    ref_nii  = nib.load(t1_path)
    seg_nii  = nib.Nifti1Image(seg_np, affine=ref_nii.affine,
                                header=ref_nii.header)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(seg_nii, output_path)
    logger.info(f"✓ Segmentation saved → {output_path}")

    # ── Tumor report ───────────────────────────────
    report = generate_tumor_report(seg_np, voxel_spacing_mm=VOXEL_SPACING_MM)
    print_tumor_report(report)

    return {
        "segmentation_path": output_path,
        "segmentation":      seg_np,
        "report":            report,
    }

# =====================================================
# 1️⃣4️⃣  Checkpoint Helper
# =====================================================

def save_checkpoint(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"  → Checkpoint saved : {path}")

# =====================================================
# 1️⃣5️⃣  Training
# =====================================================

def train():
    logger.info("=" * 60)
    logger.info("MONAI 3D UNet — BraTS Segmentation  (VM Edition)")
    logger.info("=" * 60)

    device = validate_gpu()
    data   = validate_dataset(DATA_DIR)

    random.seed(42)
    random.shuffle(data)

    split_idx   = int(len(data) * 0.8)
    train_files = data[:split_idx]
    val_files   = data[split_idx:]
    logger.info(f"Split → train : {len(train_files)}, val : {len(val_files)}")

    train_ds = Dataset(train_files, transform=train_transforms)
    val_ds   = Dataset(val_files,   transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_fn          = DiceCELoss(to_onehot_y=True, softmax=True,
                                  include_background=False)
    optimizer        = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    dice_metric_mean = DiceMetric(include_background=False, reduction="mean")
    dice_metric_none = DiceMetric(include_background=False, reduction="none")
    post_pred        = Activations(softmax=True)
    post_label       = AsDiscrete(to_onehot=4)

    best_dice         = 0.0
    epochs_no_improve = 0
    best_ckpt         = CHECKPOINT_DIR / "best_model.pth"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    run_ctx = mlflow.start_run() if mlflow else None

    try:
        if mlflow and run_ctx:
            mlflow.log_params({
                "architecture" : "UNet_3D",
                "dataset"      : "BraTS2020",
                "in_channels"  : 4,
                "out_channels" : 4,
                "lr"           : LEARNING_RATE,
                "batch_size"   : BATCH_SIZE,
                "max_epochs"   : MAX_EPOCHS,
                "patience"     : PATIENCE,
                "num_workers"  : NUM_WORKERS,
                "device"       : str(device),
                "loss"         : "DiceCELoss",
                "optimizer"    : "AdamW",
            })

        for epoch in range(MAX_EPOCHS):

            # ── Train ──────────────────────────────
            model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).long()

                optimizer.zero_grad()
                outputs = model(images)
                loss    = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)

            # ── Validation ─────────────────────────
            if (epoch + 1) % VAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch["image"].to(device)
                        labels = batch["label"].to(device)

                        outputs = model(images)
                        outputs = post_pred(outputs)
                        labels  = post_label(labels)

                        dice_metric_mean(outputs, labels)
                        dice_metric_none(outputs, labels)

                mean_dice      = dice_metric_mean.aggregate().item()
                per_class_dice = dice_metric_none.aggregate()[0]
                dice_metric_mean.reset()
                dice_metric_none.reset()

                if mlflow and run_ctx:
                    mlflow.log_metric("train_loss",    epoch_loss, step=epoch)
                    mlflow.log_metric("val_mean_dice", mean_dice,  step=epoch)
                    for i, val in enumerate(per_class_dice):
                        mlflow.log_metric(
                            f"val_dice_class_{i+1}", val.item(), step=epoch
                        )

                logger.info(
                    f"Epoch [{epoch+1}/{MAX_EPOCHS}]  "
                    f"Loss : {epoch_loss:.4f}  |  Val Dice : {mean_dice:.4f}  "
                    f"[NCR:{per_class_dice[0]:.3f} "
                    f"ED:{per_class_dice[1]:.3f} "
                    f"ET:{per_class_dice[2]:.3f}]"
                )

                # ── Save best ──────────────────────
                if mean_dice > best_dice + MIN_DELTA:
                    best_dice         = mean_dice
                    epochs_no_improve = 0
                    save_checkpoint(model, best_ckpt)

                    # Log PyFunc to MLflow
                    if mlflow and run_ctx and mlflow_pyfunc is not None:
                        mlflow_pyfunc.log_model(
                            artifact_path = MODEL_NAME,
                            python_model  = BraTS_UNet_v1_PyFunc(),
                            artifacts     = {"model_path": str(best_ckpt)},
                        )
                        logger.info("✔ Best model logged to MLflow.")
                    else:
                        logger.info("✔ Best model saved locally.")

                else:
                    epochs_no_improve += 1
                    logger.info(
                        f"  No improvement "
                        f"({epochs_no_improve}/{PATIENCE})"
                    )

                if epochs_no_improve >= PATIENCE:
                    logger.info("⛔ Early stopping triggered.")
                    break

            else:
                logger.info(
                    f"Epoch [{epoch+1}/{MAX_EPOCHS}]  Loss : {epoch_loss:.4f}"
                )
                if mlflow and run_ctx:
                    mlflow.log_metric("train_loss", epoch_loss, step=epoch)

        # ── Register model ─────────────────────────
        if mlflow and run_ctx and MlflowClient is not None:
            run_id    = run_ctx.info.run_id
            model_uri = f"runs:/{run_id}/{MODEL_NAME}"
            registered = mlflow.register_model(
                model_uri=model_uri, name="BraTS_UNet"
            )
            client = MlflowClient()
            client.transition_model_version_stage(
                name="BraTS_UNet",
                version=registered.version,
                stage="Staging",
            )
            logger.info(
                f"✔ Model v{registered.version} registered → Staging"
            )

    finally:
        if mlflow and run_ctx:
            mlflow.end_run()

    logger.info("=" * 60)
    logger.info(f"Training complete.  Best Val Dice : {best_dice:.4f}")
    logger.info(f"Best checkpoint   : {best_ckpt}")
    logger.info("=" * 60)

# =====================================================
# 1️⃣6️⃣  Entry Point
# =====================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BraTS UNet — Train or Infer"
    )
    parser.add_argument(
        "--mode", choices=["train", "infer"], default="train",
        help="'train' to train the model, 'infer' to run prediction on one patient"
    )
    # Inference-only args
    parser.add_argument("--model_path", default="./checkpoints/best_model.pth")
    parser.add_argument("--t1",    default=None)
    parser.add_argument("--t1ce", default=None)
    parser.add_argument("--t2",   default=None)
    parser.add_argument("--flair",default=None)
    parser.add_argument("--output", default="./prediction_seg.nii.gz")

    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "infer":
        for flag, val in [("--t1", args.t1), ("--t1ce", args.t1ce),
                          ("--t2", args.t2), ("--flair", args.flair)]:
            if val is None:
                raise ValueError(
                    f"{flag} is required for inference mode.\n"
                    f"  Example:\n"
                    f"    python train_brats_unet_final.py --mode infer \\\n"
                    f"      --model_path ./checkpoints/best_model.pth \\\n"
                    f"      --t1   /data/case001/BraTS20_001_t1.nii \\\n"
                    f"      --t1ce /data/case001/BraTS20_001_t1ce.nii \\\n"
                    f"      --t2   /data/case001/BraTS20_001_t2.nii \\\n"
                    f"      --flair /data/case001/BraTS20_001_flair.nii \\\n"
                    f"      --output ./case001_seg.nii.gz"
                )

        run_inference(
            model_path  = args.model_path,
            t1_path     = args.t1,
            t1ce_path   = args.t1ce,
            t2_path     = args.t2,
            flair_path  = args.flair,
            output_path = args.output,
        )