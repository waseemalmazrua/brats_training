# =====================================================
# MONAI 3D UNet — BraTS Segmentation
# Target : Standard Ubuntu GPU VM (e.g. DigitalOcean)
# Changes: Removed GCS/Vertex paths, safe glob, optional
#          MLflow, explicit GPU check, local checkpoints,
#          safe num_workers.
# =====================================================

# =====================================================
# 1️⃣  Standard Libraries
# =====================================================

import os
import sys
import glob
import random
import logging
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
# 4️⃣  Config — all values overridable via env vars
#      No hard-coded cloud paths.
# =====================================================

# ── Paths ──────────────────────────────────────────
DATA_DIR        = Path(os.environ.get("DATA_ROOT",
                        "/data/brats/BraTS2020_TrainingData"))
CHECKPOINT_DIR  = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints"))

# ── Training ───────────────────────────────────────
MAX_EPOCHS      = int(os.environ.get("MAX_EPOCHS",   "100"))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE",   "2"))
LEARNING_RATE   = float(os.environ.get("LEARNING_RATE", "1e-4"))
PATIENCE        = int(os.environ.get("PATIENCE",     "15"))
MIN_DELTA       = float(os.environ.get("MIN_DELTA",  "0.001"))
VAL_INTERVAL    = int(os.environ.get("VAL_INTERVAL", "1"))

# ── Workers — safe for Linux VM ────────────────────
# Cap at (CPU count - 1) so the OS isn't starved.
_cpu_cap        = max(1, (os.cpu_count() or 2) - 1)
NUM_WORKERS     = int(os.environ.get("NUM_WORKERS", str(min(4, _cpu_cap))))

# ── GPU ────────────────────────────────────────────
REQUIRE_GPU     = os.environ.get("REQUIRE_GPU", "true").lower() != "false"

# ── MLflow — fully optional ────────────────────────
#    Set MLFLOW_TRACKING_URI in your shell to enable.
#    Leave it unset (or empty) to skip MLflow entirely.
MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
MODEL_NAME      = os.environ.get("MODEL_NAME", "brats_unet_v1")

# =====================================================
# 5️⃣  MLflow — optional import & setup
# =====================================================

mlflow          = None   # sentinel
mlflow_pyfunc   = None
MlflowClient    = None

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
    """
    Print GPU info. Raise clearly if GPU required but absent.
    Returns the torch.device to use.
    """
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        name  = torch.cuda.get_device_name(0)
        vram  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"✓ GPU detected : {name}  |  VRAM : {vram:.1f} GB")
    else:
        logger.warning("No CUDA GPU detected — would run on CPU.")
        if REQUIRE_GPU:
            raise RuntimeError(
                "REQUIRE_GPU=true but no CUDA GPU is available.\n"
                "  • Check nvidia-smi / CUDA installation.\n"
                "  • Set REQUIRE_GPU=false to allow CPU-only training."
            )
    return torch.device("cuda" if has_gpu else "cpu")

# =====================================================
# 7️⃣  Safe Glob Helper
# =====================================================

def safe_glob_one(pattern: str, description: str) -> str:
    """
    Return the first match for `pattern`.
    Raises FileNotFoundError with a clear message instead of
    IndexError when the list is empty (glob()[0] anti-pattern).
    """
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Missing file — {description}\n"
            f"  Pattern : {pattern}\n"
            f"  Tip     : Verify DATA_ROOT and BraTS directory layout."
        )
    return matches[0]

# =====================================================
# 8️⃣  Dataset Path Validation
# =====================================================

def validate_dataset(data_dir: Path) -> list:
    """
    Validate the BraTS dataset directory and build the
    list of {image, label} dicts.  Fails with a clear
    message before any training starts.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {data_dir}\n"
            f"  Create it or set DATA_ROOT to the correct path.\n"
            f"  Expected layout:\n"
            f"    <DATA_ROOT>/<case_id>/*_t1.nii.gz\n"
            f"    <DATA_ROOT>/<case_id>/*_t1ce.nii.gz\n"
            f"    <DATA_ROOT>/<case_id>/*_t2.nii.gz\n"
            f"    <DATA_ROOT>/<case_id>/*_flair.nii.gz\n"
            f"    <DATA_ROOT>/<case_id>/*_seg.nii.gz"
        )

    cases = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir()
    ])

    if not cases:
        raise FileNotFoundError(
            f"No case sub-directories found in: {data_dir}\n"
            "  Each BraTS case must be its own folder."
        )

    data = []
    skipped = []

    for case_path in cases:
        cp = str(case_path)
        try:
            entry = {
                "image": [
                    safe_glob_one(os.path.join(cp, "*_t1.nii.gz"),    f"{case_path.name} T1"),
                    safe_glob_one(os.path.join(cp, "*_t1ce.nii.gz"),  f"{case_path.name} T1ce"),
                    safe_glob_one(os.path.join(cp, "*_t2.nii.gz"),    f"{case_path.name} T2"),
                    safe_glob_one(os.path.join(cp, "*_flair.nii.gz"), f"{case_path.name} FLAIR"),
                ],
                "label": safe_glob_one(os.path.join(cp, "*_seg.nii.gz"), f"{case_path.name} seg"),
            }
            data.append(entry)
        except FileNotFoundError as e:
            logger.warning(f"Skipping incomplete case [{case_path.name}]: {e}")
            skipped.append(case_path.name)

    if not data:
        raise RuntimeError(
            f"All {len(cases)} cases were skipped due to missing files.\n"
            "  Check your dataset structure."
        )

    logger.info(
        f"✓ Dataset OK — {len(data)} valid cases "
        f"({len(skipped)} skipped)."
    )
    return data

# =====================================================
# 9️⃣  Remap BraTS Labels  (class 4 → 3)
# =====================================================

class RemapBraTSLabels(MapTransform):
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

# =====================================================
# 1️⃣1️⃣  PyFunc Production Wrapper  (only used w/ MLflow)
# =====================================================

if mlflow_pyfunc is not None:
    class BraTS_UNet_v1_PyFunc(mlflow_pyfunc.PythonModel):

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

            self.preprocess = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1, 1, 1), mode="bilinear"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ])
            self.postprocess = Compose([
                Activations(softmax=True),
                AsDiscrete(argmax=True),
            ])

        def predict(self, context, model_input):
            data = self.preprocess({"image": model_input})
            image = data["image"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = sliding_window_inference(
                    image, roi_size=(96, 96, 96),
                    sw_batch_size=1, predictor=self.model, overlap=0.25,
                )
            return self.postprocess(output).cpu().numpy()

# =====================================================
# 1️⃣2️⃣  Checkpoint helpers
# =====================================================

def save_checkpoint(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"  → Checkpoint saved : {path}")

# =====================================================
# 1️⃣3️⃣  Main training function
# =====================================================

def train():
    logger.info("=" * 60)
    logger.info("MONAI 3D UNet — BraTS Segmentation  (VM Edition)")
    logger.info("=" * 60)

    # ── GPU ──────────────────────────────────────
    device = validate_gpu()

    # ── Dataset ──────────────────────────────────
    data = validate_dataset(DATA_DIR)

    random.seed(42)
    random.shuffle(data)

    split_idx   = int(len(data) * 0.8)
    train_files = data[:split_idx]
    val_files   = data[split_idx:]
    logger.info(f"Split → train : {len(train_files)}, val : {len(val_files)}")

    # ── DataLoaders ───────────────────────────────
    train_ds = Dataset(train_files, transform=train_transforms)
    val_ds   = Dataset(val_files,   transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Model ─────────────────────────────────────
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_function    = DiceCELoss(to_onehot_y=True, softmax=True,
                                  include_background=False)
    optimizer        = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    dice_metric_mean = DiceMetric(include_background=False, reduction="mean")
    dice_metric_none = DiceMetric(include_background=False, reduction="none")

    post_pred  = Activations(softmax=True)
    post_label = AsDiscrete(to_onehot=4)

    # ── Training state ────────────────────────────
    best_dice        = 0.0
    epochs_no_improve= 0
    best_ckpt        = CHECKPOINT_DIR / "best_model.pth"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── MLflow run (optional) ─────────────────────
    run_ctx = mlflow.start_run() if mlflow else None

    try:
        if mlflow and run_ctx:
            mlflow.log_params({
                "architecture": "UNet_3D",
                "dataset":      "BraTS",
                "version":      "v1",
                "lr":           LEARNING_RATE,
                "batch_size":   BATCH_SIZE,
                "num_workers":  NUM_WORKERS,
                "device":       str(device),
            })

        for epoch in range(MAX_EPOCHS):

            # ────── TRAIN ──────
            model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).long()

                optimizer.zero_grad()
                outputs = model(images)
                loss    = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)

            # ────── VALIDATION ──────
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

                # ── Log ──
                if mlflow and run_ctx:
                    mlflow.log_metric("train_loss",    epoch_loss, step=epoch)
                    mlflow.log_metric("val_mean_dice", mean_dice,  step=epoch)
                    for i, val in enumerate(per_class_dice):
                        mlflow.log_metric(f"val_dice_class_{i+1}",
                                          val.item(), step=epoch)

                logger.info(
                    f"Epoch [{epoch+1}/{MAX_EPOCHS}]  "
                    f"Loss : {epoch_loss:.4f}  |  Val Dice : {mean_dice:.4f}"
                )

                # ── Early stopping / checkpoint ──
                if mean_dice > best_dice + MIN_DELTA:
                    best_dice         = mean_dice
                    epochs_no_improve = 0

                    save_checkpoint(model, best_ckpt)

                    if mlflow and run_ctx and mlflow_pyfunc is not None:
                        mlflow_pyfunc.log_model(
                            artifact_path=MODEL_NAME,
                            python_model=BraTS_UNet_v1_PyFunc(),
                            artifacts={"model_path": str(best_ckpt)},
                        )
                        logger.info("✔ Best model saved & logged to MLflow.")
                    else:
                        logger.info("✔ Best model saved locally.")

                else:
                    epochs_no_improve += 1
                    logger.info(
                        f"  No improvement ({epochs_no_improve}/{PATIENCE})"
                    )

                if epochs_no_improve >= PATIENCE:
                    logger.info("⛔ Early stopping triggered.")
                    break

            else:
                # Epoch without validation — just log loss
                logger.info(
                    f"Epoch [{epoch+1}/{MAX_EPOCHS}]  Loss : {epoch_loss:.4f}"
                )
                if mlflow and run_ctx:
                    mlflow.log_metric("train_loss", epoch_loss, step=epoch)

        # ── Register model (MLflow only) ──────────
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
            logger.info("Model registered → Staging")

    finally:
        if mlflow and run_ctx:
            mlflow.end_run()

    logger.info("=" * 60)
    logger.info(f"Training complete.  Best Val Dice : {best_dice:.4f}")
    logger.info(f"Best checkpoint   : {best_ckpt}")
    logger.info("=" * 60)


# =====================================================
# 1️⃣4️⃣  Entry Point
# =====================================================

if __name__ == "__main__":
    train()