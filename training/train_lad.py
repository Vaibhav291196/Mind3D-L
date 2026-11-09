
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



import sys
import torch
from torch.optim import AdamW

from configs.config import (
    TRAIN_FILE,
    VAL_FILE,
    CLIP_FEATURE_FILE,
    CHECKPOINT_DIR,
    NORMALIZATION_PATH,
    NFE_HIDDEN_DIM,
    NFE_OUTPUT_DIM,
    NFE_TEMPORAL_LAYERS,
    FBDM_HIDDEN_DIM,
    FBDM_NUM_LAYERS,
    FBDM_NUM_HEADS,
    FBDM_TIMESTEPS,
    LAD_HIDDEN_DIM,
    LAD_NUM_LAYERS,
    NUM_POINTS,
    DROPOUT,
    LAD_BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    LAD_EPOCHS,
    LAD_LR,
    WEIGHT_DECAY,
    GRAD_CLIP,
    EARLY_STOPPING_PATIENCE,
    DIFFUSION_BETA_START,
    DIFFUSION_BETA_END,
    CHAMFER_WEIGHT,
    REPULSION_WEIGHT,
)

from models.nfe import NeuroFusionEncoder
from models.fbdm import FeatureBridgeDiffusion
from models.lad import LatentAdaptedDecoder

from losses.pointcloud_loss import PointCloudLoss

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.diffusion import DiffusionSchedule
from utils.early_stopping import EarlyStopping
from utils.seed import set_seed

from training.common import (
    create_dataloader,
    create_grad_scaler,
    create_autocast,
    get_device,
    move_batch_to_device,
)


def load_dataset_class():
    import importlib.util

    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "datasets" / "fmri_dataset.py"

    spec = importlib.util.spec_from_file_location(
        "mind3d_fmri_dataset",
        module_path
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules["mind3d_fmri_dataset"] = module
    spec.loader.exec_module(module)

    return module.FMRIDataset


def load_frozen_models(device):
    nfe = NeuroFusionEncoder(
        hidden_dim=NFE_HIDDEN_DIM,
        output_dim=NFE_OUTPUT_DIM,
        num_temporal_layers=NFE_TEMPORAL_LAYERS,
        dropout=DROPOUT
    ).to(device)

    fbdm = FeatureBridgeDiffusion(
        feature_dim=NFE_OUTPUT_DIM,
        hidden_dim=FBDM_HIDDEN_DIM,
        num_layers=FBDM_NUM_LAYERS,
        num_heads=FBDM_NUM_HEADS,
        num_timesteps=FBDM_TIMESTEPS,
        dropout=DROPOUT
    ).to(device)

    nfe_path = CHECKPOINT_DIR / "nfe_best.pt"
    fbdm_path = CHECKPOINT_DIR / "fbdm_best.pt"

    if not nfe_path.exists():
        raise FileNotFoundError(
            f"NFE checkpoint not found: {nfe_path}"
        )

    if not fbdm_path.exists():
        raise FileNotFoundError(
            f"FBDM checkpoint not found: {fbdm_path}"
        )

    load_checkpoint(nfe_path, nfe, device=device)
    load_checkpoint(fbdm_path, fbdm, device=device)

    nfe.eval()
    fbdm.eval()

    for model in (nfe, fbdm):
        for parameter in model.parameters():
            parameter.requires_grad = False

    return nfe, fbdm


def run_epoch(
    nfe,
    fbdm,
    lad,
    loader,
    schedule,
    loss_function,
    optimizer,
    device,
    scaler,
    training=True
):
    nfe.eval()
    fbdm.eval()
    lad.train(training)

    total_loss = 0.0
    total_chamfer = 0.0
    total_repulsion = 0.0
    batch_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        fmri = batch["fmri"]
        target_points = batch["points"]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with create_autocast(device):
                neuro_features, _ = nfe(fmri)

                visual_features = schedule.sample(
                    fbdm,
                    neuro_features,
                    feature_dim=NFE_OUTPUT_DIM
                )

        with torch.set_grad_enabled(training):
            with create_autocast(device):
                predicted_points = lad(
                    neuro_features,
                    visual_features
                )

            losses = loss_function(
                predicted_points.float(),
                target_points.float()
            )

            loss = losses["total"]

            if training:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    lad.parameters(),
                    GRAD_CLIP
                )

                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.detach().item()
        total_chamfer += losses["chamfer"].detach().item()
        total_repulsion += losses["repulsion"].detach().item()
        batch_count += 1

    denominator = max(batch_count, 1)

    return {
        "total": total_loss / denominator,
        "chamfer": total_chamfer / denominator,
        "repulsion": total_repulsion / denominator,
    }


def train_lad():
    set_seed()

    device = get_device()
    print(f"Training LAD on: {device}")

    FMRIDataset = load_dataset_class()

    train_dataset = FMRIDataset(
        npz_path=TRAIN_FILE,
        split="train",
        clip_feature_path=CLIP_FEATURE_FILE,
        use_clip_features=True,
        use_normalization=True,
        normalization_path=NORMALIZATION_PATH
    )

    val_dataset = FMRIDataset(
        npz_path=VAL_FILE,
        split="val",
        clip_feature_path=CLIP_FEATURE_FILE,
        use_clip_features=True,
        use_normalization=True,
        normalization_path=NORMALIZATION_PATH
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=LAD_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=LAD_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    nfe, fbdm = load_frozen_models(device)

    schedule = DiffusionSchedule(
        num_timesteps=FBDM_TIMESTEPS,
        beta_start=DIFFUSION_BETA_START,
        beta_end=DIFFUSION_BETA_END,
        device=device
    )

    lad = LatentAdaptedDecoder(
        condition_dim=NFE_OUTPUT_DIM,
        hidden_dim=LAD_HIDDEN_DIM,
        num_points=NUM_POINTS,
        num_layers=LAD_NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    loss_function = PointCloudLoss(
        chamfer_weight=CHAMFER_WEIGHT,
        repulsion_weight=REPULSION_WEIGHT
    )

    optimizer = AdamW(
        lad.parameters(),
        lr=LAD_LR,
        weight_decay=WEIGHT_DECAY
    )

    scaler = create_grad_scaler(device)
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE
    )

    best_loss = float("inf")
    checkpoint_path = CHECKPOINT_DIR / "lad_best.pt"

    for epoch in range(1, LAD_EPOCHS + 1):
        train_metrics = run_epoch(
            nfe,
            fbdm,
            lad,
            train_loader,
            schedule,
            loss_function,
            optimizer,
            device,
            scaler,
            training=True
        )

        val_metrics = run_epoch(
            nfe,
            fbdm,
            lad,
            val_loader,
            schedule,
            loss_function,
            None,
            device,
            scaler,
            training=False
        )

        print(
            f"Epoch {epoch:03d}/{LAD_EPOCHS} | "
            f"Train: {train_metrics['total']:.5f} "
            f"(CD: {train_metrics['chamfer']:.5f}, "
            f"Rep: {train_metrics['repulsion']:.5f}) | "
            f"Val: {val_metrics['total']:.5f} "
            f"(CD: {val_metrics['chamfer']:.5f}, "
            f"Rep: {val_metrics['repulsion']:.5f})"
        )

        if val_metrics["total"] < best_loss:
            best_loss = val_metrics["total"]

            save_checkpoint(
                checkpoint_path,
                lad,
                optimizer,
                epoch=epoch,
                best_loss=best_loss,
                extra={"stage": "LAD"}
            )

            print("  Best LAD checkpoint saved.")

        if early_stopping(val_metrics["total"]):
            print("Early stopping triggered.")
            break

    print(f"Best validation loss: {best_loss:.5f}")
    print(f"Checkpoint: {checkpoint_path}")

    return lad


if __name__ == "__main__":
    train_lad()
