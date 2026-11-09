
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/endoscopy_Gaussian_splat/FMRI-3D/Mind3D-L"
)

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

from configs.config import (
    DATA_DIR,
    TRAIN_FILE,
    VAL_FILE,
    CLIP_FEATURE_FILE,
    CHECKPOINT_DIR,
    SEED,
    NORMALIZATION_PATH,
)

from models.stage1_encoder import (
    FMRIPatchEncoder,
    Stage1FeatureAggregation,
    Stage1CLIPProjection,
)

from models.stage1_diffusion import Stage1DiffusionPrior

from losses.stage1_losses import (
    multi_positive_clip_loss,
    diffusion_mse_loss,
    combine_stage1_losses,
)

from utils.seed import set_seed
from utils.checkpoint import save_checkpoint


# Joint Stage 1 training settings
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = (
    BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)

NUM_EPOCHS = 100
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 150
EARLY_STOPPING_PATIENCE = 15

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

DIFFUSION_WEIGHT = 1.0
CLIP_WEIGHT = 1.0

USE_AMP = torch.cuda.is_available()
GRADIENT_CLIP = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

VOXEL_FILE = DATA_DIR / "voxels.npz"

BEST_CHECKPOINT = CHECKPOINT_DIR / "stage1_joint_best.pth"
LAST_CHECKPOINT = CHECKPOINT_DIR / "stage1_joint_last.pth"


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset_class():

    dataset_file = (
        PROJECT_ROOT / "datasets" / "fmri_dataset.py"
    )

    spec = importlib.util.spec_from_file_location(
        "mind3d_fmri_dataset",
        dataset_file,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.FMRIDataset


def create_datasets():

    FMRIDataset = load_dataset_class()

    common_args = {
        "clip_feature_path": CLIP_FEATURE_FILE,
        "use_clip_features": True,
        "use_normalization": True,
        "normalization_path": NORMALIZATION_PATH,
        "use_spatial_mapping": True,
        "voxel_path": VOXEL_FILE,
    }

    train_dataset = FMRIDataset(
        npz_path=TRAIN_FILE,
        split="train",
        **common_args,
    )

    val_dataset = FMRIDataset(
        npz_path=VAL_FILE,
        split="val",
        **common_args,
    )

    return train_dataset, val_dataset


def create_loaders(train_dataset, val_dataset):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader


# ============================================================
# STAGE 1 JOINT MODEL
# ============================================================

class Stage1JointModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.fmri_encoder = FMRIPatchEncoder()

        self.feature_aggregation = (
            Stage1FeatureAggregation()
        )

        self.clip_projection = Stage1CLIPProjection()

        self.diffusion_prior = Stage1DiffusionPrior()

        # Learnable CLIP temperature parameter.
        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / 0.07).log()
        )

    def encode_fmri(self, fmri):

        cls_features, spatial_features = (
            self.fmri_encoder(fmri)
        )

        aggregated_features = (
            self.feature_aggregation(spatial_features)
        )

        global_feature = (
            self.clip_projection(cls_features)
        )

        return aggregated_features, global_feature

    def forward_train(self, fmri, clip_features):

        spatial_features, global_feature = (
            self.encode_fmri(fmri)
        )

        _, predicted_x0 = self.diffusion_prior(
            spatial_features,
            clip_features,
        )

        return (
            spatial_features,
            global_feature,
            predicted_x0,
        )

    def forward_validation(self, fmri, clip_features):

        _, global_feature = self.encode_fmri(fmri)

        # Preserve the original MinD-3D validation pathway:
        # global fMRI feature is used as a single conditioning token.
        validation_condition = global_feature.unsqueeze(1)

        _, predicted_x0 = self.diffusion_prior(
            validation_condition,
            clip_features,
        )

        return global_feature, predicted_x0


# ============================================================
# LOSS CALCULATION
# ============================================================

def calculate_clip_loss(
    global_feature,
    clip_features,
    group_ids,
    model,
):

    # Contrastive learning requires at least two distinct groups.
    # Diffusion training continues even when this condition fails.
    if (
        global_feature.shape[0] < 2
        or torch.unique(group_ids).numel() < 2
    ):
        return global_feature.float().sum() * 0.0

    return multi_positive_clip_loss(
        global_feature,
        clip_features,
        group_ids,
        logit_scale=model.logit_scale,
    )


def calculate_total_loss(
    predicted_x0,
    clip_features,
    global_feature,
    group_ids,
    model,
):

    diffusion_loss = diffusion_mse_loss(
        predicted_x0,
        clip_features,
    )

    clip_loss = calculate_clip_loss(
        global_feature,
        clip_features,
        group_ids,
        model,
    )

    losses = combine_stage1_losses(
        diffusion_loss,
        clip_loss,
        diffusion_weight=DIFFUSION_WEIGHT,
        clip_weight=CLIP_WEIGHT,
    )

    return losses


# ============================================================
# OPTIMIZER AND SCHEDULER
# ============================================================

def create_optimizer(model):

    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if (
            parameter.ndim == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
        ):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    return optimizer


def create_scheduler(optimizer):

    def lr_lambda(step):

        if step < WARMUP_STEPS:
            return float(step + 1) / float(
                max(1, WARMUP_STEPS)
            )

        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


# ============================================================
# TRAINING HELPERS
# ============================================================

def move_batch_to_device(batch):

    return {
        key: value.to(
            DEVICE,
            non_blocking=True,
        ) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def autocast_context():

    return torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.float16,
        enabled=USE_AMP,
    )


def create_grad_scaler():

    return torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )


def clamp_logit_scale(model):

    with torch.no_grad():

        model.logit_scale.clamp_(
            max=np.log(100.0)
        )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    epoch,
):

    model.train()

    diffusion_total = 0.0
    clip_total = 0.0

    diffusion_batches = 0
    clip_batches = 0
    skipped_clip_batches = 0

    optimizer.zero_grad(set_to_none=True)

    num_batches = len(train_loader)

    for batch_index, batch in enumerate(train_loader):

        batch = move_batch_to_device(batch)

        fmri = batch["fmri"]
        clip_features = batch["clip_features"].float()
        group_ids = batch["group"].long()

        # Correct accumulation scaling for a final incomplete group.
        accumulation_group_start = (
            batch_index // GRADIENT_ACCUMULATION_STEPS
        ) * GRADIENT_ACCUMULATION_STEPS

        current_accumulation_size = min(
            GRADIENT_ACCUMULATION_STEPS,
            num_batches - accumulation_group_start,
        )

        with autocast_context():

            (
                spatial_features,
                global_feature,
                predicted_x0,
            ) = model.forward_train(
                fmri,
                clip_features,
            )

            losses = calculate_total_loss(
                predicted_x0,
                clip_features,
                global_feature,
                group_ids,
                model,
            )

            total_loss = losses["total_loss"]

            scaled_loss = (
                total_loss / current_accumulation_size
            )

        scaler.scale(scaled_loss).backward()

        is_accumulation_step = (
            (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0
        )

        is_last_batch = (
            batch_index + 1 == num_batches
        )

        if is_accumulation_step or is_last_batch:

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            clamp_logit_scale(model)

            scheduler.step()

        diffusion_value = (
            losses["diffusion_loss"].detach().item()
        )

        clip_value = (
            losses["clip_loss"].detach().item()
        )

        diffusion_total += diffusion_value
        diffusion_batches += 1

        if torch.unique(group_ids).numel() >= 2:

            clip_total += clip_value
            clip_batches += 1

        else:

            skipped_clip_batches += 1

        if (batch_index + 1) % 20 == 0:

            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_index + 1}/{num_batches} | "
                f"Diffusion: {diffusion_value:.6f} | "
                f"CLIP: {clip_value:.6f}"
            )

    mean_diffusion = (
        diffusion_total / max(1, diffusion_batches)
    )

    # Do not divide CLIP loss by batches where it was skipped.
    mean_clip = (
        clip_total / clip_batches
        if clip_batches > 0
        else 0.0
    )

    mean_total = (
        DIFFUSION_WEIGHT * mean_diffusion
        + CLIP_WEIGHT * mean_clip
    )

    return {
        "diffusion_loss": mean_diffusion,
        "clip_loss": mean_clip,
        "total_loss": mean_total,
        "clip_batches": clip_batches,
        "skipped_clip_batches": skipped_clip_batches,
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    val_loader,
    epoch,
):

    model.eval()

    # Make validation noise/timestep sampling reproducible per epoch.
    torch.manual_seed(SEED + epoch)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + epoch)

    diffusion_total = 0.0
    clip_total = 0.0

    diffusion_batches = 0
    clip_batches = 0
    skipped_clip_batches = 0

    for batch in val_loader:

        batch = move_batch_to_device(batch)

        fmri = batch["fmri"]
        clip_features = batch["clip_features"].float()
        group_ids = batch["group"].long()

        with autocast_context():

            global_feature, predicted_x0 = (
                model.forward_validation(
                    fmri,
                    clip_features,
                )
            )

            losses = calculate_total_loss(
                predicted_x0,
                clip_features,
                global_feature,
                group_ids,
                model,
            )

        diffusion_total += (
            losses["diffusion_loss"].item()
        )

        diffusion_batches += 1

        if torch.unique(group_ids).numel() >= 2:

            clip_total += losses["clip_loss"].item()
            clip_batches += 1

        else:

            skipped_clip_batches += 1

    mean_diffusion = (
        diffusion_total / max(1, diffusion_batches)
    )

    mean_clip = (
        clip_total / clip_batches
        if clip_batches > 0
        else 0.0
    )

    mean_total = (
        DIFFUSION_WEIGHT * mean_diffusion
        + CLIP_WEIGHT * mean_clip
    )

    return {
        "diffusion_loss": mean_diffusion,
        "clip_loss": mean_clip,
        "total_loss": mean_total,
        "clip_batches": clip_batches,
        "skipped_clip_batches": skipped_clip_batches,
    }


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_stage1():

    print("=" * 65)
    print("Mind3D-L | Joint Stage 1 Training")
    print("=" * 65)

    print(f"Device: {DEVICE}")
    print(f"AMP enabled: {USE_AMP}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_dataset, val_dataset = create_datasets()

    train_loader, val_loader = create_loaders(
        train_dataset,
        val_dataset,
    )

    print(f"\nTraining scans: {len(train_dataset)}")
    print(f"Validation scans: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = Stage1JointModel().to(DEVICE)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\nTotal parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    optimizer = create_optimizer(model)

    scheduler = create_scheduler(optimizer)

    scaler = create_grad_scaler()

    best_val_loss = float("inf")
    patience_counter = 0

    history = []

    # --------------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------------

    for epoch in range(NUM_EPOCHS):

        print("\n" + "=" * 65)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("=" * 65)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            epoch,
        )

        val_metrics = validate(
            model,
            val_loader,
            epoch,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print("\nTraining:")
        print(
            f"  Diffusion loss: "
            f"{train_metrics['diffusion_loss']:.6f}"
        )
        print(
            f"  CLIP loss: "
            f"{train_metrics['clip_loss']:.6f}"
        )
        print(
            f"  Total loss: "
            f"{train_metrics['total_loss']:.6f}"
        )
        print(
            f"  CLIP batches: "
            f"{train_metrics['clip_batches']}"
        )
        print(
            f"  Skipped CLIP batches: "
            f"{train_metrics['skipped_clip_batches']}"
        )

        print("\nValidation:")
        print(
            f"  Diffusion loss: "
            f"{val_metrics['diffusion_loss']:.6f}"
        )
        print(
            f"  CLIP loss: "
            f"{val_metrics['clip_loss']:.6f}"
        )
        print(
            f"  Total loss: "
            f"{val_metrics['total_loss']:.6f}"
        )
        print(
            f"  CLIP batches: "
            f"{val_metrics['clip_batches']}"
        )
        print(
            f"  Skipped CLIP batches: "
            f"{val_metrics['skipped_clip_batches']}"
        )

        print(f"\nLearning rate: {current_lr:.8e}")
        print(
            f"Logit scale: "
            f"{model.logit_scale.detach().exp().item():.4f}"
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        epoch_record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": val_metrics,
            "learning_rate": current_lr,
        }

        history.append(epoch_record)

        # ----------------------------------------------------
        # CHECKPOINTS
        # ----------------------------------------------------

        checkpoint_extra = {
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "history": history,
            "configuration": {
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps":
                    GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size":
                    EFFECTIVE_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": WARMUP_STEPS,
                "diffusion_weight": DIFFUSION_WEIGHT,
                "clip_weight": CLIP_WEIGHT,
            },
        }

        save_checkpoint(
            path=LAST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_loss=best_val_loss,
            extra=checkpoint_extra,
        )

        if val_metrics["total_loss"] < best_val_loss:

            best_val_loss = val_metrics["total_loss"]
            patience_counter = 0

            save_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_loss=best_val_loss,
                extra=checkpoint_extra,
            )

            print("\nBest checkpoint updated.")

        else:

            patience_counter += 1

            print(
                f"\nNo validation improvement: "
                f"{patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print("\nEarly stopping triggered.")
            break

        if torch.cuda.is_available():

            print(
                f"GPU memory allocated: "
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )

            print(
                f"GPU memory reserved: "
                f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )

    print("\n" + "=" * 65)
    print("Stage 1 training finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {BEST_CHECKPOINT}")
    print(f"Last checkpoint: {LAST_CHECKPOINT}")
    print("=" * 65)

    return model, history


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train_stage1()
