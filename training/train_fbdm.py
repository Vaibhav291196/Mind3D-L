
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
    DROPOUT,
    FBDM_BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    FBDM_EPOCHS,
    FBDM_LR,
    WEIGHT_DECAY,
    GRAD_CLIP,
    EARLY_STOPPING_PATIENCE,
    DIFFUSION_BETA_START,
    DIFFUSION_BETA_END,
)

from models.nfe import NeuroFusionEncoder
from models.fbdm import FeatureBridgeDiffusion
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
    from pathlib import Path

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


def run_epoch(
    nfe,
    fbdm,
    loader,
    schedule,
    optimizer,
    device,
    scaler,
    training=True
):
    fbdm.train(training)
    nfe.eval()

    # Fixed random sequence for reproducible validation.
    validation_generator = None

    if not training:
        validation_generator = torch.Generator(device=device)
        validation_generator.manual_seed(2026)

    total_loss = 0.0
    batch_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        fmri = batch["fmri"]
        clean_visual_features = batch["clip_features"]

        batch_size = fmri.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with create_autocast(device):
                neuro_features, _ = nfe(fmri)

        if training:
            timesteps = torch.randint(
                low=0,
                high=schedule.num_timesteps,
                size=(batch_size,),
                device=device,
                dtype=torch.long
            )

            noise = torch.randn_like(clean_visual_features)

        else:
            timesteps = torch.randint(
                low=0,
                high=schedule.num_timesteps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
                generator=validation_generator
            )

            noise = torch.randn(
                clean_visual_features.shape,
                device=device,
                dtype=clean_visual_features.dtype,
                generator=validation_generator
            )

        noisy_features, actual_noise = schedule.add_noise(
            clean_visual_features,
            timesteps,
            noise=noise
        )

        with torch.set_grad_enabled(training):
            with create_autocast(device):
                predicted_noise = fbdm(
                    noisy_features,
                    neuro_features,
                    timesteps
                )

            loss = torch.nn.functional.mse_loss(
                predicted_noise.float(),
                actual_noise.float()
            )

            if training:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    fbdm.parameters(),
                    GRAD_CLIP
                )

                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.detach().item()
        batch_count += 1

    return total_loss / max(batch_count, 1)


def train_fbdm():
    set_seed()

    device = get_device()
    print(f"Training FBDM on: {device}")

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
        batch_size=FBDM_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=FBDM_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    nfe = NeuroFusionEncoder(
        hidden_dim=NFE_HIDDEN_DIM,
        output_dim=NFE_OUTPUT_DIM,
        num_temporal_layers=NFE_TEMPORAL_LAYERS,
        dropout=DROPOUT
    ).to(device)

    nfe_checkpoint = CHECKPOINT_DIR / "nfe_best.pt"

    if not nfe_checkpoint.exists():
        raise FileNotFoundError(
            f"NFE checkpoint not found: {nfe_checkpoint}. "
            "Train NFE before training FBDM."
        )

    load_checkpoint(
        nfe_checkpoint,
        nfe,
        device=device
    )

    nfe.eval()
    for parameter in nfe.parameters():
        parameter.requires_grad = False

    fbdm = FeatureBridgeDiffusion(
        feature_dim=NFE_OUTPUT_DIM,
        hidden_dim=FBDM_HIDDEN_DIM,
        num_layers=FBDM_NUM_LAYERS,
        num_heads=FBDM_NUM_HEADS,
        num_timesteps=FBDM_TIMESTEPS,
        dropout=DROPOUT
    ).to(device)

    schedule = DiffusionSchedule(
        num_timesteps=FBDM_TIMESTEPS,
        beta_start=DIFFUSION_BETA_START,
        beta_end=DIFFUSION_BETA_END,
        device=device
    )

    optimizer = AdamW(
        fbdm.parameters(),
        lr=FBDM_LR,
        weight_decay=WEIGHT_DECAY
    )

    scaler = create_grad_scaler(device)
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE
    )

    best_loss = float("inf")
    checkpoint_path = CHECKPOINT_DIR / "fbdm_best.pt"

    for epoch in range(1, FBDM_EPOCHS + 1):
        train_loss = run_epoch(
            nfe,
            fbdm,
            train_loader,
            schedule,
            optimizer,
            device,
            scaler,
            training=True
        )

        val_loss = run_epoch(
            nfe,
            fbdm,
            val_loader,
            schedule,
            None,
            device,
            scaler,
            training=False
        )

        print(
            f"Epoch {epoch:03d}/{FBDM_EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss

            save_checkpoint(
                checkpoint_path,
                fbdm,
                optimizer,
                epoch=epoch,
                best_loss=best_loss,
                extra={"stage": "FBDM"}
            )

            print("  Best FBDM checkpoint saved.")

        if early_stopping(val_loss):
            print("Early stopping triggered.")
            break

    print(f"Best validation loss: {best_loss:.5f}")
    print(f"Checkpoint: {checkpoint_path}")

    return fbdm


if __name__ == "__main__":
    train_fbdm()
