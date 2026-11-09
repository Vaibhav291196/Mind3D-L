
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



import sys
import numpy as np
import torch
import torch.nn.functional as F
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
    DROPOUT,
    NFE_BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    NFE_EPOCHS,
    NFE_LR,
    WEIGHT_DECAY,
    GRAD_CLIP,
    #EARLY_STOPPING_PATIENCE,
    CONTRASTIVE_TEMPERATURE,
    DEVICE,
)

from models.nfe import NeuroFusionEncoder
from utils.checkpoint import save_checkpoint
# from utils.early_stopping import EarlyStopping
from utils.seed import set_seed
from training.common import (
    create_dataloader,
    create_grad_scaler,
    create_autocast,
    get_device,
    move_batch_to_device,
)


def load_dataset_class():
    """
    Load the local dataset module explicitly to avoid conflicts
    with the installed Hugging Face 'datasets' package.
    """
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


def multi_positive_contrastive_loss(
    neuro_features,
    visual_features,
    group_ids,
    temperature=0.07
):
    """
    Symmetric multi-positive contrastive loss.

    All samples belonging to the same object are treated as positives.
    This avoids treating repeated scans of one object as negatives.
    """
    neuro_features = F.normalize(neuro_features.float(), dim=1)
    visual_features = F.normalize(visual_features.float(), dim=1)

    logits = (
        neuro_features @ visual_features.T
    ) / temperature

    positive_mask = (
        group_ids[:, None] == group_ids[None, :]
    ).float()

    log_prob_neuro = F.log_softmax(logits, dim=1)
    log_prob_visual = F.log_softmax(logits.T, dim=1)

    positive_count = positive_mask.sum(dim=1).clamp_min(1.0)

    loss_neuro = -(
        positive_mask * log_prob_neuro
    ).sum(dim=1) / positive_count

    loss_visual = -(
        positive_mask.T * log_prob_visual
    ).sum(dim=1) / positive_count

    return 0.5 * (loss_neuro.mean() + loss_visual.mean())


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    training=True
):
    model.train(training)

    total_loss = 0.0
    batch_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        fmri = batch["fmri"]
        visual_features = batch["clip_features"]
        group_ids = batch["group"]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with create_autocast(device):
                _, alignment_features = model(fmri)

            loss = multi_positive_contrastive_loss(
                alignment_features,
                visual_features,
                group_ids,
                temperature=CONTRASTIVE_TEMPERATURE
            )

            if training:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )

                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.detach().item()
        batch_count += 1

    return total_loss / max(batch_count, 1)


def train_nfe():
    set_seed()

    device = get_device()
    print(f"Training NFE on: {device}")

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
        batch_size=NFE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=NFE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model = NeuroFusionEncoder(
        hidden_dim=NFE_HIDDEN_DIM,
        output_dim=NFE_OUTPUT_DIM,
        num_temporal_layers=NFE_TEMPORAL_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=NFE_LR,
        weight_decay=WEIGHT_DECAY
    )

    scaler = create_grad_scaler(device)
    # early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    best_loss = float("inf")
    checkpoint_path = CHECKPOINT_DIR / "nfe_best.pt"

    for epoch in range(1, NFE_EPOCHS + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            training=True
        )

        val_loss = run_epoch(
            model,
            val_loader,
            None,
            device,
            scaler,
            training=False
        )

        print(
            f"Epoch {epoch:03d}/{NFE_EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=epoch,
                best_loss=best_loss,
                extra={"stage": "NFE"}
            )

            print("  Best NFE checkpoint saved.")

        # if early_stopping(val_loss):
            # print("Early stopping triggered.")
            # break

    print(f"Best validation loss: {best_loss:.5f}")
    print(f"Checkpoint: {checkpoint_path}")

    return checkpoint_path


if __name__ == "__main__":
    train_nfe()
