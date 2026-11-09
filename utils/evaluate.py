
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import (
    VAL_FILE,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
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
    INFERENCE_BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    DIFFUSION_BETA_START,
    DIFFUSION_BETA_END,
)

from models.nfe import NeuroFusionEncoder
from models.fbdm import FeatureBridgeDiffusion
from models.lad import LatentAdaptedDecoder

from utils.checkpoint import load_checkpoint
from utils.diffusion import DiffusionSchedule
from utils.metrics import chamfer_distance, f_score
from utils.seed import set_seed


def load_dataset_class():
    import importlib.util
    import sys
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


def load_models(device):
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

    lad = LatentAdaptedDecoder(
        condition_dim=NFE_OUTPUT_DIM,
        hidden_dim=LAD_HIDDEN_DIM,
        num_points=NUM_POINTS,
        num_layers=LAD_NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    for name, model in [
        ("nfe_best.pt", nfe),
        ("fbdm_best.pt", fbdm),
        ("lad_best.pt", lad),
    ]:
        checkpoint_path = CHECKPOINT_DIR / name

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        load_checkpoint(
            checkpoint_path,
            model,
            device=device
        )

        model.eval()

    return nfe, fbdm, lad


@torch.no_grad()
def evaluate_validation_set():
    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Evaluation device: {device}")

    FMRIDataset = load_dataset_class()

    val_dataset = FMRIDataset(
        npz_path=VAL_FILE,
        split="val",
        use_clip_features=False,
        use_normalization=True,
        normalization_path=NORMALIZATION_PATH
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    nfe, fbdm, lad = load_models(device)

    schedule = DiffusionSchedule(
        num_timesteps=FBDM_TIMESTEPS,
        beta_start=DIFFUSION_BETA_START,
        beta_end=DIFFUSION_BETA_END,
        device=device
    )

    all_cd = []
    all_f_scores = []
    all_scan_indices = []

    for batch_number, batch in enumerate(val_loader, start=1):
        fmri = batch["fmri"].to(device, non_blocking=True)
        target_points = batch["points"].to(
            device,
            non_blocking=True
        )

        scan_indices = batch["scan_index"].cpu().numpy()

        neuro_features, _ = nfe(fmri)

        visual_features = schedule.sample(
            fbdm,
            neuro_features,
            feature_dim=NFE_OUTPUT_DIM
        )

        predicted_points = lad(
            neuro_features,
            visual_features
        )

        for index in range(predicted_points.shape[0]):
            predicted = predicted_points[index:index + 1].float()
            target = target_points[index:index + 1].float()

            cd = chamfer_distance(predicted, target)
            fscore = f_score(
                predicted,
                target,
                threshold=0.05
            )

            all_cd.append(float(cd.item()))
            all_f_scores.append(float(fscore.item()))

        all_scan_indices.extend(scan_indices.tolist())

        print(
            f"Evaluated batch {batch_number}/{len(val_loader)}"
        )

    # Keep evaluation records in original validation scan order.
    order = np.argsort(all_scan_indices)

    all_cd = np.asarray(all_cd)[order]
    all_f_scores = np.asarray(all_f_scores)[order]

    metrics = {
        "num_validation_scans": len(all_cd),
        "chamfer_distance_mean": float(np.mean(all_cd)),
        "chamfer_distance_std": float(np.std(all_cd)),
        "f_score_threshold_0.05_mean": float(np.mean(all_f_scores)),
        "f_score_threshold_0.05_std": float(np.std(all_f_scores)),
        "silhouette_iou": None,
        "ssim": None,
        "note": (
            "Silhouette IoU and SSIM require a defined rendering "
            "and image-comparison procedure; they have not been "
            "computed by this point-cloud-only evaluator."
        )
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUT_DIR / "val_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nValidation metrics:")
    print(json.dumps(metrics, indent=4))
    print(f"\nSaved: {metrics_path}")

    return metrics


if __name__ == "__main__":
    evaluate_validation_set()
