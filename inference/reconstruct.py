
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import (
    TEST_FILE,
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
from utils.seed import set_seed

from inference.export_ply import export_ply
from utils.visualize import render_four_views


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

    checkpoints = [
        (CHECKPOINT_DIR / "nfe_best.pt", nfe),
        (CHECKPOINT_DIR / "fbdm_best.pt", fbdm),
        (CHECKPOINT_DIR / "lad_best.pt", lad),
    ]

    for checkpoint_path, model in checkpoints:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Required checkpoint not found: {checkpoint_path}"
            )

        load_checkpoint(
            checkpoint_path,
            model,
            device=device
        )

        model.eval()

    return nfe, fbdm, lad


@torch.no_grad()
def reconstruct_test_set():
    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Inference device: {device}")

    FMRIDataset = load_dataset_class()

    test_dataset = FMRIDataset(
        npz_path=TEST_FILE,
        split="test",
        use_clip_features=False,
        use_normalization=True,
        normalization_path=NORMALIZATION_PATH
    )

    test_loader = DataLoader(
        test_dataset,
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

    all_predictions = []
    all_scan_indices = []

    for batch_number, batch in enumerate(test_loader, start=1):
        fmri = batch["fmri"].to(
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

        predicted_points = predicted_points.float().cpu().numpy()

        all_predictions.append(predicted_points)
        all_scan_indices.append(scan_indices)

        print(
            f"Processed batch {batch_number}/{len(test_loader)} "
            f"| scans: {len(scan_indices)}"
        )

    predictions = np.concatenate(all_predictions, axis=0)
    scan_indices = np.concatenate(all_scan_indices, axis=0)

    # Explicitly restore original scan order.
    order = np.argsort(scan_indices)
    predictions = predictions[order]
    scan_indices = scan_indices[order]

    if not np.array_equal(
        scan_indices,
        np.arange(len(test_dataset))
    ):
        raise RuntimeError(
            "Scan indices are missing, duplicated, or out of order."
        )

    expected_shape = (len(test_dataset), NUM_POINTS, 3)

    if predictions.shape != expected_shape:
        raise RuntimeError(
            f"Expected predictions {expected_shape}, "
            f"got {predictions.shape}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prediction_path = OUTPUT_DIR / "test_predictions.npy"

    np.save(prediction_path, predictions)

    # Save one PLY file and four-view image per test scan.
    ply_dir = OUTPUT_DIR / "ply"
    views_dir = OUTPUT_DIR / "views"

    ply_dir.mkdir(parents=True, exist_ok=True)
    views_dir.mkdir(parents=True, exist_ok=True)

    for index, points in enumerate(predictions):
        sample_name = f"test_{index:04d}"

        export_ply(
            points,
            ply_dir / f"{sample_name}.ply"
        )

        fig = render_four_views(
            points,
            output_path=views_dir / f"{sample_name}_views.png",
            title=sample_name
        )

        import matplotlib.pyplot as plt
        plt.close(fig)

    print("\nReconstruction completed.")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Predictions: {prediction_path}")
    print(f"PLY files: {ply_dir}")
    print(f"Four-view images: {views_dir}")

    return predictions


if __name__ == "__main__":
    reconstruct_test_set()
