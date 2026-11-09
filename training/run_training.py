
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from configs.config import (
    CLIP_FEATURE_FILE,
    NORMALIZATION_PATH,
    CHECKPOINT_DIR,
)


def run_training():
    print("=" * 65)
    print("MIND3D-L TRAINING PIPELINE")
    print("=" * 65)

    required_files = {
        "CLIP features": CLIP_FEATURE_FILE,
        "fMRI normalization": NORMALIZATION_PATH,
    }

    for name, path in required_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{name} file is missing: {path}"
            )

    stages = [
        ("Stage 1: Neuro-Fusion Encoder", "training/train_nfe.py"),
        ("Stage 2: Feature Bridge Diffusion Model", "training/train_fbdm.py"),
        ("Stage 3: Latent Adapted Decoder", "training/train_lad.py"),
    ]

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for stage_name, script_relative_path in stages:
        script_path = PROJECT_ROOT / script_relative_path

        if not script_path.exists():
            raise FileNotFoundError(
                f"Training script not found: {script_path}"
            )

        print("\n" + "=" * 65)
        print(stage_name)
        print("=" * 65)

        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            check=True
        )

        print(f"\nCompleted: {stage_name}")

    print("\n" + "=" * 65)
    print("ALL TRAINING STAGES COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    run_training()
