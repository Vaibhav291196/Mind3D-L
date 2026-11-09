from pathlib import Path
import torch


def save_checkpoint(
    path,
    model,
    optimizer=None,
    epoch=0,
    best_loss=None,
    extra=None
):
    """Save model and training state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_loss": best_loss,
        "extra": extra or {}
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)

    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu"
):
    """Load model and optional optimizer state."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(
        f"Checkpoint loaded: {path} "
        f"(epoch={checkpoint.get('epoch', 0)})"
    )

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_loss": checkpoint.get("best_loss"),
        "extra": checkpoint.get("extra", {})
    }
