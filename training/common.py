
import torch
from torch.utils.data import DataLoader


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def create_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0)
    )


def create_grad_scaler(device):
    """
    Enable gradient scaling only when CUDA is available.
    """
    return torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )


def create_autocast(device):
    """
    Return an autocast context for mixed-precision training.
    """
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(device.type == "cuda")
    )


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def calculate_average_loss(total_loss, batch_count):
    if batch_count == 0:
        return 0.0

    return total_loss / batch_count
