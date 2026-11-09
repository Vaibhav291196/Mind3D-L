
import numpy as np
from pathlib import Path


def export_ply(points, output_path):
    """
    Export a point cloud as an ASCII PLY file.

    Args:
        points: NumPy array or tensor with shape (N, 3).
        output_path: Destination .ply file path.
    """
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()

    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected point cloud shape (N, 3), got {points.shape}"
        )

    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains NaN or infinite values.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")

        np.savetxt(file, points, fmt="%.7f")

    return output_path


def export_batch_ply(point_clouds, output_dir, prefix="sample"):
    """
    Export a batch of point clouds.

    Args:
        point_clouds: Array with shape (B, N, 3).
        output_dir: Directory for generated PLY files.
        prefix: Filename prefix.
    """
    if hasattr(point_clouds, "detach"):
        point_clouds = point_clouds.detach().cpu().numpy()

    point_clouds = np.asarray(point_clouds, dtype=np.float32)

    if point_clouds.ndim != 3 or point_clouds.shape[-1] != 3:
        raise ValueError(
            f"Expected batch shape (B, N, 3), got {point_clouds.shape}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths = []

    for index, points in enumerate(point_clouds):
        output_path = output_dir / f"{prefix}_{index:04d}.ply"
        exported_paths.append(export_ply(points, output_path))

    return exported_paths
