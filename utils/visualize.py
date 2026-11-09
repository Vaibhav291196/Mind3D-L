
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def render_four_views(
    points,
    output_path=None,
    title="3D Point Cloud",
    point_size=1.0
):
    """
    Render a point cloud from four viewing angles.

    Args:
        points: Array or tensor with shape (N, 3).
        output_path: Optional path to save the rendered image.
        title: Figure title.
        point_size: Scatter marker size.

    Returns:
        Matplotlib figure.
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

    fig = plt.figure(figsize=(12, 10))

    views = [
        (20, 45, "View 1"),
        (20, 135, "View 2"),
        (20, 225, "View 3"),
        (20, 315, "View 4"),
    ]

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    for index, (elevation, azimuth, view_title) in enumerate(views):
        ax = fig.add_subplot(2, 2, index + 1, projection="3d")

        ax.scatter(
            x, y, z,
            s=point_size,
            alpha=0.8
        )

        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_title(view_title)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Use equal spatial limits to preserve object proportions.
        center = np.array([
            (x.max() + x.min()) / 2,
            (y.max() + y.min()) / 2,
            (z.max() + z.min()) / 2
        ])

        radius = max(
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min()
        ) / 2

        radius = max(radius, 1e-6)

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

    return fig


def render_batch(
    point_clouds,
    output_dir,
    prefix="sample"
):
    """
    Save four-view visualizations for a batch of point clouds.
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

    paths = []

    for index, points in enumerate(point_clouds):
        output_path = output_dir / f"{prefix}_{index:04d}_views.png"

        fig = render_four_views(
            points,
            output_path=output_path,
            title=f"{prefix}_{index:04d}"
        )

        plt.close(fig)
        paths.append(output_path)

    return paths
