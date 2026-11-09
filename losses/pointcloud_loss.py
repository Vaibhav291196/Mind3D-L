import torch
import torch.nn as nn
import torch.nn.functional as F


def chamfer_loss(predicted, target):
    """
    Symmetric Chamfer distance.

    predicted: (B, N, 3)
    target:    (B, M, 3)
    """

    distances = torch.cdist(predicted, target, p=2)

    pred_to_target = distances.min(dim=2)[0]
    target_to_pred = distances.min(dim=1)[0]

    return (
        pred_to_target.mean(dim=1)
        + target_to_pred.mean(dim=1)
    ).mean()


def repulsion_loss(points, k=5, radius=0.05):
    """
    Penalize points that are excessively close to each other.

    Uses nearest neighbours while excluding each point itself.
    """

    distances = torch.cdist(points, points, p=2)

    batch_size, num_points, _ = distances.shape

    diagonal = torch.eye(
        num_points,
        device=points.device,
        dtype=torch.bool
    ).unsqueeze(0)

    distances = distances.masked_fill(
        diagonal,
        float("inf")
    )

    nearest_distances = torch.topk(
        distances,
        k=min(k, num_points - 1),
        dim=-1,
        largest=False
    ).values

    penalty = F.relu(radius - nearest_distances)

    return (penalty ** 2).mean()


class PointCloudLoss(nn.Module):
    """
    Combined Chamfer and repulsion objective.
    """

    def __init__(self, chamfer_weight=1.0, repulsion_weight=0.01):
        super().__init__()

        self.chamfer_weight = chamfer_weight
        self.repulsion_weight = repulsion_weight

    def forward(self, predicted, target):

        cd = chamfer_loss(predicted, target)

        repulsion = repulsion_loss(predicted)

        total = (
            self.chamfer_weight * cd
            + self.repulsion_weight * repulsion
        )

        return {
            "total": total,
            "chamfer": cd,
            "repulsion": repulsion
        }
