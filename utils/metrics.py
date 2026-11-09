import torch


def chamfer_distance(predicted, target):
    """
    Symmetric Chamfer Distance.

    Args:
        predicted: (B, N, 3)
        target:    (B, M, 3)

    Returns:
        Mean symmetric Chamfer Distance.
    """

    distances = torch.cdist(predicted, target, p=2)

    pred_to_target = distances.min(dim=2)[0]
    target_to_pred = distances.min(dim=1)[0]

    cd = (
        pred_to_target.mean(dim=1)
        + target_to_pred.mean(dim=1)
    )

    return cd.mean()


def f_score(predicted, target, threshold=0.05):
    """
    Compute point-cloud F-score.

    Precision: fraction of predicted points within threshold
               of a target point.

    Recall: fraction of target points within threshold
            of a predicted point.
    """

    distances = torch.cdist(predicted, target, p=2)

    precision = (
        distances.min(dim=2)[0] < threshold
    ).float().mean(dim=1)

    recall = (
        distances.min(dim=1)[0] < threshold
    ).float().mean(dim=1)

    fscore = (
        2 * precision * recall
        / (precision + recall + 1e-8)
    )

    return fscore.mean()
