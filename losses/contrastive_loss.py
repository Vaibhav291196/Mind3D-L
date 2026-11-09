import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Symmetric InfoNCE loss for fMRI-CLIP alignment.

    Both inputs must contain corresponding pairs
    in the same batch order.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, neuro_features, visual_features):

        if neuro_features.shape != visual_features.shape:
            raise ValueError(
                "Neuro and visual feature shapes must match."
            )

        neuro_features = F.normalize(
            neuro_features, dim=-1
        )

        visual_features = F.normalize(
            visual_features, dim=-1
        )

        logits = (
            neuro_features @ visual_features.T
        ) / self.temperature

        labels = torch.arange(
            logits.size(0),
            device=logits.device
        )

        loss_neuro_to_visual = F.cross_entropy(
            logits, labels
        )

        loss_visual_to_neuro = F.cross_entropy(
            logits.T, labels
        )

        return (
            loss_neuro_to_visual
            + loss_visual_to_neuro
        ) / 2
