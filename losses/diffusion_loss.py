import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    Mean squared error between actual and predicted noise.
    """

    def forward(self, predicted_noise, actual_noise):

        if predicted_noise.shape != actual_noise.shape:
            raise ValueError(
                "Predicted and actual noise shapes must match."
            )

        return F.mse_loss(
            predicted_noise,
            actual_noise
        )
