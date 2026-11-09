import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureBridgeDiffusion(nn.Module):
    """
    Feature Bridge Diffusion Model (FBDM).

    Learns to generate CLIP visual features conditioned
    on fMRI representations.

    Training input:
        visual_features: (B, 512)
        neuro_features:  (B, 512)
        timesteps:       (B,)

    Output:
        predicted_noise: (B, 512)
    """

    def __init__(
        self,
        feature_dim=512,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        num_timesteps=100,
        dropout=0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps

        self.input_projection = nn.Linear(
            feature_dim,
            hidden_dim
        )

        self.condition_projection = nn.Linear(
            feature_dim,
            hidden_dim
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.register_buffer(
            "time_frequencies",
            torch.exp(
                torch.arange(0, hidden_dim, 2)
                * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
            )
        )

    def timestep_embedding(self, timesteps):

        timesteps = timesteps.float().unsqueeze(1)

        angles = timesteps * self.time_frequencies.unsqueeze(0)

        embedding = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=-1
        )

        return self.time_mlp(embedding)

    def forward(
        self,
        noisy_features,
        neuro_features,
        timesteps
    ):

        x = self.input_projection(noisy_features)

        condition = self.condition_projection(neuro_features)

        time_embedding = self.timestep_embedding(timesteps)

        x = x + condition + time_embedding

        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x.squeeze(1)

        predicted_noise = self.output_projection(x)

        return predicted_noise

    @staticmethod
    def diffusion_loss(predicted_noise, actual_noise):

        return F.mse_loss(
            predicted_noise,
            actual_noise
        )
