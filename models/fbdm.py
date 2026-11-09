import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureBridgeDiffusion(nn.Module):
    """
    Three-token conditional diffusion model.

    Tokens:
        0: Noisy visual feature
        1: fMRI conditioning feature
        2: Diffusion timestep embedding

    Input:
        noisy_features: (B, feature_dim)
        neuro_features: (B, feature_dim)
        timesteps:      (B,)

    Output:
        predicted_noise: (B, feature_dim)
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

        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even.")

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps

        self.input_projection = nn.Linear(
            feature_dim, hidden_dim
        )

        self.condition_projection = nn.Linear(
            feature_dim, hidden_dim
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Separate learnable embeddings identify the three token roles.
        self.token_embeddings = nn.Parameter(
            torch.zeros(1, 3, hidden_dim)
        )
        nn.init.normal_(self.token_embeddings, mean=0.0, std=0.02)

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
                * (
                    -torch.log(torch.tensor(10000.0))
                    / hidden_dim
                )
            )
        )

    def timestep_embedding(self, timesteps):
        timesteps = timesteps.float().unsqueeze(1)

        angles = (
            timesteps * self.time_frequencies.unsqueeze(0)
        )

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
        noisy_token = self.input_projection(noisy_features)
        condition_token = self.condition_projection(neuro_features)
        time_token = self.timestep_embedding(timesteps)

        # (B, 3, hidden_dim)
        tokens = torch.stack(
            [noisy_token, condition_token, time_token],
            dim=1
        )

        tokens = tokens + self.token_embeddings

        encoded_tokens = self.transformer(tokens)

        # Use the noisy-visual token to predict the noise.
        predicted_noise = self.output_projection(
            encoded_tokens[:, 0, :]
        )

        return predicted_noise

    @staticmethod
    def diffusion_loss(predicted_noise, actual_noise):
        return F.mse_loss(
            predicted_noise.float(),
            actual_noise.float()
        )
