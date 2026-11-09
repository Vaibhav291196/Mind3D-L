import torch
import torch.nn as nn


class NeuroFusionEncoder(nn.Module):
    """
    Neuro-Fusion Encoder (NFE).

    Input:
        fMRI: (B, 10, 31141)

    Output:
        neuro_features: (B, 512)
        alignment_features: (B, 512)
    """

    def __init__(
        self,
        input_dim=31141,
        hidden_dim=512,
        output_dim=512,
        num_temporal_layers=2,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.temporal_embedding = nn.Parameter(
            torch.randn(1, 10, hidden_dim) * 0.02
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

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_temporal_layers
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.alignment_projection = nn.Linear(
            output_dim,
            output_dim
        )

    def forward(self, fmri):

        if fmri.ndim != 3:
            raise ValueError(
                "Expected fMRI input with shape (B, T, F)"
            )

        x = self.frame_encoder(fmri)

        x = x + self.temporal_embedding[:, :x.size(1), :]

        x = self.temporal_encoder(x)

        x = x.mean(dim=1)

        neuro_features = self.output_projection(x)

        alignment_features = self.alignment_projection(
            neuro_features
        )

        return neuro_features, alignment_features
