import torch
import torch.nn as nn


class LatentAdaptedDecoder(nn.Module):
    """
    Lightweight Latent Adapted Decoder (LAD).

    Conditions:
        neuro_features:  (B, 512)
        visual_features: (B, 512)

    Output:
        point_cloud: (B, 2048, 3)
    """

    def __init__(
        self,
        condition_dim=512,
        hidden_dim=256,
        num_points=2048,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()

        self.num_points = num_points
        self.hidden_dim = hidden_dim

        self.neuro_projection = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.visual_projection = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.point_embedding = nn.Parameter(
            torch.randn(1, num_points, hidden_dim) * 0.02
        )

        layers = []

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])

        self.point_generator = nn.Sequential(*layers)

        self.coordinate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)
        )

    def forward(self, neuro_features, visual_features):

        if neuro_features.shape != visual_features.shape:
            raise ValueError(
                "Neuro and visual feature shapes must match."
            )

        neuro = self.neuro_projection(neuro_features)

        visual = self.visual_projection(visual_features)

        fused = self.fusion(
            torch.cat([neuro, visual], dim=-1)
        )

        x = fused.unsqueeze(1) + self.point_embedding

        x = self.point_generator(x)

        point_cloud = self.coordinate_head(x)

        return point_cloud
