
import torch
import torch.nn as nn


class FMRIPatchEncoder(nn.Module):
    """
    Encodes each of the 10 fMRI frames into:
      - 1 CLS token
      - 256 spatial patch tokens

    Input:
        fmri: [B, 10, 256, 256]

    Outputs:
        cls_features:    [B, 10, 512]
        spatial_features:[B, 10, 256, 512]
    """

    def __init__(
        self,
        image_size=256,
        patch_size=16,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        num_frames=10,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, hidden_dim)
        )

        self.spatial_position = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, hidden_dim)
        )

        self.frame_embedding = nn.Parameter(
            torch.zeros(1, num_frames, 1, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Temporal attention across the 10 frames.
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer,
            num_layers=2,
        )

        self.temporal_norm = nn.LayerNorm(hidden_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        nn.init.trunc_normal_(self.frame_embedding, std=0.02)

    def forward(self, fmri):

        if fmri.ndim != 4:
            raise ValueError(
                f"Expected [B,F,H,W], received {tuple(fmri.shape)}"
            )

        batch_size, frames, height, width = fmri.shape

        if frames != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames, received {frames}"
            )

        x = fmri.reshape(
            batch_size * frames, 1, height, width
        )

        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(
            batch_size * frames, -1, -1
        )

        x = torch.cat([cls, x], dim=1)

        x = x + self.spatial_position

        x = x.reshape(
            batch_size, frames, self.num_patches + 1, self.hidden_dim
        )

        x = x + self.frame_embedding

        x = x.reshape(
            batch_size * frames,
            self.num_patches + 1,
            self.hidden_dim,
        )

        x = self.transformer(x)
        x = self.norm(x)

        x = x.reshape(
            batch_size,
            frames,
            self.num_patches + 1,
            self.hidden_dim,
        )

        cls_features = x[:, :, 0, :]
        spatial_features = x[:, :, 1:, :]

        # Temporal attention for CLS tokens.
        cls_features = self.temporal_transformer(cls_features)
        cls_features = self.temporal_norm(cls_features)

        # Temporal attention at each corresponding spatial location.
        spatial_features = spatial_features.permute(0, 2, 1, 3)

        spatial_features = spatial_features.reshape(
            batch_size * self.num_patches,
            frames,
            self.hidden_dim,
        )

        spatial_features = self.temporal_transformer(spatial_features)
        spatial_features = self.temporal_norm(spatial_features)

        spatial_features = spatial_features.reshape(
            batch_size,
            self.num_patches,
            frames,
            self.hidden_dim,
        ).permute(0, 2, 1, 3)

        return cls_features, spatial_features


class Stage1FeatureAggregation(nn.Module):
    """
    Reduces 256 spatial tokens to 128 tokens per frame.

    Input:
        [B, 10, 256, 512]

    Output:
        [B, 1280, 512]
    """

    def __init__(
        self,
        hidden_dim=512,
        input_tokens=256,
        output_tokens=128,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.token_reduction = nn.Linear(
            input_tokens,
            output_tokens,
        )

        self.activation = nn.GELU()

        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        batch_size, frames, tokens, channels = x.shape

        x = self.norm1(x)

        x = x.permute(0, 1, 3, 2)

        x = self.token_reduction(x)
        x = self.activation(x)

        x = x.permute(0, 1, 3, 2)

        x = x.reshape(
            batch_size,
            frames * x.shape[2],
            channels,
        )

        return self.norm2(x)


class Stage1CLIPProjection(nn.Module):
    """
    Combines the 10 frame-level CLS features into one CLIP feature.

    Input:
        [B, 10, 512]

    Output:
        [B, 512]
    """

    def __init__(
        self,
        hidden_dim=512,
        num_frames=10,
        output_dim=512,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)

        self.projection = nn.Linear(
            num_frames * hidden_dim,
            output_dim,
        )

        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, cls_features):

        batch_size, frames, channels = cls_features.shape

        x = self.norm(cls_features)

        x = x.reshape(batch_size, frames * channels)

        x = self.projection(x)

        return self.output_norm(x)
