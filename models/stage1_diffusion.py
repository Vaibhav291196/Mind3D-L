import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def timestep_embedding(timesteps, dim):
    half = dim // 2

    frequencies = torch.exp(
        -math.log(10000) * torch.arange(
            half,
            device=timesteps.device,
            dtype=torch.float32,
        ) / max(half - 1, 1)
    )

    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)

    embedding = torch.cat(
        [torch.sin(angles), torch.cos(angles)],
        dim=-1,
    )

    if dim % 2:
        embedding = F.pad(embedding, (0, 1))

    return embedding


class Stage1DiffusionPrior(nn.Module):

    def __init__(
        self,
        dim=512,
        depth=6,
        heads=8,
        timesteps=100,
        cond_drop_prob=0.2,
        dropout=0.01,
        gradient_checkpointing=True,
    ):
        super().__init__()

        self.dim = dim
        self.timesteps = timesteps
        self.cond_drop_prob = cond_drop_prob
        self.gradient_checkpointing = gradient_checkpointing

        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

        self.null_fmri_token = nn.Parameter(
            torch.randn(1, 1, dim) * 0.02
        )

        self.query_token = nn.Parameter(
            torch.randn(1, 1, dim) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )

        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, dim)

        # Cosine diffusion schedule.
        steps = torch.linspace(
            0, timesteps, timesteps + 1, dtype=torch.float64
        )

        offset = 0.008

        alpha_bar = torch.cos(
            ((steps / timesteps + offset) / (1 + offset))
            * math.pi / 2
        ) ** 2

        alpha_bar = alpha_bar / alpha_bar[0]

        betas = (
            1 - alpha_bar[1:] / alpha_bar[:-1]
        ).clamp(0, 0.999).float()

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alpha_cumprod)

        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alpha_cumprod),
        )

        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alpha_cumprod),
        )

    # --------------------------------------------------------
    # FORWARD DIFFUSION
    # --------------------------------------------------------

    def q_sample(self, x_start, timesteps, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start)

        a = self.sqrt_alphas_cumprod[timesteps].unsqueeze(-1)
        b = self.sqrt_one_minus_alphas_cumprod[
            timesteps
        ].unsqueeze(-1)

        return a * x_start + b * noise

    # --------------------------------------------------------
    # TRANSFORMER
    # --------------------------------------------------------

    def _transform(self, sequence, causal_mask):

        use_checkpointing = (
            self.training
            and self.gradient_checkpointing
            and torch.is_grad_enabled()
        )

        for layer in self.transformer.layers:

            if use_checkpointing:
                sequence = checkpoint(
                    lambda x, current_layer=layer:
                        current_layer(x, src_mask=causal_mask),
                    sequence,
                    use_reentrant=False,
                )
            else:
                sequence = layer(
                    sequence,
                    src_mask=causal_mask,
                )

        if self.transformer.norm is not None:
            sequence = self.transformer.norm(sequence)

        return sequence

    # --------------------------------------------------------
    # PREDICT CLEAN CLIP EMBEDDING
    # --------------------------------------------------------

    def _predict_from_condition(
        self,
        condition,
        noisy_image,
        timesteps,
    ):

        batch_size = condition.shape[0]

        time_features = self.time_mlp(
            timestep_embedding(timesteps, self.dim)
        ).unsqueeze(1)

        query = self.query_token.expand(
            batch_size, -1, -1
        )

        sequence = torch.cat(
            [
                condition,
                time_features,
                noisy_image.unsqueeze(1),
                query,
            ],
            dim=1,
        )

        length = sequence.shape[1]

        causal_mask = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=sequence.device,
            ),
            diagonal=1,
        )

        output = self._transform(sequence, causal_mask)

        return self.output_projection(
            self.output_norm(output[:, -1])
        )

    def predict_x0(
        self,
        fmri_tokens,
        noisy_image,
        timesteps,
        guidance_scale=1.0,
    ):

        if fmri_tokens.ndim != 3 or noisy_image.ndim != 2:
            raise ValueError(
                "Expected fMRI [B,N,D] and image [B,D]"
            )

        batch_size = fmri_tokens.shape[0]

        if fmri_tokens.shape[1] == 0:
            raise ValueError("fMRI token sequence cannot be empty")

        if fmri_tokens.shape[-1] != self.dim:
            raise ValueError("Invalid fMRI embedding dimension")

        if noisy_image.shape != (batch_size, self.dim):
            raise ValueError("Invalid noisy image dimensions")

        if timesteps.shape != (batch_size,):
            raise ValueError("timesteps must have shape [B]")

        timesteps = timesteps.to(
            device=fmri_tokens.device,
            dtype=torch.long,
        )

        if torch.any(
            (timesteps < 0) | (timesteps >= self.timesteps)
        ):
            raise ValueError("Timesteps outside diffusion schedule")

        conditional = self._predict_from_condition(
            fmri_tokens,
            noisy_image,
            timesteps,
        )

        if guidance_scale == 1.0:
            return conditional

        null_condition = self.null_fmri_token.expand(
            batch_size,
            fmri_tokens.shape[1],
            self.dim,
        )

        unconditional = self._predict_from_condition(
            null_condition,
            noisy_image,
            timesteps,
        )

        return unconditional + guidance_scale * (
            conditional - unconditional
        )

    # --------------------------------------------------------
    # TRAINING FORWARD PASS
    # --------------------------------------------------------

    def forward(
        self,
        fmri_tokens,
        image_embed,
        timesteps=None,
        noise=None,
    ):

        if fmri_tokens.ndim != 3 or image_embed.ndim != 2:
            raise ValueError(
                "Expected fMRI [B,N,D] and image [B,D]"
            )

        batch_size = fmri_tokens.shape[0]

        if image_embed.shape != (batch_size, self.dim):
            raise ValueError("Invalid image embedding dimensions")

        if fmri_tokens.shape[-1] != self.dim:
            raise ValueError("Invalid fMRI embedding dimension")

        device = fmri_tokens.device

        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.timesteps,
                (batch_size,),
                device=device,
            )
        else:
            timesteps = timesteps.to(
                device=device,
                dtype=torch.long,
            )

        if timesteps.shape != (batch_size,):
            raise ValueError("timesteps must have shape [B]")

        if torch.any(
            (timesteps < 0) | (timesteps >= self.timesteps)
        ):
            raise ValueError("Timesteps outside diffusion schedule")

        if noise is None:
            noise = torch.randn_like(image_embed)
        elif noise.shape != image_embed.shape:
            raise ValueError("noise must match image_embed shape")

        noisy_image = self.q_sample(
            image_embed,
            timesteps,
            noise,
        )

        condition = fmri_tokens

        if self.training and self.cond_drop_prob > 0:

            drop_mask = (
                torch.rand(batch_size, device=device)
                < self.cond_drop_prob
            )

            null_condition = self.null_fmri_token.expand(
                batch_size,
                fmri_tokens.shape[1],
                self.dim,
            )

            condition = torch.where(
                drop_mask[:, None, None],
                null_condition,
                fmri_tokens,
            )

        predicted_x0 = self._predict_from_condition(
            condition,
            noisy_image,
            timesteps,
        )

        loss = F.mse_loss(
            predicted_x0.float(),
            image_embed.float(),
        )

        return loss, predicted_x0

    # --------------------------------------------------------
    # DDIM SAMPLING
    # --------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        fmri_tokens,
        sample_steps=64,
        guidance_scale=1.0,
        normalize_output=True,
    ):
        """
        Generate CLIP-space embeddings using deterministic DDIM.

        Args:
            fmri_tokens: [B, N, D] fMRI conditioning tokens.
            sample_steps: Number of reverse diffusion steps.
            guidance_scale: Classifier-free guidance scale.
            normalize_output: L2-normalize final embeddings.

        Returns:
            Generated embeddings with shape [B, D].
        """

        if fmri_tokens.ndim != 3:
            raise ValueError(
                "fmri_tokens must have shape [B, N, D]"
            )

        batch_size, num_tokens, feature_dim = fmri_tokens.shape

        if batch_size == 0 or num_tokens == 0:
            raise ValueError("Empty batch or fMRI token sequence")

        if feature_dim != self.dim:
            raise ValueError("Invalid fMRI embedding dimension")

        if not 1 <= sample_steps <= self.timesteps:
            raise ValueError(
                "sample_steps must be between 1 and self.timesteps"
            )

        if guidance_scale < 0:
            raise ValueError("guidance_scale cannot be negative")

        device = fmri_tokens.device

        # Disable transformer dropout during sampling.
        was_training = self.training
        self.eval()

        try:
            # Start from Gaussian noise at the final training timestep.
            noisy_image = torch.randn(
                batch_size,
                self.dim,
                device=device,
                dtype=fmri_tokens.dtype,
            )

            # Descending sequence of unique diffusion timesteps.
            time_sequence = torch.linspace(
                self.timesteps - 1,
                0,
                sample_steps,
                device=device,
            ).round().long()

            for index, timestep_value in enumerate(time_sequence):

                current_t = int(timestep_value.item())

                timesteps = torch.full(
                    (batch_size,),
                    current_t,
                    device=device,
                    dtype=torch.long,
                )

                predicted_x0 = self.predict_x0(
                    fmri_tokens=fmri_tokens,
                    noisy_image=noisy_image,
                    timesteps=timesteps,
                    guidance_scale=guidance_scale,
                )

                # Last step: return the predicted clean embedding.
                if index == len(time_sequence) - 1:
                    noisy_image = predicted_x0
                    break

                next_t = int(time_sequence[index + 1].item())

                # Use float32 for numerically stable DDIM updates.
                alpha_t = self.alphas_cumprod[current_t].float()
                alpha_next = self.alphas_cumprod[next_t].float()

                x_t = noisy_image.float()
                x0 = predicted_x0.float()

                predicted_noise = (
                    x_t - torch.sqrt(alpha_t) * x0
                ) / torch.sqrt(
                    (1.0 - alpha_t).clamp(min=1e-8)
                )

                # Deterministic DDIM update (eta = 0).
                next_image = (
                    torch.sqrt(alpha_next) * x0
                    + torch.sqrt(1.0 - alpha_next)
                    * predicted_noise
                )

                noisy_image = next_image.to(
                    dtype=fmri_tokens.dtype
                )

            if normalize_output:
                noisy_image = F.normalize(
                    noisy_image.float(),
                    dim=-1,
                ).to(dtype=fmri_tokens.dtype)

            return noisy_image

        finally:
            # Restore the model's previous training/evaluation state.
            self.train(was_training)
