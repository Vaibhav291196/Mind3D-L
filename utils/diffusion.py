
import torch


class DiffusionSchedule:
    """
    DDPM diffusion schedule and sampling utilities.

    The schedule is shared by FBDM training and inference.
    """

    def __init__(
        self,
        num_timesteps=100,
        beta_start=1e-4,
        beta_end=0.02,
        device=None
    ):
        self.num_timesteps = num_timesteps
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            dtype=torch.float32,
            device=self.device
        )

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(
            1.0 - self.alpha_bars
        )

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas
            * (1.0 - torch.cat([
                torch.ones(1, device=self.device),
                self.alpha_bars[:-1]
            ]))
            / (1.0 - self.alpha_bars)
        )

    def _extract(self, values, timesteps, x_shape):
        """
        Extract timestep-specific values and reshape for broadcasting.
        """
        values = values.to(timesteps.device)
        result = values[timesteps]

        return result.view(
            timesteps.shape[0],
            *([1] * (len(x_shape) - 1))
        )

    def add_noise(self, clean_features, timesteps, noise=None):
        """
        Forward diffusion: q(x_t | x_0).
        """
        if noise is None:
            noise = torch.randn_like(clean_features)

        sqrt_alpha_bar = self._extract(
            self.sqrt_alpha_bars,
            timesteps,
            clean_features.shape
        )

        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alpha_bars,
            timesteps,
            clean_features.shape
        )

        noisy_features = (
            sqrt_alpha_bar * clean_features
            + sqrt_one_minus_alpha_bar * noise
        )

        return noisy_features, noise

    @torch.no_grad()
    def sample(
        self,
        model,
        neuro_features,
        feature_dim=512,
        num_samples=None
    ):
        """
        Reverse DDPM sampling conditioned on neuro_features.

        Returns generated visual features with shape (B, feature_dim).
        """
        model.eval()

        batch_size = (
            num_samples
            if num_samples is not None
            else neuro_features.shape[0]
        )

        device = neuro_features.device

        generated = torch.randn(
            batch_size,
            feature_dim,
            device=device
        )

        for timestep in reversed(range(self.num_timesteps)):

            timesteps = torch.full(
                (batch_size,),
                timestep,
                device=device,
                dtype=torch.long
            )

            predicted_noise = model(
                generated,
                neuro_features,
                timesteps
            )

            beta_t = self.betas[timestep].to(device)
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[timestep].to(device)
            sqrt_one_minus_alpha_bar_t = (
                self.sqrt_one_minus_alpha_bars[timestep].to(device)
            )

            mean = sqrt_recip_alpha_t * (
                generated
                - beta_t
                * predicted_noise
                / sqrt_one_minus_alpha_bar_t
            )

            if timestep > 0:
                variance = self.posterior_variance[timestep].to(device)
                noise = torch.randn_like(generated)
                generated = mean + torch.sqrt(variance) * noise
            else:
                generated = mean

        return generated
