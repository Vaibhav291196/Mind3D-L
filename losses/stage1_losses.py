
import torch
import torch.nn.functional as F


def multi_positive_clip_loss(
    fmri_features: torch.Tensor,
    clip_features: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.07,
    logit_scale: torch.Tensor = None,
) -> torch.Tensor:
    """
    Symmetric multi-positive contrastive loss.

    Same object group = positive pair.
    Different object group = negative pair.
    """

    if fmri_features.ndim != 2 or clip_features.ndim != 2:
        raise ValueError("Features must have shape [B, D].")

    if fmri_features.shape != clip_features.shape:
        raise ValueError("fMRI and CLIP feature shapes must match.")

    if group_ids.ndim != 1:
        raise ValueError("group_ids must have shape [B].")

    batch_size = fmri_features.shape[0]

    if group_ids.shape[0] != batch_size:
        raise ValueError("group_ids length must match batch size.")

    if batch_size < 2:
        raise ValueError("At least two samples are required.")

    if logit_scale is None and temperature <= 0:
        raise ValueError("Temperature must be positive.")

    if torch.unique(group_ids).numel() < 2:
        raise ValueError("At least two distinct groups are required.")

    if fmri_features.device != clip_features.device:
        raise ValueError("Feature tensors must be on the same device.")

    if group_ids.device != fmri_features.device:
        raise ValueError("group_ids and features must be on the same device.")

    # Perform similarity and log-sum-exp calculations in float32,
    # even when the outer training loop uses mixed precision.
    with torch.autocast(
        device_type=fmri_features.device.type,
        enabled=False,
    ):

        fmri_features = F.normalize(
            fmri_features.float(), dim=-1
        )

        clip_features = F.normalize(
            clip_features.float(), dim=-1
        )

        similarities = fmri_features @ clip_features.T

        if logit_scale is not None:
            scale = logit_scale.float().exp().clamp(max=100.0)
            logits = similarities * scale
        else:
            logits = similarities / temperature

        positive_mask = group_ids[:, None].eq(group_ids[None, :])

        def directional_loss(scores, mask):

            log_denominator = torch.logsumexp(scores, dim=1)

            log_numerator = torch.logsumexp(
                scores.masked_fill(~mask, float("-inf")),
                dim=1,
            )

            return -(log_numerator - log_denominator).mean()

        loss_fmri_to_clip = directional_loss(
            logits,
            positive_mask,
        )

        loss_clip_to_fmri = directional_loss(
            logits.T,
            positive_mask.T,
        )

        return 0.5 * (
            loss_fmri_to_clip + loss_clip_to_fmri
        )


def diffusion_mse_loss(
    predicted_x0: torch.Tensor,
    target_clip: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between predicted clean CLIP embeddings and target embeddings.
    """

    if predicted_x0.shape != target_clip.shape:
        raise ValueError(
            f"Prediction shape {predicted_x0.shape} does not match "
            f"target shape {target_clip.shape}."
        )

    return F.mse_loss(
        predicted_x0.float(),
        target_clip.float(),
        reduction="mean",
    )


def combine_stage1_losses(
    diffusion_loss: torch.Tensor,
    clip_loss: torch.Tensor,
    diffusion_weight: float = 1.0,
    clip_weight: float = 1.0,
) -> dict:
    """
    Combined Stage 1 objective:
        L_total = L_diffusion + L_CLIP
    """

    if diffusion_weight < 0 or clip_weight < 0:
        raise ValueError("Loss weights cannot be negative.")

    total_loss = (
        diffusion_weight * diffusion_loss
        + clip_weight * clip_loss
    )

    return {
        "diffusion_loss": diffusion_loss,
        "clip_loss": clip_loss,
        "total_loss": total_loss,
    }
