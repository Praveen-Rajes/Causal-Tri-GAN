"""
CausalTriGAN-StyleGAN2 - Regularization Losses
R1 gradient penalty, Path Length regularization (StyleGAN2),
TV smoothness, L1 sparsity, mode-seeking diversity,
auxiliary classification, perceptual (LPIPS).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math


def r1_penalty(real_images, d_real_score):
    """R1 gradient penalty (Mescheder et al. 2018)."""
    grad = autograd.grad(
        outputs=d_real_score.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
    )[0]
    penalty = grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()
    return penalty


def path_length_regularization(fake_img, w_latents, pl_mean):
    """
    StyleGAN2 Path Length Regularization.
    Encourages fixed-size steps in w to result in fixed-magnitude changes
    in the generated image.

    Args:
        fake_img: [B, C, H, W] generated image
        w_latents: [B, num_ws, w_dim] intermediate latent codes
        pl_mean: running mean of path lengths (EMA)
    Returns:
        pl_penalty: path length penalty
        pl_length: mean path length (for updating EMA)
    """
    # Random noise for gradient computation
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])

    # Compute Jacobian-vector product: J^T * noise
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(),
        inputs=w_latents,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Path length: ||J^T * noise||_2 per sample
    pl_length = grad.pow(2).sum(dim=2).mean(dim=1).sqrt()  # [B]
    pl_mean_new = pl_length.mean()

    # Penalty: deviation from mean path length
    pl_penalty = (pl_length - pl_mean).pow(2).mean()

    return pl_penalty, pl_mean_new.detach()


def auxiliary_loss(aux_logits, labels):
    """Binary cross-entropy for auxiliary classifier."""
    return F.binary_cross_entropy_with_logits(aux_logits, labels)


def total_variation_loss(heatmap):
    """Total variation loss for smooth heatmaps."""
    diff_h = torch.abs(heatmap[:, :, 1:, :] - heatmap[:, :, :-1, :])
    diff_w = torch.abs(heatmap[:, :, :, 1:] - heatmap[:, :, :, :-1])
    return diff_h.mean() + diff_w.mean()


def sparsity_loss(heatmap):
    """L1 sparsity loss for heatmaps."""
    return heatmap.abs().mean()


def coverage_loss(heatmap, max_coverage=0.25):
    """Penalize heatmaps that activate more than max_coverage fraction of pixels."""
    coverage = heatmap.mean(dim=[1, 2, 3])  # [B] fraction of active pixels
    excess = F.relu(coverage - max_coverage)
    return excess.mean()


def diversity_loss(z1, z2, x1, x2):
    """Mode-seeking diversity loss."""
    dz = (z1 - z2).abs().sum(dim=1)
    dx = (x1 - x2).abs().mean(dim=(1, 2, 3))
    loss = -(dx / (dz + 1e-8)).mean()
    return loss


class PerceptualLoss(nn.Module):
    """LPIPS perceptual loss."""

    def __init__(self, device="cuda"):
        super().__init__()
        import lpips
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def forward(self, x_fake, x_real):
        # LPIPS expects 3ch — if already 3ch, pass directly
        x_fake_3 = x_fake if x_fake.shape[1] == 3 else x_fake.repeat(1, 3, 1, 1)
        x_real_3 = x_real if x_real.shape[1] == 3 else x_real.repeat(1, 3, 1, 1)
        return self.lpips_fn(x_fake_3, x_real_3).mean()
