"""
CausalTriGAN-StyleGAN2 - Adaptive Discriminator Augmentation (ADA)
Karras et al. 2020 "Training Generative Adversarial Networks with Limited Data"

Automatically adjusts augmentation strength based on discriminator overfitting.
This is THE most important technique for getting good FID with <70K samples.
"""
import torch
import torch.nn.functional as F
import numpy as np


class AdaptiveAugment:
    """
    Adaptive augmentation controller.
    Monitors D(real) sign statistics and adjusts augmentation probability.

    Target: rt ≈ 0.6 (D outputs positive for ~60% of reals)
    If rt > target: D is overfitting → increase augmentation
    If rt < target: D is underfitting → decrease augmentation
    """

    def __init__(self, target_rt=0.6, speed=500, initial_p=0.0, max_p=0.85):
        """
        Args:
            target_rt: target fraction of D(real) > 0 (0.6 is standard)
            speed: number of images to reach p=1 from p=0 at max adjustment rate
            initial_p: starting augmentation probability
            max_p: maximum augmentation probability cap
        """
        self.target_rt = target_rt
        self.speed = speed
        self.p = initial_p
        self.max_p = max_p

        # Running statistics
        self.rt_sum = 0.0
        self.rt_count = 0
        self.batch_count = 0

    def update(self, d_real_scores, batch_size):
        """
        Update augmentation probability based on D(real) statistics.

        Args:
            d_real_scores: [B] or [B,1] discriminator scores on REAL images
            batch_size: current batch size
        """
        # Compute fraction of D(real) > 0
        signs = (d_real_scores.detach().flatten() > 0).float()
        self.rt_sum += signs.sum().item()
        self.rt_count += signs.numel()
        self.batch_count += 1

        # Update p every 4 batches for stability
        if self.batch_count % 4 == 0 and self.rt_count > 0:
            rt = self.rt_sum / self.rt_count

            # Adjust p: positive when rt > target (need more aug), negative otherwise
            # speed is in kimg (thousands of images), matching NVIDIA's StyleGAN2-ADA
            adjust = np.sign(rt - self.target_rt) * (self.rt_count / (self.speed * 1e3))
            self.p = np.clip(self.p + adjust, 0.0, self.max_p)

            # Reset running stats
            self.rt_sum = 0.0
            self.rt_count = 0

    def get_p(self):
        return self.p

    def __repr__(self):
        return f"AdaptiveAugment(p={self.p:.4f}, target_rt={self.target_rt})"


def ada_augment(x, p, disable_grid=False):
    """
    Apply augmentation pipeline with probability p.
    Each augmentation is applied independently with probability p.

    Augmentations (from StyleGAN2-ADA paper):
    1. Pixel blitting: x-flip, 90° rotation, integer translation
    2. Geometric: fractional translation, rotation, anisotropic scaling
    3. Color: brightness, contrast, luma flip, hue rotation, saturation

    For grayscale medical images we use a subset:
    - Horizontal flip
    - Translation
    - Brightness / Contrast
    - Cutout
    - Additive noise

    Args:
        x: [B, C, H, W] images in [-1, 1]
        p: augmentation probability (0 to 1)
        disable_grid: if True, skip geometric transforms (useful for debugging)
    Returns:
        augmented x
    """
    if p <= 0:
        return x

    B, C, H, W = x.shape
    device = x.device

    # ===== 1. Horizontal Flip (pixel blitting) =====
    if torch.rand(1).item() < p:
        mask = (torch.rand(B, 1, 1, 1, device=device) < 0.5).float()
        x = x * (1 - mask) + x.flip(-1) * mask

    # ===== 2. Translation =====
    if torch.rand(1).item() < p and not disable_grid:
        max_shift = 0.125  # fraction of image size
        tx = (torch.rand(B, device=device) * 2 - 1) * max_shift
        ty = (torch.rand(B, device=device) * 2 - 1) * max_shift

        # Build affine grid
        theta = torch.eye(2, 3, device=device).unsqueeze(0).repeat(B, 1, 1)
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros',
                          align_corners=False)

    # ===== 3. Rotation (small angles for medical images) =====
    if torch.rand(1).item() < p * 0.5 and not disable_grid:
        max_angle = 10.0  # degrees
        angle = (torch.rand(B, device=device) * 2 - 1) * max_angle * (np.pi / 180)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros',
                          align_corners=False)

    # ===== 4. Brightness =====
    if torch.rand(1).item() < p:
        brightness = (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.4
        x = x + brightness

    # ===== 5. Contrast =====
    if torch.rand(1).item() < p:
        factor = torch.rand(B, 1, 1, 1, device=device) * 0.8 + 0.6  # [0.6, 1.4]
        mean = x.mean(dim=[2, 3], keepdim=True)
        x = (x - mean) * factor + mean

    # ===== 6. Additive Gaussian Noise =====
    if torch.rand(1).item() < p * 0.3:
        noise_std = torch.rand(1, device=device).item() * 0.1
        x = x + torch.randn_like(x) * noise_std

    # ===== 7. Cutout =====
    if torch.rand(1).item() < p * 0.5:
        cut_ratio = torch.rand(1).item() * 0.3 + 0.1  # [0.1, 0.4]
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cy = torch.randint(cut_h // 2, H - cut_h // 2, (B,), device=device)
        cx = torch.randint(cut_w // 2, W - cut_w // 2, (B,), device=device)

        mask = torch.ones(B, 1, H, W, device=device)
        for i in range(B):
            y1 = max(0, cy[i] - cut_h // 2)
            y2 = min(H, cy[i] + cut_h // 2)
            x1 = max(0, cx[i] - cut_w // 2)
            x2 = min(W, cx[i] + cut_w // 2)
            mask[i, :, y1:y2, x1:x2] = 0
        x = x * mask

    return x
