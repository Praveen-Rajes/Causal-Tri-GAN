"""
CausalTriGAN-StyleGAN2 - DiffAugment (Zhao et al. 2020)
Differentiable augmentation applied to both real and fake before D.
"""
import torch
import torch.nn.functional as F


def DiffAugment(x, policy='color,translation,cutout'):
    """
    Apply differentiable augmentations.
    Args:
        x: [B, C, H, W]
        policy: comma-separated list of augmentations
    """
    if not policy:
        return x

    for p in policy.split(','):
        p = p.strip()
        if p == 'color':
            x = rand_brightness(x)
            x = rand_saturation(x)
            x = rand_contrast(x)
        elif p == 'translation':
            x = rand_translation(x)
        elif p == 'cutout':
            x = rand_cutout(x)

    return x


def rand_brightness(x):
    """Random brightness adjustment."""
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def rand_saturation(x):
    """Random saturation (for grayscale, acts as intensity scaling)."""
    if x.size(1) == 1:
        # Grayscale: random scaling
        return x * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 0.5 + 0.75)
    x_mean = x.mean(dim=1, keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    return (x - x_mean) * factor + x_mean


def rand_contrast(x):
    """Random contrast adjustment."""
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - x_mean) * factor + x_mean


def rand_translation(x, ratio=0.125):
    """Random translation with zero padding."""
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1],
                       device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1],
                       device=x.device)

    # Create grid for translation
    grid_b, grid_y, grid_x = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device),
        indexing='ij'
    )

    grid_x = (2 * (grid_x + tx) / (x.size(3) - 1) - 1).float()
    grid_y = (2 * (grid_y + ty) / (x.size(2) - 1) - 1).float()

    grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]
    return F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)


def rand_cutout(x, ratio=0.5):
    """Random rectangular cutout."""
    cutout_h = int(x.size(2) * ratio + 0.5)
    cutout_w = int(x.size(3) * ratio + 0.5)

    offset_x = torch.randint(0, x.size(3) - cutout_w + 1, size=[x.size(0), 1, 1],
                              device=x.device)
    offset_y = torch.randint(0, x.size(2) - cutout_h + 1, size=[x.size(0), 1, 1],
                              device=x.device)

    grid_b, grid_y, grid_x = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device),
        indexing='ij'
    )

    mask_y = (grid_y >= offset_y) & (grid_y < offset_y + cutout_h)
    mask_x = (grid_x >= offset_x) & (grid_x < offset_x + cutout_w)
    mask = (~(mask_y & mask_x)).float().unsqueeze(1)

    return x * mask
