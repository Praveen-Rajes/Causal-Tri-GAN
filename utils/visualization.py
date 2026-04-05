"""
CausalTriGAN - Visualization Utilities
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def save_image_grid(images, path, nrow=8, normalize=True):
    if normalize:
        images = (images + 1) / 2
    images = images.clamp(0, 1)
    grid = make_grid(images.cpu(), nrow=nrow, padding=2, normalize=False)
    grid = grid.permute(1, 2, 0).numpy()
    # For grayscale-as-RGB (3ch identical), show as grayscale
    if grid.shape[2] == 3:
        grid = grid[:, :, 0]
        plt.imsave(path, grid, cmap='gray')
    elif grid.shape[2] == 1:
        grid = grid.squeeze(2)
        plt.imsave(path, grid, cmap='gray')
    else:
        plt.imsave(path, grid)


def save_heatmap_overlay(images, heatmaps, path, nrow=8):
    B = images.size(0)
    overlays = []

    for i in range(min(B, nrow * nrow)):
        img = ((images[i, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        hmap = heatmaps[i, 0].cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=64)
        ax.imshow(img, cmap='gray')
        ax.imshow(hmap, cmap='jet', alpha=0.4, vmin=0, vmax=1)
        ax.axis('off')
        fig.tight_layout(pad=0)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        overlays.append(buf)
        plt.close(fig)

    n = len(overlays)
    cols = nrow
    rows = (n + cols - 1) // cols
    h, w = overlays[0].shape[:2]

    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, overlay in enumerate(overlays):
        r, c = divmod(idx, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = overlay

    plt.imsave(path, grid)


def save_samples(G1, G2, fixed_z, fixed_labels, step_id, sample_dir, device,
                 ema_g1=None, label_embed=None):
    """Save sample images and heatmaps.

    Args:
        G1: Generator (may be frozen original ProjectedGAN)
        G2: Heatmap generator (or None)
        fixed_z: [N, z_dim] fixed latent vectors
        fixed_labels: [N, num_classes] fixed labels
        step_id: epoch or kimg number for filename
        sample_dir: output directory
        device: torch device
        ema_g1: optional EMA model for G1
        label_embed: optional shared label_embed module (for Phase 2+3 mode)
    """
    was_training_g2 = G2.training if G2 is not None else False
    if G2 is not None:
        G2.eval()

    with torch.no_grad():
        z = fixed_z.to(device)
        labels = fixed_labels.to(device)

        # If label_embed is provided, transform labels before passing to G1
        if label_embed is not None:
            label_embed.eval()
            c_soft = label_embed(labels)
            fake = G1(z, c_soft, truncation_psi=0.7)
        else:
            fake = G1(z, labels, truncation_psi=0.7)

        save_image_grid(fake, os.path.join(sample_dir, f"{step_id:06d}_images.png"))

        if G2 is not None:
            fake_gray = fake[:, 0:1]
            # G2 uses c_soft (from label_embed) if available, else raw labels
            if label_embed is not None:
                heatmap = G2(fake_gray, c_soft)
            else:
                heatmap = G2(fake_gray, labels)
            save_heatmap_overlay(
                fake[:64], heatmap[:64],
                os.path.join(sample_dir, f"{step_id:06d}_heatmaps.png")
            )

            sparsity = heatmap.mean().item()
            with open(os.path.join(sample_dir, "heatmap_stats.txt"), "a") as f:
                f.write(f"{step_id}: mean_sparsity={sparsity:.4f}\n")

    if was_training_g2 and G2 is not None:
        G2.train()
    if label_embed is not None:
        label_embed.train()


def plot_training_curves(log_dict, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    if 'loss_D' in log_dict:
        axes[0, 0].plot(log_dict['loss_D'], alpha=0.7)
        axes[0, 0].set_title('D Loss')
    if 'loss_G' in log_dict:
        axes[0, 1].plot(log_dict['loss_G'], alpha=0.7)
        axes[0, 1].set_title('G Loss')
    if 'fid' in log_dict:
        axes[0, 2].plot(log_dict['fid_epochs'], log_dict['fid'], 'o-')
        axes[0, 2].set_title('FID')
    if 'sufficiency_score' in log_dict:
        axes[1, 0].plot(log_dict['sufficiency_score'], alpha=0.7, label='Sufficiency')
        if 'necessity_score' in log_dict:
            axes[1, 0].plot(log_dict['necessity_score'], alpha=0.7, label='Necessity')
        axes[1, 0].legend()
        axes[1, 0].set_title('Causal Scores')
    if 'oracle_auc' in log_dict:
        axes[1, 1].plot(log_dict['oracle_auc_epochs'], log_dict['oracle_auc'], 'o-')
        axes[1, 1].set_title('Oracle AUC')
    if 'heatmap_sparsity' in log_dict:
        axes[1, 2].plot(log_dict['heatmap_sparsity'], alpha=0.7)
        axes[1, 2].set_title('Heatmap Mean Activation')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
