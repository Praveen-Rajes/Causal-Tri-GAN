#!/usr/bin/env python3
"""
CausalTriGAN - Generate Samples
Bulk generation for FID computation or paper figures.
 
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from models.generator_g1 import GeneratorG1
from models.generator_g2 import GeneratorG2
from utils.visualization import save_image_grid, save_heatmap_overlay


def load_models(checkpoint_path, cfg, device, g1_pkl=None, projgan_dir=None):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    is_phase23 = "label_embed" in ckpt

    if is_phase23:
        G1 = GeneratorG1(
            z_dim=cfg.z_dim, num_classes=cfg.num_classes,
            pretrained_path=g1_pkl, projgan_dir=projgan_dir, freeze=True
        ).to(device)
        G2 = GeneratorG2(
            img_channels=1, num_classes=cfg.num_classes,
            c_embed_dim=cfg.c_embed_dim, heatmap_temperature=cfg.heatmap_temperature
        ).to(device)
        label_embed = nn.Sequential(
            nn.Linear(cfg.num_classes, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, cfg.num_classes),
        ).to(device)
        G2.load_state_dict(ckpt.get("ema_g2", ckpt["G2"]))
        label_embed.load_state_dict(ckpt["label_embed"])
    else:
        G1 = GeneratorG1(
            z_dim=cfg.z_dim, c_embed_dim=cfg.c_embed_dim,
            num_classes=cfg.num_classes, img_channels=cfg.img_channels, ngf=cfg.ngf
        ).to(device)
        G2 = GeneratorG2(
            img_channels=1, num_classes=cfg.num_classes,
            c_embed_dim=cfg.c_embed_dim, heatmap_temperature=cfg.heatmap_temperature
        ).to(device)
        G1.load_state_dict(ckpt.get("ema_g1", ckpt["G1"]))
        G2.load_state_dict(ckpt.get("ema_g2", ckpt["G2"]))
        label_embed = None

    G1.eval()
    G2.eval()
    if label_embed is not None:
        label_embed.eval()
    return G1, G2, label_embed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--g1_pkl", type=str, default=None)
    parser.add_argument("--projgan_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="outputs/generated")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_heatmaps", action="store_true")
    parser.add_argument("--save_grid", action="store_true")
    parser.add_argument("--single_label", type=int, default=None)
    parser.add_argument("--truncation_psi", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_config()

    G1, G2, label_embed = load_models(
        args.checkpoint, cfg, device, g1_pkl=args.g1_pkl, projgan_dir=args.projgan_dir)

    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    if args.save_heatmaps:
        hmap_dir = os.path.join(args.output_dir, "heatmaps")
        os.makedirs(hmap_dir, exist_ok=True)

    print(f"Generating {args.num_samples} samples...")

    count = 0
    with torch.no_grad():
        pbar = tqdm(total=args.num_samples)
        while count < args.num_samples:
            bs = min(args.batch_size, args.num_samples - count)
            z = torch.randn(bs, cfg.z_dim, device=device)

            if args.single_label is not None:
                labels = torch.zeros(bs, cfg.num_classes, device=device)
                labels[:, args.single_label] = 1.0
            else:
                labels = torch.zeros(bs, cfg.num_classes, device=device)
                for i in range(bs):
                    n = torch.randint(1, 4, (1,)).item()
                    idx = torch.randperm(cfg.num_classes)[:n]
                    labels[i, idx] = 1.0

            if label_embed is not None:
                c_soft = label_embed(labels)
                fake = G1(z, c_soft, truncation_psi=args.truncation_psi)
            else:
                fake = G1(z, labels, truncation_psi=args.truncation_psi)

            for i in range(bs):
                arr = ((fake[i, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                PILImage.fromarray(arr, mode='L').convert('RGB').save(
                    os.path.join(img_dir, f"{count:06d}.png"))

                if args.save_heatmaps:
                    if label_embed is not None:
                        hmap = G2(fake[i:i+1, 0:1], c_soft[i:i+1])
                    else:
                        hmap = G2(fake[i:i+1, 0:1], labels[i:i+1])
                    hmap_arr = (hmap[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    PILImage.fromarray(hmap_arr, mode='L').save(
                        os.path.join(hmap_dir, f"{count:06d}.png"))

                count += 1
                pbar.update(1)
        pbar.close()

    if args.save_grid:
        z = torch.randn(64, cfg.z_dim, device=device)
        labels = torch.zeros(64, cfg.num_classes, device=device)
        for i in range(64):
            labels[i, i % cfg.num_classes] = 1.0
        with torch.no_grad():
            if label_embed is not None:
                c_soft = label_embed(labels)
                fake = G1(z, c_soft, truncation_psi=args.truncation_psi)
                hmap = G2(fake[:, 0:1], c_soft)
            else:
                fake = G1(z, labels, truncation_psi=args.truncation_psi)
                hmap = G2(fake[:, 0:1], labels)
        save_image_grid(fake, os.path.join(args.output_dir, "grid_images.png"))
        save_heatmap_overlay(fake, hmap, os.path.join(args.output_dir, "grid_heatmaps.png"))

    print(f"Done! {count} images saved to {img_dir}")


if __name__ == "__main__":
    main()
