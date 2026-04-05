#!/usr/bin/env python3
"""
CausalTriGAN - Inference Script
Generate CXR images + heatmaps + text reports from trained model.


"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from models.generator_g1 import GeneratorG1
from models.generator_g2 import GeneratorG2

LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]


def load_models(checkpoint_path, device, g1_pkl=None, projgan_dir=None):
    cfg = get_config()
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
    return G1, G2, label_embed, cfg


def load_report_generator(device, mode="both"):
    """Load G3 report generator. mode='both' loads findings + impression."""
    try:
        if mode == "both":
            from models.report_generator import DualReportGenerator
            return DualReportGenerator(device=device)
        else:
            from models.report_generator import ReportGenerator
            return ReportGenerator(mode=mode, device=device)
    except Exception as e:
        print(f"[WARN] Could not load G3: {e}")
        return None


def make_label_vector(pathology_names, num_classes=14):
    label = torch.zeros(1, num_classes)
    if isinstance(pathology_names, str):
        pathology_names = [p.strip() for p in pathology_names.split(",")]
    for name in pathology_names:
        for i, lbl in enumerate(LABELS):
            if name.lower() == lbl.lower():
                label[0, i] = 1.0
                break
    return label


def _wrap_text(text, max_width=50):
    """Word-wrap text to max_width characters per line."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > max_width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "\n".join(lines)


def save_triplet(image_pil, heatmap_np, report_data, label_names, save_path, index):
    """Save triplet visualization. report_data can be str or dict with findings/impression."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(np.array(image_pil), cmap='gray')
    axes[0].set_title("Generated CXR (G1)", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(np.array(image_pil), cmap='gray')
    axes[1].imshow(heatmap_np, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    sparsity = heatmap_np.mean() * 100
    axes[1].set_title(f"Causal Heatmap (G2) \u2014 {sparsity:.1f}% active", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].axis('off')
    info_text = f"Pathology: {', '.join(label_names)}\n"
    info_text += f"Heatmap Sparsity: {sparsity:.1f}%\n"
    info_text += "\u2500" * 45 + "\n"

    if isinstance(report_data, dict):
        # Dual mode: both findings and impression
        findings = report_data.get("findings", "")
        impression = report_data.get("impression", "")
        if findings:
            info_text += "\nFINDINGS:\n" + _wrap_text(findings, 45) + "\n"
        if impression:
            info_text += "\nIMPRESSION:\n" + _wrap_text(impression, 45)
        if not findings and not impression:
            info_text += "\nReport: G3 generation failed"
    elif report_data:
        info_text += "\nReport:\n" + _wrap_text(report_data, 45)
    else:
        info_text += "\nReport: G3 not loaded"

    axes[2].text(0.05, 0.95, info_text, transform=axes[2].transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[2].set_title("Radiology Report (G3)", fontsize=12, fontweight='bold')

    plt.suptitle(f"CausalTriGAN \u2014 Sample #{index+1}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def generate(G1, G2, label_embed, G3, labels, z_dim, device,
             num_samples=1, seed=None, truncation_psi=0.7):
    if seed is not None:
        torch.manual_seed(seed)

    results = []
    for i in range(num_samples):
        z = torch.randn(1, z_dim, device=device)
        lbl = labels.to(device)

        if label_embed is not None:
            c_soft = label_embed(lbl)
            fake_img = G1(z, c_soft, truncation_psi=truncation_psi)
            heatmap = G2(fake_img[:, 0:1], c_soft)
        else:
            fake_img = G1(z, lbl, truncation_psi=truncation_psi)
            heatmap = G2(fake_img[:, 0:1], lbl)

        report = ""
        if G3 is not None:
            try:
                from models.report_generator import DualReportGenerator
                if isinstance(G3, DualReportGenerator):
                    report = G3.generate_both(fake_img)  # returns dict
                else:
                    report = G3.generate_single(fake_img)  # returns str
            except Exception as e:
                report = f"[G3 error: {e}]"

        active_labels = [LABELS[j] for j in range(len(LABELS)) if lbl[0, j] > 0.5]

        arr = ((fake_img[0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        results.append({
            "image": Image.fromarray(arr, mode='L'),
            "heatmap": heatmap[0, 0].cpu().numpy(),
            "report": report,
            "labels": active_labels,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="CausalTriGAN Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--g1_pkl", type=str, default=None)
    parser.add_argument("--projgan_dir", type=str, default=None)
    parser.add_argument("--pathology", type=str, default=None)
    parser.add_argument("--all_pathologies", action="store_true")
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs/inference")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--report_mode", type=str, default="both",
                        choices=["impression", "findings", "both"])
    parser.add_argument("--no_report", action="store_true")
    parser.add_argument("--save_individual", action="store_true")
    parser.add_argument("--truncation_psi", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    G1, G2, label_embed, cfg = load_models(
        args.checkpoint, device, g1_pkl=args.g1_pkl, projgan_dir=args.projgan_dir)

    G3 = None
    if not args.no_report:
        G3 = load_report_generator(device, mode=args.report_mode)

    if args.all_pathologies:
        conditions = [(name, make_label_vector(name)) for name in LABELS]
    elif args.pathology:
        conditions = [(args.pathology, make_label_vector(args.pathology))]
    else:
        conditions = []
        for i in range(args.num):
            label = torch.zeros(1, 14)
            n_active = torch.randint(1, 4, (1,)).item()
            active_idx = torch.randperm(14)[:n_active]
            label[0, active_idx] = 1.0
            label_names = [LABELS[j] for j in active_idx]
            conditions.append((",".join(label_names), label))

    total = 0
    for cond_name, label_vec in conditions:
        print(f"\nGenerating: {cond_name}")
        results = generate(G1, G2, label_embed, G3, label_vec, cfg.z_dim, device,
                          num_samples=args.num, seed=args.seed,
                          truncation_psi=args.truncation_psi)

        for i, result in enumerate(results):
            safe_name = cond_name.replace(",", "_").replace(" ", "_")[:50]
            base = f"{safe_name}_{i+1:02d}"
            triplet_path = os.path.join(args.output_dir, f"{base}_triplet.png")
            save_triplet(result["image"], result["heatmap"],
                        result["report"], result["labels"], triplet_path, total)

            if args.save_individual:
                result["image"].save(os.path.join(args.output_dir, f"{base}_cxr.png"))
                hmap_img = Image.fromarray(
                    (result["heatmap"] * 255).clip(0, 255).astype(np.uint8), mode='L')
                hmap_img.save(os.path.join(args.output_dir, f"{base}_heatmap.png"))

            print(f"  [{i+1}/{args.num}] Saved: {triplet_path}")
            rpt = result["report"]
            if isinstance(rpt, dict):
                if rpt.get("findings"):
                    print(f"           Findings:   {rpt['findings'][:100]}...")
                if rpt.get("impression"):
                    print(f"           Impression: {rpt['impression'][:100]}...")
            elif rpt:
                print(f"           Report: {rpt[:120]}...")
            total += 1

    print(f"\nDone! {total} samples saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
