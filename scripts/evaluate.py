#!/usr/bin/env python3
"""
CausalTriGAN - Evaluation Pipeline
FID, Oracle AUC, Heatmap Metrics, Grad-CAM IoU, G3 Report Quality.


"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from data.dataset import get_dataloaders
from models.generator_g1 import GeneratorG1
from models.generator_g2 import GeneratorG2
from models.oracle import Oracle


def load_models(checkpoint_path, cfg, device, g1_pkl=None, projgan_dir=None):
    """Load G1, G2, and label_embed from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine mode: Phase 2+3 checkpoint or from-scratch
    is_phase23 = "label_embed" in ckpt

    if is_phase23:
        # Phase 2+3 mode: G1 from pkl, G2 + label_embed from checkpoint
        if g1_pkl is None:
            raise ValueError("Phase 2+3 checkpoint requires --g1_pkl")

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

        if "ema_g2" in ckpt:
            G2.load_state_dict(ckpt["ema_g2"])
        else:
            G2.load_state_dict(ckpt["G2"])
        label_embed.load_state_dict(ckpt["label_embed"])
        print("[Eval] Phase 2+3 mode: G1 from pkl, G2+label_embed from checkpoint")

    else:
        # From-scratch mode: G1 and G2 from checkpoint
        G1 = GeneratorG1(
            z_dim=cfg.z_dim, c_embed_dim=cfg.c_embed_dim,
            num_classes=cfg.num_classes, img_channels=cfg.img_channels, ngf=cfg.ngf
        ).to(device)

        G2 = GeneratorG2(
            img_channels=1, num_classes=cfg.num_classes,
            c_embed_dim=cfg.c_embed_dim, heatmap_temperature=cfg.heatmap_temperature
        ).to(device)

        if "ema_g1" in ckpt:
            G1.load_state_dict(ckpt["ema_g1"])
        else:
            G1.load_state_dict(ckpt["G1"])
        if "ema_g2" in ckpt:
            G2.load_state_dict(ckpt["ema_g2"])
        else:
            G2.load_state_dict(ckpt["G2"])

        label_embed = None
        print("[Eval] From-scratch mode: G1+G2 from checkpoint")

    G1.eval()
    G2.eval()
    if label_embed is not None:
        label_embed.eval()

    return G1, G2, label_embed


@torch.no_grad()
def generate_samples(G1, G2, label_embed, num_samples, cfg, device,
                     batch_size=64, truncation_psi=None):
    if truncation_psi is None:
        truncation_psi = cfg.truncation_psi

    images, heatmaps, labels_list = [], [], []
    remaining = num_samples

    while remaining > 0:
        bs = min(batch_size, remaining)
        z = torch.randn(bs, cfg.z_dim, device=device)
        labels = torch.zeros(bs, cfg.num_classes, device=device)
        for i in range(bs):
            n_active = torch.randint(1, 4, (1,)).item()
            active_idx = torch.randperm(cfg.num_classes)[:n_active]
            labels[i, active_idx] = 1.0

        if label_embed is not None:
            c_soft = label_embed(labels)
            fake = G1(z, c_soft, truncation_psi=truncation_psi)
            hmap = G2(fake[:, 0:1], c_soft)
        else:
            fake = G1(z, labels, truncation_psi=truncation_psi)
            hmap = G2(fake[:, 0:1], labels)

        images.append(fake.cpu())
        heatmaps.append(hmap.cpu())
        labels_list.append(labels.cpu())
        remaining -= bs

    return (torch.cat(images)[:num_samples],
            torch.cat(heatmaps)[:num_samples],
            torch.cat(labels_list)[:num_samples])


def save_images_for_fid(images, save_dir):
    from PIL import Image as PILImage
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(tqdm(images, desc="Saving images for FID")):
        arr = ((img[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(arr, mode='L').convert('RGB')
        pil_img.save(os.path.join(save_dir, f"{i:06d}.png"))


def save_real_images_for_fid(dataloader, save_dir, max_images=10000):
    from PIL import Image as PILImage
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for imgs, _ in tqdm(dataloader, desc="Saving real images"):
        for img in imgs:
            if count >= max_images:
                return
            arr = ((img[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            pil_img = PILImage.fromarray(arr, mode='L').convert('RGB')
            pil_img.save(os.path.join(save_dir, f"{count:06d}.png"))
            count += 1


def compute_fid(real_dir, fake_dir):
    from cleanfid import fid
    return fid.compute_fid(real_dir, fake_dir)


@torch.no_grad()
def compute_oracle_auc(G1, label_embed, oracle, num_samples, cfg, device,
                       batch_size=64, truncation_psi=1.0):
    all_preds, all_labels = [], []
    remaining = num_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        z = torch.randn(bs, cfg.z_dim, device=device)
        labels = torch.zeros(bs, cfg.num_classes, device=device)
        for i in range(bs):
            labels[i, i % cfg.num_classes] = 1.0

        if label_embed is not None:
            c_soft = label_embed(labels)
            fake = G1(z, c_soft, truncation_psi=truncation_psi)
        else:
            fake = G1(z, labels, truncation_psi=truncation_psi)

        preds = oracle.get_chexpert_preds(fake)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        remaining -= bs

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    aucs = {}
    for i, name in enumerate(cfg.chexpert_labels):
        if all_labels[:, i].sum() > 10:
            try:
                aucs[name] = roc_auc_score(all_labels[:, i], all_preds[:, i])
            except ValueError:
                aucs[name] = float('nan')
    return aucs, np.nanmean(list(aucs.values()))


@torch.no_grad()
def compute_heatmap_metrics(G1, G2, label_embed, oracle, num_samples, cfg, device,
                            batch_size=32, truncation_psi=1.0):
    suf_scores, nec_scores, sparsities = [], [], []
    remaining = num_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        z = torch.randn(bs, cfg.z_dim, device=device)
        labels = torch.zeros(bs, cfg.num_classes, device=device)
        for i in range(bs):
            labels[i, i % cfg.num_classes] = 1.0

        if label_embed is not None:
            c_soft = label_embed(labels)
            fake = G1(z, c_soft, truncation_psi=truncation_psi)
            hmap = G2(fake[:, 0:1], c_soft)
        else:
            fake = G1(z, labels, truncation_psi=truncation_psi)
            hmap = G2(fake[:, 0:1], labels)

        x_masked = fake * hmap
        pred_masked = oracle.get_chexpert_preds(x_masked)
        x_comp = fake * (1 - hmap)
        pred_comp = oracle.get_chexpert_preds(x_comp)

        for i in range(bs):
            active = labels[i] > 0.5
            if active.any():
                suf_scores.append(pred_masked[i][active].mean().item())
                nec_scores.append(1.0 - pred_comp[i][active].mean().item())
        sparsities.append(hmap.mean().item())
        remaining -= bs

    return {
        "sufficiency_score": float(np.mean(suf_scores)),
        "necessity_score": float(np.mean(nec_scores)),
        "mean_sparsity": float(np.mean(sparsities)),
    }


@torch.no_grad()
def evaluate_reports(G1, label_embed, num_samples, cfg, device, report_mode="impression"):
    try:
        from models.report_generator import ReportGenerator
        G3 = ReportGenerator(mode=report_mode, device=device)
    except Exception as e:
        print(f"  [G3] Could not load: {e}")
        return {"g3_available": False}

    z = torch.randn(min(num_samples, 10), cfg.z_dim, device=device)
    labels = torch.zeros(z.size(0), cfg.num_classes, device=device)
    for i in range(z.size(0)):
        labels[i, i % cfg.num_classes] = 1.0

    if label_embed is not None:
        c_soft = label_embed(labels)
        fake = G1(z, c_soft, truncation_psi=cfg.truncation_psi)
    else:
        fake = G1(z, labels, truncation_psi=cfg.truncation_psi)

    reports = G3.generate_reports(fake)
    avg_len = np.mean([len(r.split()) for r in reports]) if reports else 0
    return {
        "g3_available": True, "g3_mode": report_mode,
        "num_reports": len(reports),
        "avg_report_length_words": float(avg_len),
        "sample_reports": reports[:3],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--g1_pkl", type=str, default=None,
                        help="Path to G1 ProjectedGAN pkl (required for Phase 2+3 checkpoints)")
    parser.add_argument("--projgan_dir", type=str, default=None)
    parser.add_argument("--ablation", type=str, default="full")
    parser.add_argument("--num_fid_samples", type=int, default=10000)
    parser.add_argument("--num_eval_samples", type=int, default=5000)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--truncation_psi", type=float, default=None)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--eval_reports", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = {}
    if args.data_root:
        overrides["data_root"] = args.data_root
    cfg = get_config(ablation=args.ablation, **overrides)

    print(f"\n{'='*60}")
    print(f"  CausalTriGAN Evaluation — {cfg.ablation_name}")
    print(f"  Checkpoint: {args.checkpoint}")
    if args.g1_pkl:
        print(f"  G1 pkl: {args.g1_pkl}")
    print(f"{'='*60}\n")

    G1, G2, label_embed = load_models(
        args.checkpoint, cfg, device, g1_pkl=args.g1_pkl, projgan_dir=args.projgan_dir)
    oracle = Oracle().to(device)

    trunc_psi = args.truncation_psi or cfg.truncation_psi
    results = {"checkpoint": args.checkpoint, "truncation_psi": trunc_psi}

    # 1. FID
    if not args.skip_fid:
        print("\n[1/5] Computing FID...")
        fake_dir = os.path.join(cfg.eval_dir, "fid_fake")
        real_dir = os.path.join(cfg.eval_dir, "fid_real")
        fake_imgs, _, _ = generate_samples(
            G1, G2, label_embed, args.num_fid_samples, cfg, device, truncation_psi=trunc_psi)
        save_images_for_fid(fake_imgs, fake_dir)
        if not os.path.exists(os.path.join(real_dir, "000000.png")):
            _, val_loader = get_dataloaders(cfg)
            save_real_images_for_fid(val_loader, real_dir, args.num_fid_samples)
        fid_score = compute_fid(real_dir, fake_dir)
        results["FID"] = fid_score
        print(f"  FID: {fid_score:.2f}")

    # 2. Oracle AUC
    print("\n[2/5] Computing Oracle AUC...")
    aucs, mean_auc = compute_oracle_auc(
        G1, label_embed, oracle, args.num_eval_samples, cfg, device, truncation_psi=trunc_psi)
    results["oracle_auc_mean"] = mean_auc
    results["oracle_auc_per_class"] = aucs
    print(f"  Mean Oracle AUC: {mean_auc:.4f}")

    # 3. Heatmap Metrics
    print("\n[3/5] Computing Heatmap Metrics...")
    hmap_metrics = compute_heatmap_metrics(
        G1, G2, label_embed, oracle, args.num_eval_samples, cfg, device, truncation_psi=trunc_psi)
    results.update(hmap_metrics)
    print(f"  Sufficiency: {hmap_metrics['sufficiency_score']:.4f}")
    print(f"  Necessity:   {hmap_metrics['necessity_score']:.4f}")

    # 4. G3 Reports
    if args.eval_reports:
        print("\n[4/5] Evaluating G3 Reports...")
        report_results = evaluate_reports(G1, label_embed, 10, cfg, device)
        results["g3_evaluation"] = report_results
        if report_results.get("g3_available"):
            for i, rep in enumerate(report_results.get("sample_reports", [])[:3]):
                print(f"  Sample {i+1}: {rep[:100]}...")

    # 5. Save
    results_path = os.path.join(cfg.eval_dir, f"eval_results_{cfg.ablation_name}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")


if __name__ == "__main__":
    main()
