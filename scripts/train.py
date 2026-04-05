#!/usr/bin/env python3
"""
CausalTriGAN - Main Training Script
Matches the notebook (CausalTriGAN_Training.ipynb) workflow exactly.

Phase 1: Train G1 externally via projected_gan/train.py --cond=1 --cfg=fastgan
Phase 2: Train G2 + label_embed on REAL images (G1 frozen, no adversarial)
Phase 3: Train G2 + label_embed + d_aux on REAL images (G1 frozen, no adversarial)


"""
import os
import sys
import argparse
import time
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from data.dataset import get_dataloaders
from models.generator_g1 import GeneratorG1
from models.generator_g2 import GeneratorG2
from models.discriminator import ProjectedDiscriminator
from models.oracle import Oracle
from models.ema import EMA
from losses.adversarial import logistic_loss_d, logistic_loss_g
from losses.causal import causal_intervention_loss
from losses.anatomical import AnatomicalPriorLoss
from losses.regularization import (
    auxiliary_loss,
    total_variation_loss, sparsity_loss, coverage_loss,
    diversity_loss, PerceptualLoss,
)
from utils.augmentation import DiffAugment
from utils.ada import AdaptiveAugment, ada_augment
from utils.visualization import save_samples, plot_training_curves
from utils.logging import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="CausalTriGAN Training")
    parser.add_argument("--g1_pkl", type=str, default=None,
                        help="Path to pretrained ProjectedGAN .pkl (from notebook/train.py)")
    parser.add_argument("--projgan_dir", type=str, default=None,
                        help="Path to cloned projected_gan repo (for pkl loading)")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Train G1 from scratch (Phase 1+2+3) instead of loading pkl")
    parser.add_argument("--ablation", type=str, default="full",
                        help="Ablation config name")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--kimg", type=int, default=None,
                        help="Override total training kimg (for Phase 2+3)")
    parser.add_argument("--batch_size", type=int, default=None)
    # Legacy compatibility
    parser.add_argument("--pretrained_g1", type=str, default=None,
                        help="Alias for --g1_pkl")
    return parser.parse_args()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2+3 Trainer (matches notebook — G1 frozen, train on real images)
# ═══════════════════════════════════════════════════════════════════════

class Phase23Trainer:
    """
    Trains G2 + label_embed (+ d_aux in Phase 3) on REAL images.
    G1 is frozen and only used for visualization/evaluation.
    Matches CausalTriGAN_Training.ipynb Phases 2 & 3 exactly.
    """

    def __init__(self, cfg, g1_pkl=None, projgan_dir=None):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self._build_models(g1_pkl, projgan_dir)
        self._build_losses()
        self._build_data()
        self._build_misc()

        self.global_step = 0
        self.start_epoch = 1
        self.best_fid = float("inf")

    def _build_models(self, g1_pkl, projgan_dir):
        cfg = self.cfg
        print("\n[Building Models — Phase 2+3 Mode]")

        # G1: Load from pkl and FREEZE (matches notebook)
        self.G1 = GeneratorG1(
            z_dim=cfg.z_dim, num_classes=cfg.num_classes,
            pretrained_path=g1_pkl, projgan_dir=projgan_dir,
            freeze=True,
        ).to(self.device)
        print(f"  G1 (ProjectedGAN): FROZEN")

        # G2: Heatmap generator (1ch input)
        self.G2 = GeneratorG2(
            img_channels=1, num_classes=cfg.num_classes,
            c_embed_dim=cfg.c_embed_dim,
            heatmap_temperature=cfg.heatmap_temperature
        ).to(self.device)
        print(f"  G2 (U-Net): {count_params(self.G2)/1e6:.1f}M trainable")

        # Shared label embedding (matches notebook: 14→256→128→14)
        self.label_embed = nn.Sequential(
            nn.Linear(cfg.num_classes, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, cfg.num_classes),
        ).to(self.device)
        print(f"  label_embed: {count_params(self.label_embed)/1e3:.1f}K trainable")

        # D-aux head (auxiliary classification, Phase 3 only)
        # Simple classifier on G1 features
        self.d_aux = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3 * 1, 256),  # will be updated below
            nn.LeakyReLU(0.2),
            nn.Linear(256, cfg.num_classes),
        ).to(self.device)
        # Proper input: 3ch image → pool → classify
        self.d_aux = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, cfg.num_classes),
        ).to(self.device)
        print(f"  d_aux: {count_params(self.d_aux)/1e3:.1f}K trainable")

        # Oracle: Frozen TorchXRayVision
        self.oracle = Oracle().to(self.device)
        print(f"  Oracle: {sum(p.numel() for p in self.oracle.parameters())/1e6:.1f}M (frozen)")

        # EMA for G2 only (G1 is frozen)
        self.ema_g2 = EMA(self.G2, decay=cfg.ema_decay, start_step=cfg.ema_start_step)

        total_train = count_params(self.G2) + count_params(self.label_embed) + count_params(self.d_aux)
        print(f"  Total trainable (Phase 2+3): {total_train/1e6:.1f}M\n")

    def _build_losses(self):
        cfg = self.cfg
        self.anat_loss_fn = AnatomicalPriorLoss(
            cfg.anatomical_priors, cfg.img_size, self.device
        )

    def _build_data(self):
        self.train_loader, self.val_loader = get_dataloaders(self.cfg)
        print(f"[Data] Train: {len(self.train_loader)} batches, Val: {len(self.val_loader)} batches")

    def _build_misc(self):
        cfg = self.cfg
        self.logger = Logger(cfg.log_dir, run_name=cfg.ablation_name)

        # Fixed z/labels for visualization
        torch.manual_seed(42)
        self.fixed_z = torch.randn(cfg.num_sample_images, cfg.z_dim)
        self.fixed_labels = torch.zeros(cfg.num_sample_images, cfg.num_classes)
        for i in range(cfg.num_sample_images):
            self.fixed_labels[i, i % cfg.num_classes] = 1.0
        torch.manual_seed(int(time.time()))

    def _oracle_pseudo_heatmap(self, real_img, label_vec):
        """Generate pseudo-GT heatmap using proper CAM (matches notebook exactly).
        Uses Oracle's denseblock4 feature hook + classifier weights."""
        return self.oracle.compute_cam(real_img, label_vec)

    def _save_checkpoint(self, epoch, phase, is_best=False):
        state = {
            "epoch": epoch,
            "phase": phase,
            "global_step": self.global_step,
            "best_fid": self.best_fid,
            "G2": self.G2.state_dict(),
            "ema_g2": self.ema_g2.state_dict(),
            "label_embed": self.label_embed.state_dict(),
            "d_aux": self.d_aux.state_dict(),
            "config": vars(self.cfg),
        }
        path = os.path.join(self.cfg.checkpoint_dir,
                           f"phase{phase}_epoch{epoch:02d}.pt")
        torch.save(state, path)
        print(f"  [Checkpoint] Saved: {path}")

        if is_best:
            best_path = os.path.join(self.cfg.checkpoint_dir, "checkpoint_best.pt")
            torch.save(state, best_path)

    def _load_checkpoint(self, path):
        print(f"[Resume] Loading {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.G2.load_state_dict(ckpt["G2"])
        self.label_embed.load_state_dict(ckpt["label_embed"])
        if "d_aux" in ckpt:
            self.d_aux.load_state_dict(ckpt["d_aux"])
        if "ema_g2" in ckpt:
            self.ema_g2.load_state_dict(ckpt["ema_g2"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)
        print(f"[Resume] Starting from epoch {self.start_epoch}")

    def _save_vis_samples(self, epoch, phase):
        """Generate samples using G1 + G2 for visualization."""
        self.G1.eval()
        self.G2.eval()
        with torch.no_grad():
            z = self.fixed_z.to(self.device)
            labels = self.fixed_labels.to(self.device)
            c_soft = self.label_embed(labels)
            fake = self.G1(z, c_soft, truncation_psi=0.7)
            save_samples(self.G1, self.G2, self.fixed_z, self.fixed_labels,
                        epoch, self.cfg.sample_dir, self.device,
                        label_embed=self.label_embed)

    # ─── Phase 2: Heatmap Learning (matches notebook run_phase2) ───
    def run_phase2(self, n_epochs):
        """
        Train G2 + label_embed on REAL images.
        Losses: oracle pseudo-heatmap L1 + causal (suff+nec) + anatomical KL
        No adversarial loss. G1 is not used in the training loop.
        """
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  PHASE 2: Heatmap Learning ({n_epochs} epochs)")
        print(f"  Trainable: G2, label_embed")
        print(f"  Input: REAL images (not G1 output)")
        print(f"  Losses: Oracle-CAM L1 + Causal + Anatomical + TV + Sparse")
        print(f"{'='*60}\n")

        self.G2.train()
        self.label_embed.train()
        for p in self.d_aux.parameters():
            p.requires_grad_(False)

        opt = torch.optim.Adam(
            list(self.G2.parameters()) + list(self.label_embed.parameters()),
            lr=cfg.lr_phase2, betas=(0.9, 0.999)
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs * len(self.train_loader)
        )
        scaler = GradScaler("cuda", enabled=cfg.use_amp)

        for epoch in range(self.start_epoch, self.start_epoch + n_epochs):
            skipped = 0
            epoch_metrics = {"oracle": [], "suff": [], "nec": [],
                           "anat": [], "total": []}

            pbar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader),
                       desc=f"Phase2 Ep{epoch}", ncols=120)

            for step_idx, (real_imgs, label_vecs) in pbar:
                real_imgs = real_imgs.to(self.device, non_blocking=True)
                label_vecs = label_vecs.to(self.device, non_blocking=True)
                img_gray = real_imgs[:, 0:1]  # 1ch from real images

                with autocast("cuda", enabled=cfg.use_amp):
                    c_soft = self.label_embed(label_vecs)
                    heatmap = self.G2(img_gray, c_soft)
                    heatmap = torch.nan_to_num(heatmap, nan=0.5,
                                              posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                    # Oracle pseudo-GT heatmap (Grad-CAM on real images)
                    pseudo_gt = torch.stack([
                        self._oracle_pseudo_heatmap(real_imgs[i:i+1], label_vecs[i])[0]
                        for i in range(real_imgs.shape[0])
                    ], dim=0)

                    # L1 loss vs oracle pseudo-GT
                    loss_oracle = F.l1_loss(heatmap, pseudo_gt)

                    # Causal intervention loss
                    loss_causal = torch.tensor(0.0, device=self.device)
                    suff_score = nec_score = 0.0
                    if cfg.lambda_exp > 0:
                        loss_c, c_metrics = causal_intervention_loss(
                            real_imgs, heatmap, label_vecs, self.oracle,
                            lambda_comp=cfg.lambda_comp
                        )
                        loss_causal = loss_c * cfg.lambda_exp
                        suff_score = c_metrics.get("sufficiency_score", 0)
                        nec_score = c_metrics.get("necessity_score", 0)

                    # Anatomical prior
                    loss_anat = torch.tensor(0.0, device=self.device)
                    if cfg.lambda_anat > 0:
                        loss_anat = self.anat_loss_fn(heatmap, label_vecs) * cfg.lambda_anat

                    # Heatmap regularization
                    loss_tv = total_variation_loss(heatmap) * cfg.lambda_tv
                    loss_sparse = sparsity_loss(heatmap) * cfg.lambda_sparse

                    loss = loss_oracle + loss_causal + loss_anat + loss_tv + loss_sparse

                # Check for NaN
                if not torch.isfinite(loss):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    pbar.set_postfix({"skip": skipped, "reason": "nonfinite"})
                    continue

                scaler.scale(loss).backward()

                if (step_idx % cfg.grad_accum_steps == 0) or (step_idx == len(self.train_loader)):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.G2.parameters()) + list(self.label_embed.parameters()),
                        cfg.grad_clip_norm
                    )
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad(set_to_none=True)

                    # EMA update
                    self.ema_g2.update(self.G2, self.global_step)

                self.global_step += 1

                epoch_metrics["oracle"].append(loss_oracle.item())
                epoch_metrics["suff"].append(suff_score)
                epoch_metrics["nec"].append(nec_score)
                epoch_metrics["anat"].append(loss_anat.item())
                epoch_metrics["total"].append(loss.item())

                pbar.set_postfix({
                    "total": f"{loss.item():.3f}",
                    "oracle": f"{loss_oracle.item():.3f}",
                    "suff": f"{suff_score:.3f}",
                    "nec": f"{nec_score:.3f}",
                    "skip": skipped,
                })

                if self.global_step % 50 == 0:
                    self.logger.log_scalars({
                        "phase2/loss_oracle": loss_oracle.item(),
                        "phase2/loss_causal": loss_causal.item(),
                        "phase2/loss_anat": loss_anat.item(),
                        "phase2/loss_total": loss.item(),
                        "phase2/suff": suff_score,
                        "phase2/nec": nec_score,
                    }, self.global_step)

            # End of epoch
            avg = {k: sum(v)/max(len(v), 1) for k, v in epoch_metrics.items()}
            print(f"  Ep{epoch}: oracle={avg['oracle']:.4f}  suff={avg['suff']:.4f}  "
                  f"nec={avg['nec']:.4f}  anat={avg['anat']:.4f}  "
                  f"total={avg['total']:.4f}  skipped={skipped}")

            self._save_checkpoint(epoch, phase=2)
            self._save_vis_samples(epoch, phase=2)

        print("  >>> Phase 2 complete.")

    # ─── Phase 3: Joint Refinement (matches notebook run_phase3) ───
    def run_phase3(self, n_epochs):
        """
        Train G2 + label_embed + d_aux on REAL images.
        Adds d_aux auxiliary classification loss on top of Phase 2 losses.
        G1 is still frozen and not used in the training loop.
        """
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  PHASE 3: Joint Refinement ({n_epochs} epochs)")
        print(f"  Trainable: G2, label_embed, d_aux")
        print(f"  Input: REAL images")
        print(f"  Losses: Phase 2 losses + D-aux classification")
        print(f"{'='*60}\n")

        self.G2.train()
        self.label_embed.train()
        self.d_aux.train()
        for p in self.d_aux.parameters():
            p.requires_grad_(True)

        opt = torch.optim.Adam(
            list(self.G2.parameters()) + list(self.label_embed.parameters())
            + list(self.d_aux.parameters()),
            lr=cfg.lr_phase3, betas=(0.9, 0.999)
        )
        scaler = GradScaler("cuda", enabled=cfg.use_amp)

        for epoch in range(self.start_epoch, self.start_epoch + n_epochs):
            skipped = 0
            epoch_metrics = {"oracle": [], "causal": [], "anat": [],
                           "aux": [], "total": []}

            pbar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader),
                       desc=f"Phase3 Ep{epoch}", ncols=120)

            for step_idx, (real_imgs, label_vecs) in pbar:
                real_imgs = real_imgs.to(self.device, non_blocking=True)
                label_vecs = label_vecs.to(self.device, non_blocking=True)
                img_gray = real_imgs[:, 0:1]

                with autocast("cuda", enabled=cfg.use_amp):
                    c_soft = self.label_embed(label_vecs)
                    heatmap = self.G2(img_gray, c_soft)
                    heatmap = torch.nan_to_num(heatmap, nan=0.5,
                                              posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                    # Oracle pseudo-GT
                    pseudo_gt = torch.stack([
                        self._oracle_pseudo_heatmap(real_imgs[i:i+1], label_vecs[i])[0]
                        for i in range(real_imgs.shape[0])
                    ], dim=0)

                    loss_oracle = F.l1_loss(heatmap, pseudo_gt)

                    # Causal intervention loss
                    loss_causal = torch.tensor(0.0, device=self.device)
                    if cfg.lambda_exp > 0:
                        loss_c, _ = causal_intervention_loss(
                            real_imgs, heatmap, label_vecs, self.oracle,
                            lambda_comp=cfg.lambda_comp
                        )
                        loss_causal = loss_c * cfg.lambda_exp

                    # Anatomical prior
                    loss_anat = torch.tensor(0.0, device=self.device)
                    if cfg.lambda_anat > 0:
                        loss_anat = self.anat_loss_fn(heatmap, label_vecs) * cfg.lambda_anat

                    # Heatmap regularization
                    loss_tv = total_variation_loss(heatmap) * cfg.lambda_tv
                    loss_sparse = sparsity_loss(heatmap) * cfg.lambda_sparse

                    # D-aux: auxiliary classification on real images
                    aux_logits = self.d_aux(real_imgs)
                    loss_aux = F.binary_cross_entropy_with_logits(
                        aux_logits, label_vecs
                    ) * cfg.lambda_aux

                    loss = (loss_oracle + loss_causal + loss_anat
                            + loss_tv + loss_sparse + loss_aux)

                if not torch.isfinite(loss):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()

                if (step_idx % cfg.grad_accum_steps == 0) or (step_idx == len(self.train_loader)):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.G2.parameters()) + list(self.label_embed.parameters())
                        + list(self.d_aux.parameters()),
                        cfg.grad_clip_norm
                    )
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                    self.ema_g2.update(self.G2, self.global_step)

                self.global_step += 1

                epoch_metrics["oracle"].append(loss_oracle.item())
                epoch_metrics["causal"].append(loss_causal.item())
                epoch_metrics["anat"].append(loss_anat.item())
                epoch_metrics["aux"].append(loss_aux.item())
                epoch_metrics["total"].append(loss.item())

                pbar.set_postfix({
                    "total": f"{loss.item():.3f}",
                    "aux": f"{loss_aux.item():.3f}",
                    "skip": skipped,
                })

                if self.global_step % 50 == 0:
                    self.logger.log_scalars({
                        "phase3/loss_oracle": loss_oracle.item(),
                        "phase3/loss_causal": loss_causal.item(),
                        "phase3/loss_anat": loss_anat.item(),
                        "phase3/loss_aux": loss_aux.item(),
                        "phase3/loss_total": loss.item(),
                    }, self.global_step)

            avg = {k: sum(v)/max(len(v), 1) for k, v in epoch_metrics.items()}
            print(f"  Ep{epoch}: oracle={avg['oracle']:.4f}  causal={avg['causal']:.4f}  "
                  f"anat={avg['anat']:.4f}  aux={avg['aux']:.4f}  "
                  f"total={avg['total']:.4f}  skipped={skipped}")

            self._save_checkpoint(epoch, phase=3)
            self._save_vis_samples(epoch, phase=3)

        print("  >>> Phase 3 complete.")

    def train(self, resume_path=None):
        """Run Phase 2 then Phase 3."""
        cfg = self.cfg

        if resume_path:
            self._load_checkpoint(resume_path)

        # Auto-detect resume from existing checkpoints
        if resume_path is None:
            p3_ckpts = sorted(glob.glob(os.path.join(cfg.checkpoint_dir, "phase3_epoch*.pt")))
            p2_ckpts = sorted(glob.glob(os.path.join(cfg.checkpoint_dir, "phase2_epoch*.pt")))
            if p3_ckpts:
                self._load_checkpoint(p3_ckpts[-1])
                remaining_p3 = cfg.phase3_epochs - (self.start_epoch - 1)
                if remaining_p3 > 0:
                    self.run_phase3(remaining_p3)
                self.logger.close()
                return
            elif p2_ckpts:
                self._load_checkpoint(p2_ckpts[-1])
                remaining_p2 = cfg.phase2_epochs - (self.start_epoch - 1)
                if remaining_p2 > 0:
                    self.run_phase2(remaining_p2)
                self.start_epoch = 1
                self.run_phase3(cfg.phase3_epochs)
                self.logger.close()
                return

        # Fresh start
        print(f"\n{'='*60}")
        print(f"  CausalTriGAN Phase 2+3 Training — {cfg.ablation_name}")
        print(f"  Phase 2: {cfg.phase2_epochs} epochs | Phase 3: {cfg.phase3_epochs} epochs")
        print(f"  BS: {cfg.batch_size} | Accum: {cfg.grad_accum_steps}")
        print(f"{'='*60}\n")

        self.run_phase2(cfg.phase2_epochs)
        self.start_epoch = 1  # Reset for Phase 3
        self.run_phase3(cfg.phase3_epochs)

        self.logger.close()
        print(f"\n{'='*60}")
        print(f"  Training Complete! Total time: {self.logger.elapsed()}")
        print(f"  Checkpoints: {cfg.checkpoint_dir}")
        print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════
#  From-Scratch Trainer (Phase 1+2+3, uses built-in backbone)
# ═══════════════════════════════════════════════════════════════════════

class FromScratchTrainer:
    """
    Full 3-phase training with built-in FastGAN backbone.
    Phase 1: Adversarial training of G1+D (kimg-based)
    Phase 2+3: Heatmap learning (switches to Phase23Trainer logic)

    NOTE: For best results, use the notebook approach instead:
      1. Train G1 via projected_gan/train.py --cond=1 --cfg=fastgan
      2. Then use Phase23Trainer with --g1_pkl
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self._build_models()
        self._build_losses()
        self._build_optimizers()
        self._build_data()
        self._build_misc()

        self.global_step = 0
        self.start_kimg = 0.0
        self.best_fid = float("inf")

        if cfg.use_ada:
            self.ada = AdaptiveAugment(
                target_rt=cfg.ada_target_rt, speed=cfg.ada_speed, max_p=cfg.ada_max_p
            )
        else:
            self.ada = None

        if cfg.resume_checkpoint:
            self._load_checkpoint(cfg.resume_checkpoint)

    def _build_models(self):
        cfg = self.cfg
        print("\n[Building Models — From-Scratch Mode]")

        self.G1 = GeneratorG1(
            z_dim=cfg.z_dim, c_embed_dim=cfg.c_embed_dim,
            num_classes=cfg.num_classes, img_channels=cfg.img_channels, ngf=cfg.ngf
        ).to(self.device)
        print(f"  G1 (FastGAN): {count_params(self.G1)/1e6:.1f}M")

        self.G2 = GeneratorG2(
            img_channels=1, num_classes=cfg.num_classes,
            c_embed_dim=cfg.c_embed_dim, heatmap_temperature=cfg.heatmap_temperature
        ).to(self.device)
        print(f"  G2 (U-Net): {count_params(self.G2)/1e6:.1f}M")

        self.D = ProjectedDiscriminator(
            img_channels=cfg.img_channels, num_classes=cfg.num_classes,
            backbone_name=cfg.d_backbone, ccm_channels=cfg.d_ccm_channels
        ).to(self.device)
        print(f"  D (Projected): {count_params(self.D)/1e6:.1f}M trainable")

        self.oracle = Oracle().to(self.device)

        self.ema_g1 = EMA(self.G1, decay=cfg.ema_decay, start_step=cfg.ema_start_step)
        self.ema_g2 = EMA(self.G2, decay=cfg.ema_decay, start_step=cfg.ema_start_step)

    def _build_losses(self):
        cfg = self.cfg
        self.anat_loss_fn = AnatomicalPriorLoss(
            cfg.anatomical_priors, cfg.img_size, self.device
        )
        self.percep_loss_fn = PerceptualLoss(device=self.device)

    def _build_optimizers(self):
        cfg = self.cfg
        self.opt_G = torch.optim.Adam([
            {"params": self.G1.parameters(), "lr": cfg.lr_G},
            {"params": self.G2.parameters(), "lr": cfg.lr_G},
        ], betas=(cfg.beta1, cfg.beta2))

        d_trainable = [p for p in self.D.parameters() if p.requires_grad]
        self.opt_D = torch.optim.Adam(
            d_trainable, lr=cfg.lr_D, betas=(cfg.beta1, cfg.beta2)
        )

        self.scaler_G = GradScaler("cuda", enabled=cfg.use_amp)
        self.scaler_D = GradScaler("cuda", enabled=cfg.use_amp)

    def _build_data(self):
        self.train_loader, self.val_loader = get_dataloaders(self.cfg)

    def _build_misc(self):
        cfg = self.cfg
        self.logger = Logger(cfg.log_dir, run_name=cfg.ablation_name)
        torch.manual_seed(42)
        self.fixed_z = torch.randn(cfg.num_sample_images, cfg.z_dim)
        self.fixed_labels = torch.zeros(cfg.num_sample_images, cfg.num_classes)
        for i in range(cfg.num_sample_images):
            self.fixed_labels[i, i % cfg.num_classes] = 1.0
        torch.manual_seed(int(time.time()))

    def _configure_phase(self, phase):
        cfg = self.cfg
        lr_G, lr_D = cfg.get_lr(phase)
        if phase == 1:
            for p in self.G1.parameters():
                p.requires_grad = True
            for p in self.G2.parameters():
                p.requires_grad = False
        elif phase == 2:
            for p in self.G1.parameters():
                p.requires_grad = False
            for p in self.G2.parameters():
                p.requires_grad = True
        elif phase == 3:
            for p in self.G1.parameters():
                p.requires_grad = True
            for p in self.G2.parameters():
                p.requires_grad = True
        for pg in self.opt_G.param_groups:
            pg["lr"] = lr_G
        for pg in self.opt_D.param_groups:
            pg["lr"] = lr_D
        print(f"\n  Phase {phase} | lr_G={lr_G}, lr_D={lr_D}")

    def _augment_for_D(self, images):
        cfg = self.cfg
        if self.ada is not None:
            return ada_augment(images, self.ada.get_p())
        else:
            return DiffAugment(images, cfg.diffaugment_policy)

    def _train_discriminator(self, real_imgs, real_labels, z, fake_labels, phase):
        cfg = self.cfg
        self.opt_D.zero_grad()
        with autocast("cuda", enabled=cfg.use_amp):
            with torch.no_grad():
                fake_imgs = self.G1(z, fake_labels)
            real_aug = self._augment_for_D(real_imgs)
            fake_aug = self._augment_for_D(fake_imgs)
            d_real, aux_real = self.D(real_aug, real_labels)
            d_fake, aux_fake = self.D(fake_aug, fake_labels)
            loss_adv = logistic_loss_d(d_real, d_fake)
            loss_aux = auxiliary_loss(aux_real, real_labels)
            loss_D = loss_adv + cfg.lambda_aux * loss_aux
        if self.ada is not None:
            self.ada.update(d_real.detach(), real_imgs.size(0))
        self.scaler_D.scale(loss_D).backward()
        self.scaler_D.unscale_(self.opt_D)
        d_trainable = [p for p in self.D.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(d_trainable, cfg.grad_clip_norm)
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()
        return {"loss_D": loss_D.item(), "D_real": d_real.mean().item(),
                "D_fake": d_fake.mean().item()}

    def _train_generator(self, real_imgs, real_labels, z, fake_labels, phase, cur_kimg):
        cfg = self.cfg
        self.opt_G.zero_grad()
        loss_G = torch.tensor(0.0, device=self.device)
        metrics = {}
        with autocast("cuda", enabled=cfg.use_amp):
            fake_imgs = self.G1(z, fake_labels)
            fake_aug = self._augment_for_D(fake_imgs)
            d_fake, aux_fake = self.D(fake_aug, fake_labels)
            loss_adv = logistic_loss_g(d_fake) * cfg.lambda_adv
            loss_aux = auxiliary_loss(aux_fake, fake_labels) * cfg.lambda_aux
            loss_G = loss_G + loss_adv + loss_aux
            metrics["loss_G_adv"] = loss_adv.item()

            if phase >= 2:
                fake_gray = fake_imgs[:, 0:1]
                heatmap = self.G2(fake_gray.detach() if phase == 2 else fake_gray, fake_labels)
                causal_weight = cfg.get_causal_weight(cur_kimg, phase)
                if causal_weight > 0:
                    loss_c, c_m = causal_intervention_loss(
                        fake_imgs, heatmap, fake_labels, self.oracle,
                        lambda_comp=cfg.lambda_comp)
                    loss_G = loss_G + causal_weight * loss_c
                    metrics.update(c_m)
                loss_tv = total_variation_loss(heatmap) * cfg.lambda_tv
                loss_sparse = sparsity_loss(heatmap) * cfg.lambda_sparse
                loss_cover = coverage_loss(heatmap, max_coverage=cfg.max_heatmap_coverage) * cfg.lambda_coverage
                loss_G = loss_G + loss_tv + loss_sparse + loss_cover
                if cfg.lambda_anat > 0:
                    loss_anat = self.anat_loss_fn(heatmap, fake_labels) * cfg.lambda_anat
                    loss_G = loss_G + loss_anat

            if phase >= 3 and cfg.lambda_div > 0:
                z2 = torch.randn_like(z)
                fake2 = self.G1(z2, fake_labels)
                loss_div = diversity_loss(z, z2, fake_imgs, fake2) * cfg.lambda_div
                loss_G = loss_G + loss_div

        metrics["loss_G"] = loss_G.item()
        self.scaler_G.scale(loss_G).backward()
        self.scaler_G.unscale_(self.opt_G)
        torch.nn.utils.clip_grad_norm_(
            list(self.G1.parameters()) + list(self.G2.parameters()), cfg.grad_clip_norm)
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()
        return metrics

    def _save_checkpoint(self, cur_kimg, is_best=False):
        state = {
            "cur_kimg": cur_kimg, "global_step": self.global_step,
            "best_fid": self.best_fid,
            "G1": self.G1.state_dict(), "G2": self.G2.state_dict(),
            "D": self.D.state_dict(),
            "ema_g1": self.ema_g1.state_dict(), "ema_g2": self.ema_g2.state_dict(),
            "opt_G": self.opt_G.state_dict(), "opt_D": self.opt_D.state_dict(),
            "scaler_G": self.scaler_G.state_dict(), "scaler_D": self.scaler_D.state_dict(),
            "config": vars(self.cfg),
        }
        if self.ada:
            state["ada_p"] = self.ada.get_p()
        path = os.path.join(self.cfg.checkpoint_dir, f"checkpoint_{cur_kimg:.0f}kimg.pt")
        torch.save(state, path)
        if is_best:
            torch.save(state, os.path.join(self.cfg.checkpoint_dir, "checkpoint_best_fid.pt"))

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.G1.load_state_dict(ckpt["G1"])
        self.G2.load_state_dict(ckpt["G2"])
        self.D.load_state_dict(ckpt["D"])
        self.ema_g1.load_state_dict(ckpt["ema_g1"])
        self.ema_g2.load_state_dict(ckpt["ema_g2"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        self.scaler_G.load_state_dict(ckpt["scaler_G"])
        self.scaler_D.load_state_dict(ckpt["scaler_D"])
        self.global_step = ckpt["global_step"]
        self.start_kimg = ckpt.get("cur_kimg", 0.0)
        self.best_fid = ckpt.get("best_fid", float("inf"))
        if "ada_p" in ckpt and self.ada:
            self.ada.p = ckpt["ada_p"]
        print(f"[Resume] {self.start_kimg:.1f} kimg, step {self.global_step}")

    def train(self):
        cfg = self.cfg
        current_phase = 0
        p1_end = cfg.phase1_kimg
        p2_end = cfg.phase1_kimg + cfg.phase2_kimg
        cur_kimg = self.start_kimg
        last_save_kimg = int(cur_kimg / cfg.save_every_kimg) * cfg.save_every_kimg
        last_sample_kimg = int(cur_kimg / cfg.sample_every_kimg) * cfg.sample_every_kimg
        data_iter = iter(self.train_loader)

        print(f"\n  From-Scratch Training: {cfg.total_kimg} kimg")
        print(f"  P1(0-{p1_end}) P2({p1_end}-{p2_end}) P3({p2_end}-{cfg.total_kimg})")

        while cur_kimg < cfg.total_kimg:
            phase = cfg.get_phase(cur_kimg)
            if phase != current_phase:
                self._configure_phase(phase)
                current_phase = phase

            self.G1.train()
            self.G2.train()
            self.D.train()

            tick_end = min(cur_kimg + cfg.sample_every_kimg, cfg.total_kimg)
            steps = int((tick_end - cur_kimg) * 1000 / cfg.batch_size)
            pbar = tqdm(range(steps), desc=f"{cur_kimg:.0f}/{cfg.total_kimg}kimg [P{phase}]",
                       ncols=120)

            for _ in pbar:
                try:
                    real_imgs, real_labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    real_imgs, real_labels = next(data_iter)

                real_imgs = real_imgs.to(self.device)
                real_labels = real_labels.to(self.device)
                z = torch.randn(real_imgs.size(0), cfg.z_dim, device=self.device)

                d_m = self._train_discriminator(real_imgs, real_labels, z, real_labels, phase)
                z = torch.randn(real_imgs.size(0), cfg.z_dim, device=self.device)
                g_m = self._train_generator(real_imgs, real_labels, z, real_labels, phase, cur_kimg)

                self.ema_g1.update(self.G1, self.global_step)
                self.ema_g2.update(self.G2, self.global_step)
                self.global_step += 1
                cur_kimg = (self.global_step * cfg.batch_size) / 1000.0 + self.start_kimg

                pbar.set_postfix({"kimg": f"{cur_kimg:.1f}",
                                  "D": f"{d_m['loss_D']:.3f}",
                                  "G": f"{g_m.get('loss_G', 0):.3f}"})

                if cur_kimg >= cfg.total_kimg:
                    break

            save_mark = int(cur_kimg / cfg.save_every_kimg) * cfg.save_every_kimg
            if save_mark > last_save_kimg:
                self._save_checkpoint(cur_kimg)
                last_save_kimg = save_mark

            sample_mark = int(cur_kimg / cfg.sample_every_kimg) * cfg.sample_every_kimg
            if sample_mark > last_sample_kimg:
                g2_vis = self.G2 if phase >= 2 else None
                save_samples(self.G1, g2_vis, self.fixed_z, self.fixed_labels,
                            int(cur_kimg), cfg.sample_dir, self.device)
                last_sample_kimg = sample_mark

        self._save_checkpoint(cur_kimg)
        self.logger.close()
        print(f"\n  From-Scratch Training Complete: {cur_kimg:.0f} kimg")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Handle --pretrained_g1 alias
    g1_pkl = args.g1_pkl or args.pretrained_g1

    overrides = {}
    if args.data_root:
        overrides["data_root"] = args.data_root
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.resume:
        overrides["resume_checkpoint"] = args.resume

    cfg = get_config(ablation=args.ablation, **overrides)

    if args.from_scratch or g1_pkl is None:
        if g1_pkl is None and not args.from_scratch:
            print("[WARNING] No --g1_pkl provided. Use --from_scratch for Phase 1+2+3,")
            print("         or provide --g1_pkl /path/to/network-snapshot.pkl")
            print("         (recommended: train G1 via projected_gan/train.py first)")
            print()
            # Default to from-scratch if no pkl
            args.from_scratch = True

        if args.from_scratch:
            print("[Mode] From-Scratch (Phase 1+2+3 with built-in backbone)")
            if args.kimg:
                overrides["total_kimg"] = args.kimg
                cfg = get_config(ablation=args.ablation, **overrides)
            trainer = FromScratchTrainer(cfg)
            trainer.train()
    else:
        print(f"[Mode] Phase 2+3 with pretrained G1")
        print(f"[G1 pkl] {g1_pkl}")
        trainer = Phase23Trainer(cfg, g1_pkl=g1_pkl, projgan_dir=args.projgan_dir)
        trainer.train(resume_path=args.resume)


if __name__ == "__main__":
    main()
