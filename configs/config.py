
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class Config:
    # ======================== PATHS ========================
    data_root: str = "/workspace/data/chexpert_dataset"
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    sample_dir: str = "outputs/samples"
    log_dir: str = "outputs/logs"
    eval_dir: str = "outputs/eval_results"

    # Pretrained ProjectedGAN checkpoint (from notebook/projected_gan train.py)
    pretrained_g1_path: Optional[str] = None
    projgan_dir: Optional[str] = None  # Path to cloned projected_gan repo

    # ======================== DATA ========================
    img_size: int = 256
    img_channels: int = 3  # 3ch grayscale (matches ProjectedGAN RGB output)
    num_classes: int = 14
    uncertainty_policy: str = "uones"

    chexpert_labels: List[str] = field(default_factory=lambda: [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices"
    ])

    # ======================== ProjectedGAN MODEL ========================
    z_dim: int = 256              # FastGAN latent dimension (256 for better fidelity)
    c_embed_dim: int = 128        # Class embedding dimension
    ngf: int = 128                # FastGAN generator feature multiplier

    # Projected Discriminator
    d_backbone: str = "tf_efficientnet_lite0"  # EfficientNet backbone
    d_ccm_channels: int = 64                    # CCM output channels per scale

    # G2 (Heatmap Generator) - keep U-Net based
    g2_base_ch: int = 64

    # ======================== TRAINING (kimg-based) ========================
    total_kimg: int = 1200          # Total training length in kimg
    phase1_kimg: int = 500          # Image foundation (G1 + D)
    phase2_kimg: int = 300          # Heatmap learning (G2)
    phase3_kimg: int = 400          # Joint refinement (G1 + G2)

    batch_size: int = 16
    grad_accum_steps: int = 1
    num_workers: int = 8

    lr_G: float = 2e-4
    lr_D: float = 2e-4
    lr_G_phase3: float = 5e-5
    lr_D_phase3: float = 5e-5

    # Phase 2+3 epoch-based settings (used when loading pretrained G1 pkl)
    lr_phase2: float = 2e-4
    lr_phase3: float = 1e-4
    phase2_epochs: int = 30
    phase3_epochs: int = 30

    beta1: float = 0.0
    beta2: float = 0.99

    ema_decay: float = 0.9999
    ema_start_step: int = 2000
    ema_rampup: int = 10000

    use_amp: bool = True
    grad_clip_norm: float = 1.0

    # Truncation trick (at inference/FID evaluation)
    truncation_psi: float = 0.7

    # Adaptive Discriminator Augmentation (ADA)
    use_ada: bool = True
    ada_target_rt: float = 0.6
    ada_speed: int = 500
    ada_max_p: float = 0.85

    # ======================== LOSS WEIGHTS ========================
    lambda_adv: float = 1.0
    lambda_aux: float = 1.0
    # R1 disabled for ProjectedGAN — frozen backbone features regularize D.
    # Original ProjectedGAN paper does not use R1 penalty.
    lambda_r1: float = 0.0
    r1_interval: int = 0  # 0 = disabled
    lambda_exp: float = 2.0
    lambda_comp: float = 1.0
    lambda_anat: float = 1.0
    lambda_percep: float = 0.5
    lambda_tv: float = 0.1
    lambda_sparse: float = 1.0
    lambda_div: float = 0.5
    lambda_coverage: float = 5.0
    max_heatmap_coverage: float = 0.25

    causal_warmup_kimg: int = 50       # Warmup causal loss over 50 kimg
    causal_warmup_start: float = 0.5

    heatmap_temperature: float = 0.5

    # ======================== ANATOMICAL PRIORS ========================
    anatomical_priors: Dict[int, Tuple[float, float, float]] = field(
        default_factory=lambda: {
            2:  (0.55, 0.58, 0.20),
            3:  (0.45, 0.50, 0.25),
            4:  (0.40, 0.50, 0.15),
            5:  (0.50, 0.50, 0.25),
            6:  (0.55, 0.50, 0.20),
            7:  (0.45, 0.50, 0.20),
            8:  (0.60, 0.50, 0.20),
            9:  (0.25, 0.50, 0.20),
            10: (0.70, 0.50, 0.20),
            12: (0.50, 0.50, 0.15),
        }
    )

    # ======================== DIFFAUGMENT ========================
    diffaugment_policy: str = "color,translation,cutout"

    # ======================== EVALUATION ========================
    eval_every_kimg: int = 100       # Evaluate every 100 kimg
    fid_num_samples: int = 50000
    eval_num_samples: int = 10000
    sample_every_kimg: int = 50      # Save samples every 50 kimg
    num_sample_images: int = 64

    # ======================== CHECKPOINTING ========================
    save_every_kimg: int = 50        # Save checkpoint every 50 kimg
    resume_checkpoint: Optional[str] = None

    # ======================== ABLATION ========================
    ablation_name: str = "full"

    def get_phase(self, cur_kimg: float) -> int:
        """Determine training phase from current kimg."""
        if cur_kimg <= self.phase1_kimg:
            return 1
        elif cur_kimg <= self.phase1_kimg + self.phase2_kimg:
            return 2
        else:
            return 3

    def get_lr(self, phase: int):
        if phase == 3:
            return self.lr_G_phase3, self.lr_D_phase3
        return self.lr_G, self.lr_D

    def get_causal_weight(self, cur_kimg: float, phase: int) -> float:
        """Compute causal loss weight with kimg-based warmup."""
        if phase == 1:
            return 0.0
        phase_start = self.phase1_kimg if phase == 2 else (
            self.phase1_kimg + self.phase2_kimg
        )
        kimg_in = cur_kimg - phase_start
        if kimg_in < self.causal_warmup_kimg:
            frac = kimg_in / self.causal_warmup_kimg
            return self.lambda_exp * (self.causal_warmup_start + (1 - self.causal_warmup_start) * frac)
        return self.lambda_exp

    def make_dirs(self):
        for d in [self.output_dir, self.checkpoint_dir, self.sample_dir,
                  self.log_dir, self.eval_dir]:
            os.makedirs(d, exist_ok=True)


ABLATION_CONFIGS = {
    "full": {},
    "no_causal": {"lambda_exp": 0.0, "ablation_name": "no_causal"},
    "no_anat": {"lambda_anat": 0.0, "ablation_name": "no_anat"},
    "no_percep": {"lambda_percep": 0.0, "ablation_name": "no_percep"},
    "no_div": {"lambda_div": 0.0, "ablation_name": "no_div"},
    "no_diffaugment": {"diffaugment_policy": "", "ablation_name": "no_diffaugment"},
    "no_progressive": {"phase1_kimg": 0, "phase2_kimg": 0, "phase3_kimg": 1200,
                        "ablation_name": "no_progressive"},
    "causal_nec_only": {"lambda_comp": 0.0, "ablation_name": "causal_nec_only"},
    "causal_suf_only": {"ablation_name": "causal_suf_only"},
}


def get_config(ablation: str = "full", **overrides) -> Config:
    cfg = Config()
    if ablation in ABLATION_CONFIGS:
        for k, v in ABLATION_CONFIGS[ablation].items():
            setattr(cfg, k, v)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.make_dirs()
    return cfg
