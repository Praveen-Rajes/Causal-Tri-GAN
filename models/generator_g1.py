"""
CausalTriGAN - G1: ProjectedGAN Generator Wrapper

Loads the ORIGINAL ProjectedGAN G_ema from .pkl (NVIDIA pickle format)
or falls back to a built-in FastGAN backbone for from-scratch training.

Notebook workflow (recommended):
  1. Train G1 externally: projected_gan/train.py --cond=1 --cfg=fastgan --kimg=700
  2. Load G_ema from network-snapshot.pkl → this wrapper freezes it
  3. Phase 2+3 only train G2 + label_embed on real images

The original G_ema has built-in label conditioning (c_dim=14) and its
forward(z, c, truncation_psi) API is preserved.
"""
import os
import sys
import pickle
import torch
import torch.nn as nn

from models.blocks import GLU, SEBlock, InitLayer, UpBlockComp, UpBlock


# ---------------------------------------------------------------------------
# Built-in FastGAN backbone (fallback when no pretrained pkl is provided)
# ---------------------------------------------------------------------------
class FastGANBackbone(nn.Module):
    """
    FastGAN generator backbone (from ProjectedGAN).
    Produces 3-channel 256x256 images from z-dim noise.
    Architecture matches autonomousvision/projected-gan cfg=fastgan.
    """

    def __init__(self, ngf=128, z_dim=256, nc=3, img_resolution=256):
        super().__init__()
        self.z_dim = z_dim
        self.img_resolution = img_resolution

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5,
                     512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.init = InitLayer(z_dim, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlockComp(nfc[8], nfc[16])
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        self.feat_64 = UpBlockComp(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = nn.Sequential(
            nn.Conv2d(nfc[256], nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        if z.dim() == 2:
            z = z.unsqueeze(2).unsqueeze(3)

        feat_4 = self.init(z)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)
        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))
        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))
        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

        return self.to_big(feat_256)


# ---------------------------------------------------------------------------
# G1 Wrapper — loads original ProjectedGAN G_ema or uses fallback backbone
# ---------------------------------------------------------------------------
class GeneratorG1(nn.Module):
    """
    Wrapper around the original ProjectedGAN G_ema generator.

    Two modes:
      A) **Pretrained pkl** (recommended, matches notebook):
         Loads G_ema object directly from .pkl via pickle.
         The original model already has cond=True, c_dim=14 built-in.
         forward(z, c, truncation_psi) works natively.
         Requires projected_gan/ to be on sys.path for unpickling.

      B) **From-scratch** (fallback):
         Uses built-in FastGANBackbone with a separate label_embed MLP.
         Label conditioning: z_cond = z + MLP(labels).

    In both modes the API is: forward(z, labels, truncation_psi) -> [B,3,256,256]
    """

    def __init__(self, z_dim=256, c_embed_dim=128, num_classes=14,
                 img_channels=3, ngf=128, pretrained_path=None,
                 projgan_dir=None, freeze=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = num_classes
        self.img_channels = img_channels
        self.ngf = ngf
        self._use_original = False  # True when pkl loaded

        if pretrained_path and pretrained_path.endswith('.pkl'):
            self._load_original_pkl(pretrained_path, projgan_dir)
        elif pretrained_path and pretrained_path.endswith('.pt'):
            self._build_fallback(z_dim, c_embed_dim, num_classes, ngf)
            self._load_pt_weights(pretrained_path)
        else:
            self._build_fallback(z_dim, c_embed_dim, num_classes, ngf)

        if freeze:
            self.freeze()

        self._print_info()

    # ----- Mode A: Load original ProjectedGAN G_ema from pkl -----
    def _load_original_pkl(self, pkl_path, projgan_dir=None):
        """Load G_ema directly from NVIDIA pickle (preserves original arch)."""
        # Add projected_gan repo to sys.path so pickle can find pg_modules
        if projgan_dir is None:
            # Try common locations
            candidates = [
                os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'projected_gan'),
                os.path.join(os.getcwd(), 'projected_gan'),
                '/workspace/projected_gan',
                '/workspace/causal_trigan/projected_gan',
            ]
            for d in candidates:
                if os.path.isdir(d):
                    projgan_dir = d
                    break

        if projgan_dir and projgan_dir not in sys.path:
            sys.path.insert(0, projgan_dir)
            print(f"[G1] Added to sys.path: {projgan_dir}")

        print(f"[G1] Loading original ProjectedGAN G_ema from: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if 'G_ema' in data:
            self._original_model = data['G_ema']
        elif 'G' in data:
            self._original_model = data['G']
        else:
            raise ValueError(f"No generator found in pkl: {list(data.keys())}")

        self._original_model.eval()
        self._use_original = True

        # Expose key attributes from original model
        self.z_dim = self._original_model.z_dim
        self.c_dim = getattr(self._original_model, 'c_dim', 14)
        self.img_channels = getattr(self._original_model, 'img_channels', 3)

        print(f"[G1] Original G_ema loaded: z_dim={self.z_dim}, c_dim={self.c_dim}")

    # ----- Mode B: Fallback with built-in backbone -----
    def _build_fallback(self, z_dim, c_embed_dim, num_classes, ngf):
        """Build from-scratch FastGAN backbone + label MLP."""
        self.label_embed = nn.Sequential(
            nn.Linear(num_classes, c_embed_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(c_embed_dim, z_dim),
        )
        self.backbone = FastGANBackbone(
            ngf=ngf, z_dim=z_dim, nc=3, img_resolution=256
        )

    def _load_pt_weights(self, pt_path):
        """Load state_dict from .pt file into fallback backbone."""
        print(f"[G1] Loading .pt weights from: {pt_path}")
        data = torch.load(pt_path, map_location='cpu', weights_only=False)

        if 'G_ema_state_dict' in data:
            src_state = data['G_ema_state_dict']
        elif isinstance(data, dict) and any(k.startswith('backbone.') for k in data):
            src_state = data
        else:
            src_state = data

        # Try loading into full model first
        try:
            self.load_state_dict(src_state, strict=False)
            print(f"  Loaded into full model")
            return
        except Exception:
            pass

        # Fall back to loading into backbone only
        dst_state = self.backbone.state_dict()
        loaded, skipped = 0, 0
        for key, val in src_state.items():
            clean_key = key.replace('synthesis.', '')
            if clean_key in dst_state and dst_state[clean_key].shape == val.shape:
                dst_state[clean_key] = val
                loaded += 1
            else:
                skipped += 1
        self.backbone.load_state_dict(dst_state, strict=False)
        print(f"  Loaded {loaded} backbone params, skipped {skipped}")

    def _print_info(self):
        if self._use_original:
            params = sum(p.numel() for p in self._original_model.parameters())
            print(f"[G1-Original] {params/1e6:.1f}M params (from pkl)")
        else:
            total = sum(p.numel() for p in self.parameters())
            print(f"[G1-Fallback] {total/1e6:.1f}M params (built-in backbone)")

    def freeze(self):
        """Freeze all G1 parameters (for Phase 2+3)."""
        if self._use_original:
            for p in self._original_model.parameters():
                p.requires_grad = False
            self._original_model.eval()
        else:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()
        print("[G1] Frozen (all params requires_grad=False)")

    def unfreeze(self):
        """Unfreeze G1 parameters."""
        if self._use_original:
            for p in self._original_model.parameters():
                p.requires_grad = True
        else:
            for p in self.parameters():
                p.requires_grad = True

    def forward(self, z, labels, truncation_psi=1.0, **kwargs):
        """
        Generate images.

        Args:
            z: [B, z_dim] latent noise
            labels: [B, c_dim] condition labels (14-dim CheXpert)
            truncation_psi: truncation strength (1.0=none, 0.7=recommended)
        Returns:
            [B, 3, 256, 256] in [-1, 1]
        """
        if self._use_original:
            # Original ProjectedGAN G_ema: forward(z, c, truncation_psi=...)
            return self._original_model(z, labels, truncation_psi=truncation_psi)
        else:
            # Fallback: additive z-space conditioning
            z_shift = self.label_embed(labels)
            z_cond = z + z_shift
            if truncation_psi < 1.0:
                z_mean = z_cond.mean(dim=0, keepdim=True)
                z_cond = z_mean + truncation_psi * (z_cond - z_mean)
            return self.backbone(z_cond)

    def parameters(self, recurse=True):
        """Return parameters (routes to correct model)."""
        if self._use_original:
            return self._original_model.parameters()
        else:
            return super().parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        if self._use_original:
            return self._original_model.named_parameters(prefix=prefix, recurse=recurse)
        else:
            return super().named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        if self._use_original:
            return self._original_model.state_dict(*args, **kwargs)
        else:
            return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        if self._use_original:
            return self._original_model.load_state_dict(state_dict, strict=strict)
        else:
            return super().load_state_dict(state_dict, strict=strict)

    def to(self, *args, **kwargs):
        if self._use_original:
            self._original_model = self._original_model.to(*args, **kwargs)
            return self
        else:
            return super().to(*args, **kwargs)

    def eval(self):
        if self._use_original:
            self._original_model.eval()
            return self
        else:
            return super().eval()

    def train(self, mode=True):
        if self._use_original:
            # Keep original in eval mode always (frozen)
            self._original_model.eval()
            return self
        else:
            return super().train(mode)

    @staticmethod
    def get_gray(img):
        """Extract single-channel grayscale from 3ch output."""
        return img[:, 0:1]
