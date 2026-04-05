"""
CausalTriGAN-ProjectedGAN - Projected Multi-Scale Discriminator
Uses pretrained EfficientNet features at multiple scales with:
  - Frozen backbone (pretrained feature extraction)
  - Cross-Channel Mixing (CCM) at each scale
  - Small discriminator heads at each scale
  - Auxiliary classification head for CausalTriGAN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("timm is required for ProjectedGAN discriminator. "
                      "Install with: pip install timm")

from models.blocks import CCMBlock, DiscHead


class ProjectedDiscriminator(nn.Module):
    """
    Projected multi-scale discriminator (ProjectedGAN).

    Architecture:
        img(1ch) -> repeat to 3ch -> EfficientNet (frozen, pretrained)
        -> features at 4 scales
        -> CCM (trainable) at each scale
        -> DiscHead (trainable) at each scale -> per-scale scores
        -> sum of all scale scores = final real/fake score

        + Auxiliary classifier head from pooled multi-scale features.
    """

    def __init__(self, img_channels=3, num_classes=14,
                 backbone_name='tf_efficientnet_lite0', ccm_channels=64):
        super().__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes

        # Load pretrained EfficientNet (frozen feature extractor)
        backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)

        # Freeze backbone — pretrained features only, no gradient updates
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone

        # Get feature channel counts at each scale
        # EfficientNet-lite0 features_only returns features at 5 scales
        # We use the last 4 scales
        dummy = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feats = backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        # Use last 4 scales (skip the first which is too high-res)
        self.num_scales = min(4, len(self.feat_channels))
        self.scale_indices = list(range(len(self.feat_channels) - self.num_scales,
                                        len(self.feat_channels)))
        used_channels = [self.feat_channels[i] for i in self.scale_indices]

        print(f"[D-ProjectedGAN] Backbone: {backbone_name}")
        print(f"[D-ProjectedGAN] Feature scales: {used_channels}")

        # CCM and discriminator heads at each scale
        self.ccm_layers = nn.ModuleList()
        self.disc_heads = nn.ModuleList()
        for ch in used_channels:
            self.ccm_layers.append(CCMBlock(ch, ccm_channels))
            self.disc_heads.append(DiscHead(ccm_channels))

        # Auxiliary classifier head
        # Pool features from all scales, concatenate, classify
        self.aux_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in range(self.num_scales)
        ])
        aux_in_features = ccm_channels * self.num_scales
        self.aux_head = nn.Sequential(
            nn.Linear(aux_in_features, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_classes),
        )

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_frozen = sum(p.numel() for p in self.backbone.parameters())
        print(f"[D-ProjectedGAN] Trainable: {total_trainable/1e6:.1f}M, "
              f"Frozen backbone: {total_frozen/1e6:.1f}M")

    def forward(self, x, labels=None):
        """
        Args:
            x: [B, 3, 256, 256] — 3ch input (both real and fake are 3ch)
            labels: [B, num_classes] (optional, used for projection conditioning)
        Returns:
            score: [B, 1] - sum of multi-scale real/fake scores
            aux_logits: [B, num_classes] - auxiliary classification logits
        """
        # Extract multi-scale features (backbone is frozen via requires_grad=False,
        # but we still build the computation graph so gradients flow to G1)
        all_feats = self.backbone(x)

        # Process each scale
        scores = []
        pooled_feats = []
        for i, scale_idx in enumerate(self.scale_indices):
            feat = all_feats[scale_idx]

            ccm_out = self.ccm_layers[i](feat)
            score_map = self.disc_heads[i](ccm_out)  # [B, 1, H, W]
            scores.append(score_map.mean(dim=[2, 3]))  # [B, 1]

            pooled = self.aux_pools[i](ccm_out).squeeze(-1).squeeze(-1)  # [B, ccm_ch]
            pooled_feats.append(pooled)

        # Aggregate scores
        total_score = sum(scores)  # [B, 1]

        # Auxiliary classifier
        aux_feat = torch.cat(pooled_feats, dim=1)  # [B, ccm_ch * num_scales]
        aux_logits = self.aux_head(aux_feat)  # [B, num_classes]

        return total_score, aux_logits


# Alias for backward compatibility
Discriminator = ProjectedDiscriminator
