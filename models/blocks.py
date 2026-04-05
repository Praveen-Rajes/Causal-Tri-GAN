"""
CausalTriGAN-ProjectedGAN - Core Building Blocks
Implements:
  - FastGAN generator blocks (GLU, SLE, UpBlock, UpBlockComp)
  - Projected discriminator blocks (CCM, DiscHead)
  - G2 U-Net blocks (FiLM, AttentionGate) — unchanged
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# FastGAN Generator Blocks
# ==============================================================================

class GLU(nn.Module):
    """Gated Linear Unit — splits channels in half, applies sigmoid gate."""

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0
        return x[:, :nc // 2] * torch.sigmoid(x[:, nc // 2:])


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for Skip-Layer Excitation (SLE)."""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    """FastGAN initial layer: z → 4x4 feature map."""

    def __init__(self, nz, channel):
        super().__init__()
        self.init = nn.Sequential(
            nn.ConvTranspose2d(nz, channel * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(channel * 2),
            GLU(),
        )

    def forward(self, noise):
        return self.init(noise)


class UpBlockComp(nn.Module):
    """FastGAN upsample block with residual path and two conv layers."""

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes * 2),
            GLU(),
            nn.Conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes * 2),
            GLU(),
        )
        self.direct = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_planes, out_planes * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes * 2),
            GLU(),
        )

    def forward(self, x):
        return (self.main(x) + self.direct(x)) / 2


class UpBlock(nn.Module):
    """FastGAN simple upsample block with single conv layer."""

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes * 2),
            GLU(),
        )

    def forward(self, x):
        return self.main(x)


# ==============================================================================
# Projected Discriminator Blocks
# ==============================================================================

class CCMBlock(nn.Module):
    """Cross-Channel Mixing: 1x1 conv to mix projected features."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class DiscHead(nn.Module):
    """Small discriminator head at one feature scale."""

    def __init__(self, in_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, 1, 1, 1, 0),
        )

    def forward(self, x):
        return self.main(x)


# ==============================================================================
# G2 Helper Blocks (U-Net for Heatmap generation — unchanged)
# ==============================================================================

class FiLM(nn.Module):
    """Feature-wise Linear Modulation."""

    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, num_features)
        self.beta_fc = nn.Linear(cond_dim, num_features)
        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x, cond):
        gamma = 1 + self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class AttentionGate(nn.Module):
    """Attention gating for U-Net skip connections."""

    def __init__(self, gate_ch, skip_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or skip_ch // 2
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode='bilinear', align_corners=False)
        attn = self.psi(self.relu(g + s))
        return skip * attn
