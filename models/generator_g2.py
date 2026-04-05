"""
CausalTriGAN - G2: U-Net Heatmap Generator
Takes generated CXR (1ch grayscale) + labels → heatmap H ∈ [0,1]^{1×256×256}
Uses Attention Gates + FiLM conditioning. ~35M parameters.

G2 always receives 1-channel input (extracted from G1's 3ch output via [:, 0:1]).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import FiLM, AttentionGate


class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat


class DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.attn_gate = AttentionGate(in_ch, skip_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.film = FiLM(cond_dim, out_ch)

    def forward(self, x, skip, cond):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip = self.attn_gate(x, skip)
        h = torch.cat([x, skip], dim=1)
        h = self.conv(h)
        h = self.film(h, cond)
        return h


class GeneratorG2(nn.Module):
    """
    U-Net Heatmap Generator with Attention Gates and FiLM conditioning.
    Encoder: X̂(1ch) → 64 → 128 → 256 → 512 → 1024
    Decoder: 1024 → 512 → 256 → 128 → 64 → 1 (sigmoid)
    """

    def __init__(self, img_channels=1, num_classes=14, c_embed_dim=128,
                 heatmap_temperature=0.5):
        super().__init__()
        self.temperature = heatmap_temperature

        # G2 always takes 1ch grayscale input (extracted from G1's 3ch output)
        in_channels = 1

        self.c_embed = nn.Sequential(
            nn.Linear(num_classes, c_embed_dim),
            nn.ReLU(),
            nn.Linear(c_embed_dim, c_embed_dim),
        )
        cond_dim = c_embed_dim

        self.enc1 = EncBlock(in_channels, 64)
        self.enc2 = EncBlock(64, 128)
        self.enc3 = EncBlock(128, 256)
        self.enc4 = EncBlock(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.dec4 = DecBlock(1024, 512, 512, cond_dim)
        self.dec3 = DecBlock(512, 256, 256, cond_dim)
        self.dec2 = DecBlock(256, 128, 128, cond_dim)
        self.dec1 = DecBlock(128, 64, 64, cond_dim)

        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x, labels):
        cond = self.c_embed(labels)
        h, s1 = self.enc1(x)
        h, s2 = self.enc2(h)
        h, s3 = self.enc3(h)
        h, s4 = self.enc4(h)
        h = self.bottleneck(h)
        h = self.dec4(h, s4, cond)
        h = self.dec3(h, s3, cond)
        h = self.dec2(h, s2, cond)
        h = self.dec1(h, s1, cond)
        logits = self.out_conv(h)
        heatmap = torch.sigmoid(logits / self.temperature)
        return heatmap
