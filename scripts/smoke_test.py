#!/usr/bin/env python3
"""
Performs one forward + backward pass through the full pipeline.

"""
import os
import sys
import argparse
import traceback
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def test(name, fn, device):
    try:
        fn(device)
        print(f"  [{PASS}] {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  [{FAIL}] {name}: {e}")
        traceback.print_exc()
        results.append((name, False))


# ============================================================
def test_config(device):
    from configs.config import get_config, ABLATION_CONFIGS
    cfg = get_config()
    assert cfg.z_dim == 256
    assert cfg.img_channels == 3, f"img_channels should be 3, got {cfg.img_channels}"
    assert cfg.total_kimg == 1200
    assert cfg.phase1_kimg + cfg.phase2_kimg + cfg.phase3_kimg == cfg.total_kimg
    assert cfg.get_phase(0) == 1
    assert cfg.get_phase(cfg.phase1_kimg + 1) == 2
    assert cfg.get_phase(cfg.total_kimg) == 3
    assert cfg.lambda_r1 == 0.0, "R1 should be disabled for ProjectedGAN"
    for abl in ABLATION_CONFIGS:
        get_config(ablation=abl)


def test_g1_forward(device):
    from configs.config import get_config
    from models.generator_g1 import GeneratorG1
    cfg = get_config()
    G1 = GeneratorG1(
        z_dim=cfg.z_dim, c_embed_dim=cfg.c_embed_dim,
        num_classes=cfg.num_classes, img_channels=cfg.img_channels,
        ngf=cfg.ngf
    ).to(device)
    z = torch.randn(2, cfg.z_dim, device=device)
    labels = torch.zeros(2, cfg.num_classes, device=device)
    labels[0, 2] = 1.0
    labels[1, 10] = 1.0

    # Basic forward — G1 outputs 3ch RGB
    img = G1(z, labels)
    assert img.shape == (2, 3, 256, 256), f"Expected (2,3,256,256), got {img.shape}"

    # With truncation
    img2 = G1(z, labels, truncation_psi=0.7)
    assert img2.shape == (2, 3, 256, 256)

    # get_gray helper
    gray = G1.get_gray(img)
    assert gray.shape == (2, 1, 256, 256)

    del G1
    torch.cuda.empty_cache() if device.type == 'cuda' else None


def test_g1_backward(device):
    from configs.config import get_config
    from models.generator_g1 import GeneratorG1
    cfg = get_config()
    G1 = GeneratorG1(
        z_dim=cfg.z_dim, c_embed_dim=cfg.c_embed_dim,
        num_classes=cfg.num_classes, img_channels=cfg.img_channels,
        ngf=cfg.ngf
    ).to(device)
    z = torch.randn(2, cfg.z_dim, device=device)
    labels = torch.zeros(2, cfg.num_classes, device=device)
    labels[0, 2] = 1.0
    img = G1(z, labels)
    loss = img.mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in G1.parameters())
    assert has_grad, "No gradients flowed through G1"
    del G1
    torch.cuda.empty_cache() if device.type == 'cuda' else None


def test_g2_forward(device):
    from configs.config import get_config
    from models.generator_g2 import GeneratorG2
    cfg = get_config()
    # G2 always takes 1ch input
    G2 = GeneratorG2(
        img_channels=1, num_classes=cfg.num_classes,
        c_embed_dim=cfg.c_embed_dim, heatmap_temperature=cfg.heatmap_temperature
    ).to(device)
    fake_gray = torch.randn(2, 1, 256, 256, device=device)
    labels = torch.zeros(2, cfg.num_classes, device=device)
    labels[0, 2] = 1.0
    hmap = G2(fake_gray, labels)
    assert hmap.shape == (2, 1, 256, 256), f"Expected (2,1,256,256), got {hmap.shape}"
    assert hmap.min() >= 0.0 and hmap.max() <= 1.0, "Heatmap should be in [0, 1]"
    del G2
    torch.cuda.empty_cache() if device.type == 'cuda' else None


def test_discriminator_forward(device):
    from configs.config import get_config
    from models.discriminator import ProjectedDiscriminator
    cfg = get_config()
    D = ProjectedDiscriminator(
        img_channels=cfg.img_channels, num_classes=cfg.num_classes,
        backbone_name=cfg.d_backbone, ccm_channels=cfg.d_ccm_channels
    ).to(device)
    # D receives 3ch input
    x = torch.randn(2, 3, 256, 256, device=device)
    labels = torch.zeros(2, cfg.num_classes, device=device)
    labels[0, 2] = 1.0
    score, aux = D(x, labels)
    assert score.shape == (2, 1), f"Score shape: {score.shape}"
    assert aux.shape == (2, cfg.num_classes), f"Aux shape: {aux.shape}"
    del D
    torch.cuda.empty_cache() if device.type == 'cuda' else None


def test_discriminator_backward(device):
    from configs.config import get_config
    from models.discriminator import ProjectedDiscriminator
    cfg = get_config()
    D = ProjectedDiscriminator(
        img_channels=cfg.img_channels, num_classes=cfg.num_classes,
        backbone_name=cfg.d_backbone, ccm_channels=cfg.d_ccm_channels
    ).to(device)
    x = torch.randn(2, 3, 256, 256, device=device)
    labels = torch.zeros(2, cfg.num_classes, device=device)
    labels[0, 2] = 1.0
    score, aux = D(x, labels)
    loss = -score.mean() + aux.mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                   for p in D.parameters() if p.requires_grad)
    assert has_grad, "No gradients flowed through D"
    del D
    torch.cuda.empty_cache() if device.type == 'cuda' else None


def test_losses(device):
    from losses.adversarial import logistic_loss_d, logistic_loss_g
    from losses.regularization import (
        auxiliary_loss,
        total_variation_loss, sparsity_loss, diversity_loss,
    )
    # Adversarial
    d_real = torch.randn(4, 1, device=device)
    d_fake = torch.randn(4, 1, device=device)
    ld = logistic_loss_d(d_real, d_fake)
    lg = logistic_loss_g(d_fake)
    assert ld.shape == (), f"loss_d shape: {ld.shape}"
    assert lg.shape == (), f"loss_g shape: {lg.shape}"

    # Auxiliary
    logits = torch.randn(4, 14, device=device)
    labels = torch.zeros(4, 14, device=device)
    labels[:, 2] = 1.0
    a_loss = auxiliary_loss(logits, labels)
    assert a_loss.shape == ()

    # TV, sparsity, diversity
    hmap = torch.rand(2, 1, 256, 256, device=device)
    tv = total_variation_loss(hmap)
    sp = sparsity_loss(hmap)
    assert tv.shape == () and sp.shape == ()

    z1 = torch.randn(2, 256, device=device)
    z2 = torch.randn(2, 256, device=device)
    x1 = torch.randn(2, 3, 256, 256, device=device)
    x2 = torch.randn(2, 3, 256, 256, device=device)
    div = diversity_loss(z1, z2, x1, x2)
    assert div.shape == ()


def test_causal_loss(device):
    from losses.causal import causal_intervention_loss

    class MockOracle:
        valid_chexpert = [2, 3, 5, 8, 10]
        def get_chexpert_preds(self, x):
            return torch.sigmoid(torch.randn(x.size(0), 14, device=x.device))

    oracle = MockOracle()
    # Causal loss handles both 3ch and 1ch input (extracts grayscale internally)
    x_fake = torch.randn(4, 3, 256, 256, device=device)
    hmap = torch.rand(4, 1, 256, 256, device=device)
    labels = torch.zeros(4, 14, device=device)
    labels[:, 2] = 1.0
    labels[:, 10] = 1.0

    loss, metrics = causal_intervention_loss(x_fake, hmap, labels, oracle)
    assert loss.shape == ()
    assert "sufficiency_score" in metrics
    assert "necessity_score" in metrics


def test_anatomical_loss(device):
    from configs.config import get_config
    from losses.anatomical import AnatomicalPriorLoss
    cfg = get_config()
    anat = AnatomicalPriorLoss(cfg.anatomical_priors, cfg.img_size, str(device))
    hmap = torch.rand(4, 1, 256, 256, device=device)
    labels = torch.zeros(4, 14, device=device)
    labels[:, 2] = 1.0
    loss = anat(hmap, labels)
    assert loss.shape == ()


def test_ema(device):
    from models.generator_g1 import GeneratorG1
    from models.ema import EMA
    from configs.config import get_config
    cfg = get_config()
    G1 = GeneratorG1(
        z_dim=cfg.z_dim, c_embed_dim=cfg.c_embed_dim,
        num_classes=cfg.num_classes, img_channels=cfg.img_channels,
        ngf=cfg.ngf
    ).to(device)
    ema = EMA(G1, decay=0.999, start_step=0)
    ema.update(G1, step=0)
    ema.update(G1, step=1)
    sd = ema.state_dict()
    assert len(sd) > 0
    del G1, ema
    torch.cuda.empty_cache() if device.type == 'cuda' else None


def test_ada(device):
    from utils.ada import AdaptiveAugment, ada_augment
    ada = AdaptiveAugment(target_rt=0.6, speed=500, max_p=0.85)
    assert ada.get_p() == 0.0
    for _ in range(20):
        scores = torch.ones(16, device=device)
        ada.update(scores, 16)
    assert ada.get_p() > 0, f"ADA p should have increased, got {ada.get_p()}"

    x = torch.randn(4, 3, 256, 256, device=device)
    x_aug = ada_augment(x, p=0.5)
    assert x_aug.shape == x.shape

    x_no_aug = ada_augment(x, p=0.0)
    assert torch.equal(x, x_no_aug)


def test_diffaugment(device):
    from utils.augmentation import DiffAugment
    x = torch.randn(4, 3, 256, 256, device=device)
    x_aug = DiffAugment(x, 'color,translation,cutout')
    assert x_aug.shape == x.shape

    x_no = DiffAugment(x, '')
    assert torch.equal(x, x_no)


def test_full_pipeline_forward(device):
    """End-to-end: z -> G1 -> img(3ch) -> G2(1ch) -> hmap -> D(3ch) -> score."""
    from configs.config import get_config
    from models.generator_g1 import GeneratorG1
    from models.generator_g2 import GeneratorG2
    from models.discriminator import ProjectedDiscriminator

    cfg = get_config()
    G1 = GeneratorG1(
        z_dim=cfg.z_dim, c_embed_dim=cfg.c_embed_dim,
        num_classes=cfg.num_classes, img_channels=cfg.img_channels,
        ngf=cfg.ngf
    ).to(device)
    G2 = GeneratorG2(
        img_channels=1, num_classes=cfg.num_classes,
        c_embed_dim=cfg.c_embed_dim, heatmap_temperature=cfg.heatmap_temperature
    ).to(device)
    D = ProjectedDiscriminator(
        img_channels=cfg.img_channels, num_classes=cfg.num_classes,
        backbone_name=cfg.d_backbone, ccm_channels=cfg.d_ccm_channels
    ).to(device)

    z = torch.randn(2, cfg.z_dim, device=device)
    labels = torch.zeros(2, cfg.num_classes, device=device)
    labels[0, 2] = 1.0
    labels[1, 10] = 1.0

    img = G1(z, labels)           # [2, 3, 256, 256]
    gray = img[:, 0:1]            # [2, 1, 256, 256]
    hmap = G2(gray, labels)       # [2, 1, 256, 256]
    score, aux = D(img, labels)   # D gets 3ch

    assert img.shape == (2, 3, 256, 256)
    assert hmap.shape == (2, 1, 256, 256)
    assert score.shape == (2, 1)
    assert aux.shape == (2, 14)

    del G1, G2, D
    torch.cuda.empty_cache() if device.type == 'cuda' else None


def test_full_pipeline_backward(device):
    """End-to-end forward + backward to verify gradient flow."""
    from configs.config import get_config
    from models.generator_g1 import GeneratorG1
    from models.generator_g2 import GeneratorG2
    from models.discriminator import ProjectedDiscriminator
    from losses.adversarial import logistic_loss_d, logistic_loss_g
    from losses.regularization import auxiliary_loss

    cfg = get_config()
    G1 = GeneratorG1(
        z_dim=cfg.z_dim, c_embed_dim=cfg.c_embed_dim,
        num_classes=cfg.num_classes, img_channels=cfg.img_channels,
        ngf=cfg.ngf
    ).to(device)
    G2 = GeneratorG2(
        img_channels=1, num_classes=cfg.num_classes,
        c_embed_dim=cfg.c_embed_dim, heatmap_temperature=cfg.heatmap_temperature
    ).to(device)
    D = ProjectedDiscriminator(
        img_channels=cfg.img_channels, num_classes=cfg.num_classes,
        backbone_name=cfg.d_backbone, ccm_channels=cfg.d_ccm_channels
    ).to(device)

    opt_G = torch.optim.Adam(
        list(G1.parameters()) + list(G2.parameters()), lr=1e-4
    )
    d_trainable = [p for p in D.parameters() if p.requires_grad]
    opt_D = torch.optim.Adam(d_trainable, lr=1e-4)

    z = torch.randn(2, cfg.z_dim, device=device)
    real = torch.randn(2, 3, 256, 256, device=device)  # 3ch real
    labels = torch.zeros(2, cfg.num_classes, device=device)
    labels[0, 2] = 1.0

    # D step
    opt_D.zero_grad()
    with torch.no_grad():
        fake = G1(z, labels)
    d_real, aux_real = D(real, labels)
    d_fake, aux_fake = D(fake, labels)
    loss_D = logistic_loss_d(d_real, d_fake) + auxiliary_loss(aux_real, labels)
    loss_D.backward()
    opt_D.step()

    # G step
    opt_G.zero_grad()
    fake = G1(z, labels)
    gray = fake[:, 0:1]
    hmap = G2(gray, labels)
    d_fake_g, aux_g = D(fake, labels)
    loss_G = logistic_loss_g(d_fake_g) + auxiliary_loss(aux_g, labels)
    loss_G.backward()
    opt_G.step()

    assert loss_D.item() != 0, "D loss is zero"
    assert loss_G.item() != 0, "G loss is zero"

    del G1, G2, D
    torch.cuda.empty_cache() if device.type == 'cuda' else None


def test_phase23_label_embed(device):
    """Test Phase 2+3 mode: label_embed + G2 on real images (matches notebook)."""
    import torch.nn as nn
    from configs.config import get_config
    from models.generator_g2 import GeneratorG2

    cfg = get_config()

    # Shared label_embed (matches notebook: 14→256→128→14)
    label_embed = nn.Sequential(
        nn.Linear(cfg.num_classes, 256), nn.SiLU(),
        nn.Linear(256, 128), nn.SiLU(),
        nn.Linear(128, cfg.num_classes),
    ).to(device)

    G2 = GeneratorG2(
        img_channels=1, num_classes=cfg.num_classes,
        c_embed_dim=cfg.c_embed_dim, heatmap_temperature=cfg.heatmap_temperature
    ).to(device)

    # Simulate real images (Phase 2+3 trains on real, not G1 output)
    real_imgs = torch.randn(2, 3, 256, 256, device=device)
    labels = torch.zeros(2, cfg.num_classes, device=device)
    labels[0, 2] = 1.0

    c_soft = label_embed(labels)     # [2, 14] → [2, 14]
    img_gray = real_imgs[:, 0:1]     # Extract 1ch from real
    heatmap = G2(img_gray, c_soft)   # G2 uses c_soft, not raw labels

    assert c_soft.shape == (2, cfg.num_classes)
    assert heatmap.shape == (2, 1, 256, 256)

    # Verify gradients flow through label_embed + G2
    loss = heatmap.mean()
    loss.backward()
    g2_has_grad = any(p.grad is not None for p in G2.parameters())
    embed_has_grad = any(p.grad is not None for p in label_embed.parameters())
    assert g2_has_grad, "G2 got no gradients"
    assert embed_has_grad, "label_embed got no gradients"

    del G2, label_embed
    torch.cuda.empty_cache() if device.type == 'cuda' else None


def test_amp_forward(device):
    """Test mixed precision forward pass."""
    if device.type != 'cuda':
        return
    from configs.config import get_config
    from models.generator_g1 import GeneratorG1
    from models.discriminator import ProjectedDiscriminator

    cfg = get_config()
    G1 = GeneratorG1(
        z_dim=cfg.z_dim, c_embed_dim=cfg.c_embed_dim,
        num_classes=cfg.num_classes, img_channels=cfg.img_channels,
        ngf=cfg.ngf
    ).to(device)
    D = ProjectedDiscriminator(
        img_channels=cfg.img_channels, num_classes=cfg.num_classes,
        backbone_name=cfg.d_backbone, ccm_channels=cfg.d_ccm_channels
    ).to(device)

    z = torch.randn(2, cfg.z_dim, device=device)
    labels = torch.zeros(2, cfg.num_classes, device=device)
    labels[0, 2] = 1.0

    with torch.amp.autocast('cuda'):
        fake = G1(z, labels)
        score, aux = D(fake, labels)
    assert fake.shape == (2, 3, 256, 256)
    assert score.shape == (2, 1)

    del G1, D
    torch.cuda.empty_cache()


def test_checkpoint_save_load(device):
    """Test checkpoint round-trip."""
    import tempfile
    from configs.config import get_config
    from models.generator_g1 import GeneratorG1
    from models.ema import EMA

    cfg = get_config()
    G1 = GeneratorG1(
        z_dim=cfg.z_dim, c_embed_dim=cfg.c_embed_dim,
        num_classes=cfg.num_classes, img_channels=cfg.img_channels,
        ngf=cfg.ngf
    ).to(device)
    ema = EMA(G1, decay=0.999, start_step=0)

    state = {
        "G1": G1.state_dict(),
        "ema_g1": ema.state_dict(),
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(state, f.name)
        ckpt = torch.load(f.name, map_location=device, weights_only=False)

    G1_new = GeneratorG1(
        z_dim=cfg.z_dim, c_embed_dim=cfg.c_embed_dim,
        num_classes=cfg.num_classes, img_channels=cfg.img_channels,
        ngf=cfg.ngf
    ).to(device)
    G1_new.load_state_dict(ckpt["ema_g1"])

    z = torch.randn(2, cfg.z_dim, device=device)
    labels = torch.zeros(2, cfg.num_classes, device=device)
    labels[0, 2] = 1.0
    with torch.no_grad():
        img = G1_new(z, labels, truncation_psi=0.7)
    assert img.shape == (2, 3, 256, 256)

    os.unlink(f.name)
    del G1, G1_new, ema
    torch.cuda.empty_cache() if device.type == 'cuda' else None


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CausalTriGAN-ProjectedGAN Smoke Test")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to test on (default: cuda if available)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"  CausalTriGAN-ProjectedGAN Smoke Test")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  PyTorch: {torch.__version__}")
    print(f"{'='*60}\n")

    tests = [
        ("Config loading & validation", test_config),
        ("G1 (ProjectedGAN) forward pass", test_g1_forward),
        ("G1 (ProjectedGAN) backward pass", test_g1_backward),
        ("G2 (U-Net) forward pass", test_g2_forward),
        ("Discriminator forward pass", test_discriminator_forward),
        ("Discriminator backward pass", test_discriminator_backward),
        ("All loss functions", test_losses),
        ("Causal intervention loss", test_causal_loss),
        ("Anatomical prior loss", test_anatomical_loss),
        ("EMA save/load", test_ema),
        ("ADA augmentation", test_ada),
        ("DiffAugment", test_diffaugment),
        ("Full pipeline: forward", test_full_pipeline_forward),
        ("Full pipeline: forward + backward", test_full_pipeline_backward),
        ("Phase 2+3: label_embed + G2 on real images", test_phase23_label_embed),
        ("Checkpoint save/load roundtrip", test_checkpoint_save_load),
    ]

    if device.type == 'cuda':
        tests.append(("AMP mixed precision", test_amp_forward))

    print(f"Running {len(tests)} tests...\n")
    for name, fn in tests:
        test(name, fn, device)

    # Summary
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{len(results)} passed, {failed} failed")
    if failed == 0:
        print(f"  ALL TESTS PASSED - Safe to train!")
    else:
        print(f"  FAILURES:")
        for name, ok in results:
            if not ok:
                print(f"    - {name}")
        print(f"  FIX these before training!")
    print(f"{'='*60}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  Peak VRAM usage during tests: {peak:.2f} GB")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
