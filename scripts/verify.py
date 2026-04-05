#!/usr/bin/env python3
"""
CausalTriGAN-ProjectedGAN - Data & Environment Verification


"""
import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verify_gpu():
    print("\n[1/5] GPU Verification")
    print("-" * 40)
    if not torch.cuda.is_available():
        print("  [FAIL] CUDA not available!")
        return False
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM: {vram:.1f} GB")
    if vram < 20:
        print(f"  [WARN] Low VRAM. Reduce batch_size if OOM.")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    try:
        with torch.amp.autocast('cuda'):
            x = torch.randn(2, 2, device='cuda')
            y = x @ x
        print("  AMP: OK")
    except Exception as e:
        print(f"  [WARN] AMP issue: {e}")
    return True


def verify_dependencies():
    print("\n[2/5] Dependencies")
    print("-" * 40)
    ok = True
    deps = {
        "torchxrayvision": "torchxrayvision",
        "transformers": "transformers",
        "datasets (HuggingFace)": "datasets",
        "clean-fid": "cleanfid",
        "lpips": "lpips",
        "nltk": "nltk",
        "tensorboard": "tensorboard",
        "PIL": "PIL",
        "sklearn": "sklearn",
        "tqdm": "tqdm",
        "pandas": "pandas",
    }
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"  {name}: OK")
        except ImportError:
            print(f"  {name}: [MISSING] pip install {module}")
            ok = False
    return ok


def verify_data(data_root):
    print("\n[3/5] Data Verification (HuggingFace Dataset)")
    print("-" * 40)

    if not os.path.exists(data_root):
        print(f"  [FAIL] Dataset not found: {data_root}")
        return False

    try:
        from datasets import load_from_disk
        dataset_dict = load_from_disk(data_root)
        print(f"  Loaded: {type(dataset_dict).__name__}")

        if hasattr(dataset_dict, 'keys'):
            print(f"  Splits: {list(dataset_dict.keys())}")
            ds = dataset_dict["train"] if "train" in dataset_dict else list(dataset_dict.values())[0]
        else:
            ds = dataset_dict

        print(f"  Samples: {len(ds)}")
        print(f"  Columns: {ds.column_names}")

        required = ["image"]
        label_cols = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices"
        ]

        for col in required:
            if col not in ds.column_names:
                print(f"  [FAIL] Missing column: {col}")
                return False

        found_labels = [c for c in label_cols if c in ds.column_names]
        print(f"  Label columns found: {len(found_labels)}/14")

        sample = ds[0]
        img = sample["image"]
        print(f"  Sample image: size={img.size}, mode={img.mode}")

        print(f"  Label distribution (first 1000 samples):")
        for col in found_labels[:6]:
            values = np.array(ds[:min(1000, len(ds))][col], dtype=float)
            values = np.nan_to_num(values, nan=0.0)
            pos = (values == 1).sum()
            neg = (values == 0).sum()
            unc = (values == -1).sum()
            print(f"    {col}: pos={pos}, neg={neg}, uncertain={unc}")

        if len(ds) < 1000:
            print(f"  [WARN] Only {len(ds)} samples. Expected ~40K+")
        else:
            print(f"  [OK] {len(ds)} samples")

        return True

    except Exception as e:
        print(f"  [FAIL] Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_oracle():
    print("\n[4/5] Oracle Verification")
    print("-" * 40)
    try:
        import torchxrayvision as xrv
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.eval()
        print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
        print(f"  Pathologies: {model.pathologies[:5]}... ({len(model.pathologies)} total)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        x = torch.randn(1, 1, 224, 224, device=device) * 1024
        with torch.no_grad():
            preds = model(x)
        print(f"  Forward pass: shape={preds.shape}, range=[{preds.min():.3f}, {preds.max():.3f}]")
        print("  Oracle: OK")
        return True
    except Exception as e:
        print(f"  [FAIL] Oracle error: {e}")
        return False


def verify_models():
    print("\n[5/5] Model Architecture Test (ProjectedGAN)")
    print("-" * 40)
    from configs.config import get_config
    from models.generator_g1 import GeneratorG1
    from models.generator_g2 import GeneratorG2
    from models.discriminator import ProjectedDiscriminator as Discriminator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_config()

    # G1 (FastGAN / ProjectedGAN)
    G1 = GeneratorG1(
        z_dim=cfg.z_dim, c_embed_dim=cfg.c_embed_dim,
        num_classes=cfg.num_classes, img_channels=cfg.img_channels,
        ngf=cfg.ngf
    ).to(device)
    z = torch.randn(2, cfg.z_dim, device=device)
    labels = torch.zeros(2, cfg.num_classes, device=device)
    labels[0, 2] = 1.0  # Cardiomegaly
    labels[1, 10] = 1.0  # Pleural Effusion
    with torch.no_grad():
        fake = G1(z, labels)
    print(f"  G1 (FastGAN): {sum(p.numel() for p in G1.parameters())/1e6:.1f}M params, "
          f"output={fake.shape}, range=[{fake.min():.2f}, {fake.max():.2f}]")

    # G2 (always 1ch input)
    G2 = GeneratorG2(
        img_channels=1, num_classes=cfg.num_classes,
        c_embed_dim=cfg.c_embed_dim, heatmap_temperature=cfg.heatmap_temperature
    ).to(device)
    with torch.no_grad():
        fake_gray = fake[:, 0:1]  # Extract 1ch grayscale for G2
        hmap = G2(fake_gray, labels)
    print(f"  G2 (U-Net): {sum(p.numel() for p in G2.parameters())/1e6:.1f}M params, "
          f"output={hmap.shape}, range=[{hmap.min():.2f}, {hmap.max():.2f}]")

    # D (Projected Discriminator)
    D = Discriminator(
        img_channels=cfg.img_channels, num_classes=cfg.num_classes,
        backbone_name=cfg.d_backbone, ccm_channels=cfg.d_ccm_channels
    ).to(device)
    with torch.no_grad():
        score, aux = D(fake, labels)
    print(f"  D  (Projected): {sum(p.numel() for p in D.parameters())/1e6:.1f}M params, "
          f"score={score.shape}, aux={aux.shape}")

    # Memory estimate
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    G1.train(); G2.train(); D.train()
    z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
    labels = torch.zeros(cfg.batch_size, cfg.num_classes, device=device)
    with torch.amp.autocast('cuda'):
        fake = G1(z, labels)
        hmap = G2(fake[:, 0:1], labels)
        score, aux = D(fake, labels)
        loss = -score.mean()
    loss.backward()
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak VRAM (1 step, BS={cfg.batch_size}): {peak_mem:.2f} GB")

    del G1, G2, D
    torch.cuda.empty_cache()
    return True


def main():
    parser = argparse.ArgumentParser(description="CausalTriGAN-ProjectedGAN Verification")
    parser.add_argument("--data_root", type=str,
                        default="/workspace/data/chexpert_dataset")
    args = parser.parse_args()

    print("=" * 60)
    print("  CausalTriGAN-ProjectedGAN Environment Verification")
    print("=" * 60)

    results = {}
    results["gpu"] = verify_gpu()
    results["deps"] = verify_dependencies()
    results["data"] = verify_data(args.data_root)
    results["oracle"] = verify_oracle()
    results["models"] = verify_models()

    print(f"\n{'='*60}")
    print("  VERIFICATION SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  All checks passed! Ready to train with ProjectedGAN.")
        print(f"  Run: python scripts/train.py --data_root {args.data_root}")
    else:
        print("\n  Some checks failed. Fix issues above before training.")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
