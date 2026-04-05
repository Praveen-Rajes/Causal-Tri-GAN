#!/usr/bin/env python3
"""
CausalTriGAN - Prepare ProjectedGAN Dataset
Converts HuggingFace DatasetDict to ProjectedGAN zip format with dataset.json labels.

"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]
NUM_LABELS = 14


def find_label_columns(column_names):
    """Map CheXpert labels to actual column names in the dataset."""
    col_map = []
    for lbl in CHEXPERT_LABELS:
        key = lbl.lower().replace(' ', '_')
        match = None
        for col in column_names:
            ckey = col.lower().replace(' ', '_')
            if ckey == key or lbl.split()[0].lower() in col.lower():
                match = col
                break
        col_map.append(match)
    return col_map


def export_images_and_labels(dataset, img_dir, max_images=60000):
    """Export images as 256x256 RGB PNGs and build label list for dataset.json."""
    os.makedirs(img_dir, exist_ok=True)

    column_names = [c for c in dataset.column_names if c != "image"]
    col_map = find_label_columns(column_names)

    print(f"Label column mapping:")
    for i, (lbl, col) in enumerate(zip(CHEXPERT_LABELS, col_map)):
        status = col if col else "MISSING (will be 0)"
        print(f"  [{i:2d}] {lbl:30s} -> {status}")

    num_export = min(len(dataset), max_images)
    labels_list = []

    for i in tqdm(range(num_export), desc="Exporting images"):
        try:
            row = dataset[i]
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img)

            # Convert to 256x256 3-channel grayscale (matches ProjectedGAN input)
            gray = img.convert('L').resize((256, 256), Image.LANCZOS)
            rgb = Image.merge('RGB', [gray, gray, gray])

            fname = f"{i:06d}.png"
            rgb.save(os.path.join(img_dir, fname))

            # Build 14-dim label vector
            vec = []
            for col in col_map:
                if col is None:
                    vec.append(0.0)
                    continue
                val = row.get(col, 0)
                try:
                    v = float(val)
                    # U-ones: treat -1 (uncertain) as 1.0, positive as 1.0
                    vec.append(1.0 if v > 0 or v == -1 else 0.0)
                except (TypeError, ValueError):
                    vec.append(0.0)

            labels_list.append([fname, vec])
        except Exception as e:
            print(f"  [WARN] Skipping index {i}: {e}")
            continue

    # Write dataset.json (ProjectedGAN format)
    meta_path = os.path.join(img_dir, "dataset.json")
    with open(meta_path, 'w') as f:
        json.dump({"labels": labels_list}, f)

    print(f"\nExported {len(labels_list)} images to {img_dir}")
    print(f"Saved dataset.json with {NUM_LABELS}-dim labels (c_dim={NUM_LABELS})")

    # Print label stats
    label_arr = np.array([l[1] for l in labels_list])
    pos_counts = label_arr.sum(axis=0)
    print(f"\nLabel distribution:")
    for i, lbl in enumerate(CHEXPERT_LABELS):
        print(f"  {lbl:30s}: {int(pos_counts[i]):6d} ({pos_counts[i]/len(labels_list)*100:.1f}%)")

    return labels_list


def build_zip(img_dir, zip_path, projgan_dir):
    """Run ProjectedGAN's dataset_tool.py to create the zip."""
    dataset_tool = os.path.join(projgan_dir, "dataset_tool.py")

    if not os.path.exists(dataset_tool):
        print(f"\n[WARN] dataset_tool.py not found at {dataset_tool}")
        print("  Run 'python scripts/setup_projgan.py' first, or set --projgan_dir")
        print("  Falling back to manual zip creation...")
        build_zip_manual(img_dir, zip_path)
        return

    # Remove old zip if exists
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"Removed old zip: {zip_path}")

    python_exe = sys.executable or "python3"
    cmd = [
        python_exe, dataset_tool,
        f"--source={img_dir}",
        f"--dest={zip_path}",
        "--resolution=256x256",
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.stdout:
        lines = proc.stdout.strip().splitlines()
        print("--- dataset_tool stdout (last 20 lines) ---")
        print('\n'.join(lines[-20:]))
    if proc.stderr:
        lines = proc.stderr.strip().splitlines()
        print("--- dataset_tool stderr (last 20 lines) ---")
        print('\n'.join(lines[-20:]))

    if proc.returncode != 0 or not os.path.exists(zip_path):
        print(f"\n[WARN] dataset_tool failed (exit code {proc.returncode})")
        print("  Falling back to manual zip creation...")
        build_zip_manual(img_dir, zip_path)
        return

    size_gb = os.path.getsize(zip_path) / 1e9
    print(f"\nZIP ready: {zip_path} ({size_gb:.2f} GB)")


def build_zip_manual(img_dir, zip_path):
    """Manually create zip in ProjectedGAN format (fallback)."""
    import zipfile

    print(f"Creating zip manually: {zip_path}")
    pngs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    meta_path = os.path.join(img_dir, "dataset.json")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        # Add dataset.json
        if os.path.exists(meta_path):
            zf.write(meta_path, "dataset.json")

        # Add images
        for fname in tqdm(pngs, desc="Zipping"):
            zf.write(os.path.join(img_dir, fname), fname)

    size_gb = os.path.getsize(zip_path) / 1e9
    print(f"ZIP ready: {zip_path} ({size_gb:.2f} GB)")


def verify_zip(zip_path):
    """Verify the created zip has correct format."""
    import zipfile

    print(f"\nVerifying zip: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        if 'dataset.json' not in names:
            print("  [ERROR] dataset.json missing from zip!")
            return False

        meta = json.loads(zf.read('dataset.json').decode('utf-8'))
        labels = meta.get('labels', [])
        if not labels:
            print("  [ERROR] dataset.json has no labels!")
            return False

        first_label = labels[0][1] if labels else []
        if len(first_label) != NUM_LABELS:
            print(f"  [ERROR] Expected {NUM_LABELS}-dim labels, got {len(first_label)}")
            return False

        num_pngs = sum(1 for n in names if n.endswith('.png'))
        print(f"  OK: {num_pngs} images, {len(labels)} labels, c_dim={len(first_label)}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Prepare ProjectedGAN dataset from HuggingFace format")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to HuggingFace DatasetDict (saved with save_to_disk)")
    parser.add_argument("--output_dir", type=str, default="data/projgan_prepared",
                        help="Output directory for raw images and zip")
    parser.add_argument("--projgan_dir", type=str, default="projected_gan",
                        help="Path to projected_gan repo (for dataset_tool.py)")
    parser.add_argument("--max_images", type=int, default=60000,
                        help="Maximum number of images to export")
    parser.add_argument("--skip_zip", action="store_true",
                        help="Only export images + dataset.json, skip zip creation")
    args = parser.parse_args()

    # Load HuggingFace dataset
    print(f"Loading dataset from: {args.data_root}")
    dataset_dict = load_from_disk(args.data_root)

    # Get train split
    if hasattr(dataset_dict, 'keys'):
        if "train" in dataset_dict:
            dataset = dataset_dict["train"]
        else:
            first_key = list(dataset_dict.keys())[0]
            dataset = dataset_dict[first_key]
            print(f"  Using split: '{first_key}'")
    else:
        dataset = dataset_dict

    print(f"  Total samples: {len(dataset)}")
    print(f"  Columns: {dataset.column_names}")

    # Export images and labels
    img_dir = os.path.join(args.output_dir, "raw")
    labels_list = export_images_and_labels(dataset, img_dir, max_images=args.max_images)

    if args.skip_zip:
        print(f"\nSkipped zip creation. Raw images at: {img_dir}")
        return

    # Build zip
    zip_name = f"mimic_cxr_256_c{NUM_LABELS}.zip"
    zip_path = os.path.join(args.output_dir, zip_name)
    build_zip(img_dir, zip_path, args.projgan_dir)

    # Verify
    verify_zip(zip_path)

    print(f"\nDone! Use this zip for Phase 1 training:")
    print(f"  python projected_gan/train.py --data={zip_path} --cond=1 --cfg=fastgan ...")


if __name__ == "__main__":
    main()
