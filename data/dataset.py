"""
Loads from a saved DatasetDict with 'image' + 14 CheXpert label columns.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_from_disk


CHEXPERT_COLUMNS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]


class CheXpertHFDataset(Dataset):
    def __init__(self, hf_dataset, img_size=256, uncertainty_policy="uones",
                 is_train=True):
        self.dataset = hf_dataset
        self.img_size = img_size
        self.uncertainty_policy = uncertainty_policy

        labels = []
        for col in CHEXPERT_COLUMNS:
            if col in hf_dataset.column_names:
                col_data = np.array(hf_dataset[col], dtype=np.float32)
            else:
                col_data = np.zeros(len(hf_dataset), dtype=np.float32)
            labels.append(col_data)

        self.labels = np.stack(labels, axis=1)
        self.labels = np.nan_to_num(self.labels, nan=0.0)

        if uncertainty_policy == "uones":
            self.labels[self.labels == -1] = 1.0
        else:
            self.labels[self.labels == -1] = 0.0

        self.labels = np.clip(self.labels, 0, 1)

        # Output 3ch grayscale (matches ProjectedGAN's expected input format)
        # CXR → 3-channel grayscale → normalize to [-1, 1]
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05),
                                        scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        pos_counts = (self.labels == 1).sum(axis=0)
        split_name = "train" if is_train else "val"
        print(f"[Dataset] {split_name}: {len(self)} images, policy: {uncertainty_policy}")
        print(f"  Top pathologies: ", end="")
        sorted_idx = np.argsort(-pos_counts)
        for i in sorted_idx[:5]:
            print(f"{CHEXPERT_COLUMNS[i]}={int(pos_counts[i])}, ", end="")
        print()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"]
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        img = self.transform(img)  # [3, H, W] in [-1, 1]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def get_dataloaders(cfg):
    print(f"[Data] Loading dataset from: {cfg.data_root}")
    dataset_dict = load_from_disk(cfg.data_root)

    if hasattr(dataset_dict, 'keys'):
        if "train" in dataset_dict:
            full_dataset = dataset_dict["train"]
        else:
            first_key = list(dataset_dict.keys())[0]
            full_dataset = dataset_dict[first_key]
    else:
        full_dataset = dataset_dict

    print(f"[Data] Total samples: {len(full_dataset)}")
    print(f"[Data] Columns: {full_dataset.column_names}")

    has_val = (hasattr(dataset_dict, 'keys') and
               any(k in dataset_dict for k in ["validate", "val", "validation", "test"]))

    if has_val:
        for val_key in ["validate", "val", "validation", "test"]:
            if val_key in dataset_dict:
                train_hf = full_dataset
                val_hf = dataset_dict[val_key]
                print(f"[Data] Using existing splits: train={len(train_hf)}, val={len(val_hf)}")
                break
    else:
        split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_hf = split["train"]
        val_hf = split["test"]
        print(f"[Data] Created 90/10 split: train={len(train_hf)}, val={len(val_hf)}")

    train_ds = CheXpertHFDataset(train_hf, img_size=cfg.img_size,
                                  uncertainty_policy=cfg.uncertainty_policy, is_train=True)
    val_ds = CheXpertHFDataset(val_hf, img_size=cfg.img_size,
                                uncertainty_policy=cfg.uncertainty_policy, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=cfg.num_workers > 0,
                              prefetch_factor=3 if cfg.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
                            persistent_workers=cfg.num_workers > 0)

    print(f"[Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader
