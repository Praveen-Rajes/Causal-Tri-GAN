"""
CausalTriGAN-StyleGAN2 - Oracle: Frozen TorchXRayVision DenseNet-121 Wrapper
Differentiable preprocessing for causal intervention loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv


class Oracle(nn.Module):
    CHEXPERT_TO_TXV = {
        0:  None,   # No Finding
        1:  17,     # Enlarged Cardiomediastinum
        2:  10,     # Cardiomegaly
        3:  16,     # Lung Opacity
        4:  14,     # Lung Lesion
        5:  4,      # Edema
        6:  1,      # Consolidation
        7:  8,      # Pneumonia
        8:  0,      # Atelectasis
        9:  3,      # Pneumothorax
        10: 7,      # Pleural Effusion
        11: None,   # Pleural Other
        12: 15,     # Fracture
        13: None,   # Support Devices
    }

    def __init__(self):
        super().__init__()
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._build_mapping()
        self._register_cam_hook()

    def _build_mapping(self):
        self.valid_chexpert = []
        self.txv_indices = []
        for cp_idx, txv_idx in self.CHEXPERT_TO_TXV.items():
            if txv_idx is not None:
                self.valid_chexpert.append(cp_idx)
                self.txv_indices.append(txv_idx)

    def _register_cam_hook(self):
        """Register forward hook on denseblock4 for CAM computation."""
        self._cam_feats = {}
        def _hook(m, inp, out):
            self._cam_feats['feat'] = out
        self.model.features.denseblock4.register_forward_hook(_hook)

    @torch.no_grad()
    def compute_cam(self, x_3ch, label_vector):
        """
        Proper CAM (Class Activation Mapping) matching notebook implementation.
        x_3ch        : [1, 3, 256, 256] in [-1, 1]
        label_vector : [14] float tensor
        Returns      : [1, 1, 256, 256] in [0, 1]
        """
        # Preprocess: 1ch, resize to 224, scale by 1024
        gray = x_3ch[:, 0:1]
        gray = F.interpolate(gray, size=(224, 224), mode='bilinear', align_corners=False)
        gray = gray * 1024.0

        # Forward pass (populates self._cam_feats via hook)
        _ = self.model(gray)
        feat = self._cam_feats['feat']         # [1, 1024, h, w]
        w = self.model.classifier.weight       # [n_pathologies, 1024]

        lv = label_vector.to(feat.device)
        cam = torch.zeros(1, 1, feat.shape[2], feat.shape[3], device=feat.device)
        ws = 0.0

        for ci in range(14):
            v = lv[ci].item()
            if v <= 0:
                continue
            ti = self.CHEXPERT_TO_TXV.get(ci)
            if ti is None:
                continue
            # Weighted feature activation: classifier_weight * feature_maps
            c = (w[ti].view(1, -1, 1, 1) * feat).sum(1, keepdim=True).relu()
            cam = cam + c * v
            ws += v

        if ws > 0:
            cam = cam / ws

        cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
        mn, mx = cam.min(), cam.max()
        return (cam - mn) / (mx - mn + 1e-8)

    def preprocess(self, x):
        # TorchXRayVision expects 1ch input — extract grayscale if 3ch
        if x.shape[1] == 3:
            x = x[:, 0:1]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x * 1024.0
        return x

    def forward(self, x, preprocess=True):
        if preprocess:
            x = self.preprocess(x)
        preds = self.model(x)
        return preds

    def get_chexpert_preds(self, x, preprocess=True):
        txv_preds = self.forward(x, preprocess)
        B = txv_preds.size(0)
        preds_14 = torch.zeros(B, 14, device=txv_preds.device, dtype=txv_preds.dtype)
        for cp_idx, txv_idx in zip(self.valid_chexpert, self.txv_indices):
            preds_14[:, cp_idx] = txv_preds[:, txv_idx]
        return preds_14

    def verify_on_real(self, dataloader, device, num_batches=50):
        from sklearn.metrics import roc_auc_score
        import numpy as np
        all_preds, all_labels = [], []
        self.eval()
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(dataloader):
                if i >= num_batches:
                    break
                imgs = imgs.to(device)
                preds = self.get_chexpert_preds(imgs)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        aucs = {}
        chexpert_labels = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices"
        ]
        for i, name in enumerate(chexpert_labels):
            if i in self.valid_chexpert and all_labels[:, i].sum() > 10:
                try:
                    auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                    aucs[name] = auc
                except ValueError:
                    aucs[name] = float('nan')
        mean_auc = np.nanmean(list(aucs.values()))
        print(f"[Oracle Verification] Mean AUC: {mean_auc:.4f}")
        for name, auc in aucs.items():
            print(f"  {name}: {auc:.4f}")
        return aucs, mean_auc
