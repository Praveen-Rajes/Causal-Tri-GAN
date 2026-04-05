"""
CausalTriGAN-StyleGAN2 - Causal Intervention Loss (L_causal)
Core novelty: necessity + sufficiency testing via Oracle masking.
"""
import torch
import torch.nn.functional as F
from torch.amp import autocast


def causal_intervention_loss(x_fake, heatmap, labels, oracle,
                              lambda_comp=1.0, valid_indices=None,
                              mode="both"):
    # Oracle expects 1ch input — extract grayscale from 3ch G1 output
    if x_fake.shape[1] == 3:
        x_1ch = x_fake[:, 0:1]
    else:
        x_1ch = x_fake

    x_masked = x_1ch * heatmap
    oracle_preds_masked = oracle.get_chexpert_preds(x_masked)

    x_complement = x_1ch * (1.0 - heatmap)
    oracle_preds_complement = oracle.get_chexpert_preds(x_complement)

    if valid_indices is None:
        valid_indices = oracle.valid_chexpert

    mask = torch.zeros_like(labels)
    for idx in valid_indices:
        mask[:, idx] = labels[:, idx]

    n_valid = mask.sum().clamp(min=1.0)

    with autocast('cuda', enabled=False):
        preds_masked_f32 = oracle_preds_masked.float().clamp(1e-7, 1 - 1e-7)
        preds_comp_f32 = oracle_preds_complement.float().clamp(1e-7, 1 - 1e-7)
        labels_f32 = labels.float()
        mask_f32 = mask.float()

        loss_suf = F.binary_cross_entropy(
            preds_masked_f32, labels_f32, weight=mask_f32, reduction='sum'
        ) / n_valid

        target_zero = torch.zeros_like(labels_f32)
        loss_nec = F.binary_cross_entropy(
            preds_comp_f32, target_zero, weight=mask_f32, reduction='sum'
        ) / n_valid

    if mode == "sufficiency_only":
        loss = loss_suf
    elif mode == "necessity_only":
        loss = lambda_comp * loss_nec
    else:
        loss = loss_suf + lambda_comp * loss_nec

    with torch.no_grad():
        suf_score = (oracle_preds_masked * mask).sum() / n_valid
        nec_score = 1.0 - (oracle_preds_complement * mask).sum() / n_valid

    metrics = {
        "loss_sufficiency": loss_suf.item(),
        "loss_necessity": loss_nec.item(),
        "sufficiency_score": suf_score.item(),
        "necessity_score": nec_score.item(),
    }

    return loss, metrics
