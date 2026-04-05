"""
CausalTriGAN-StyleGAN2 - Anatomical Prior Loss (L_anat)
KL divergence between heatmap and Gaussian spatial priors per pathology.
"""
import torch
import torch.nn.functional as F


def build_gaussian_prior(center_y, center_x, sigma, size=256, device="cuda"):
    y = torch.linspace(0, 1, size, device=device)
    x = torch.linspace(0, 1, size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    gaussian = torch.exp(-((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / (gaussian.sum() + 1e-8)
    return gaussian


class AnatomicalPriorLoss(torch.nn.Module):
    def __init__(self, anatomical_priors, img_size=256, device="cuda"):
        super().__init__()
        self.img_size = img_size
        self.priors = {}
        for class_idx, (cy, cx, sigma) in anatomical_priors.items():
            prior = build_gaussian_prior(cy, cx, sigma, img_size, device)
            self.priors[class_idx] = prior

    def forward(self, heatmap, labels):
        B = heatmap.size(0)
        loss = torch.tensor(0.0, device=heatmap.device)
        count = 0
        hmap = heatmap.squeeze(1)
        for class_idx, prior in self.priors.items():
            mask = labels[:, class_idx] > 0.5
            if mask.sum() == 0:
                continue
            h_pos = hmap[mask]
            h_norm = h_pos / (h_pos.sum(dim=(-2, -1), keepdim=True) + 1e-8)
            prior_expanded = prior.unsqueeze(0).expand_as(h_norm)
            kl = F.kl_div(
                (h_norm + 1e-8).log(),
                prior_expanded,
                reduction='batchmean',
                log_target=False
            )
            loss = loss + kl
            count += 1
        if count > 0:
            loss = loss / count
        return loss
