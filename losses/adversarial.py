"""
CausalTriGAN-StyleGAN2 - Adversarial Loss
Non-saturating logistic loss (StyleGAN2 standard) + optional hinge.
"""
import torch
import torch.nn.functional as F


def logistic_loss_d(d_real, d_fake):
    """StyleGAN2 non-saturating logistic discriminator loss."""
    loss_real = F.softplus(-d_real).mean()
    loss_fake = F.softplus(d_fake).mean()
    return loss_real + loss_fake


def logistic_loss_g(d_fake):
    """StyleGAN2 non-saturating logistic generator loss."""
    return F.softplus(-d_fake).mean()


# Keep hinge as alternative
def hinge_loss_d(d_real, d_fake):
    loss_real = F.relu(1.0 - d_real).mean()
    loss_fake = F.relu(1.0 + d_fake).mean()
    return loss_real + loss_fake


def hinge_loss_g(d_fake):
    return -d_fake.mean()
