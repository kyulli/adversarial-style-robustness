"""
Utility functions for adversarial attacks.
"""

import torch
import numpy as np


def compute_perturbation_magnitude(perturbation, metric='linf'):
    """
    Compute per-sample perturbation magnitude.

    Args:
        perturbation: [N, C, H, W] tensor
        metric: 'linf', 'l2', or 'l1'

    Returns:
        magnitudes: [N] tensor
    """
    B = perturbation.shape[0]
    flat = perturbation.abs().view(B, -1)

    if metric == 'linf':
        return flat.max(dim=1)[0]
    elif metric == 'l2':
        return (perturbation ** 2).view(B, -1).sum(dim=1).sqrt()
    elif metric == 'l1':
        return flat.sum(dim=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def attack_success_rate(model, x_adv, y_true, device='cuda'):
    """
    Compute the fraction of adversarial examples that fool the model.

    Args:
        model: Classifier
        x_adv: Adversarial images [N, C, H, W]
        y_true: True labels [N]
        device: torch device

    Returns:
        success_rate: float in [0, 1]
    """
    model.eval()
    with torch.no_grad():
        logits = model(x_adv.to(device))
        preds = logits.argmax(dim=1)
    success = (preds != y_true.to(device)).float().mean().item()
    return success


def mean_squared_error(x_orig, x_adv):
    """MSE between clean and adversarial images."""
    return ((x_orig - x_adv) ** 2).mean().item()


def psnr(x_orig, x_adv, max_val=1.0):
    """Peak Signal-to-Noise Ratio. Higher = more imperceptible."""
    mse = mean_squared_error(x_orig, x_adv)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))
