"""
Fast Gradient Sign Method (FGSM) attack.
Goodfellow et al., 2014: "Explaining and Harnessing Adversarial Examples"
"""

import torch
import torch.nn as nn


class FGSM:
    """Fast Gradient Sign Method attack."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def generate(self, x, y, epsilon=0.03):
        """
        Generate adversarial examples using FGSM.

        Args:
            x: Input images [N, C, H, W], values in [0, 1]
            y: True labels [N]
            epsilon: Perturbation magnitude (L-inf bound)

        Returns:
            x_adv: Adversarial images
            perturbation: Applied perturbation
        """
        self.model.eval()
        x = x.clone().detach().to(self.device).requires_grad_(True)
        y = y.clone().detach().to(self.device)

        logits = self.model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        perturbation = epsilon * x.grad.sign()
        x_adv = (x + perturbation).clamp(0, 1).detach()
        return x_adv, perturbation.detach()
