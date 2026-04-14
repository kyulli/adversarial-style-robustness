"""
Projected Gradient Descent (PGD) attack.
Madry et al., 2018: "Towards Deep Learning Models Resistant to Adversarial Attacks"
"""

import torch
import torch.nn as nn


class PGD:
    """Projected Gradient Descent (PGD) attack."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def generate(self, x, y, epsilon=0.03, step_size=0.001, num_steps=20, random_start=True):
        """
        Generate adversarial examples using PGD.

        Args:
            x: Input images [N, C, H, W], values in [0, 1]
            y: True labels [N]
            epsilon: L-inf perturbation bound
            step_size: Per-step perturbation size (alpha)
            num_steps: Number of PGD iterations
            random_start: Initialize from a random perturbation within the epsilon ball

        Returns:
            x_adv: Adversarial images
            perturbation: Final perturbation (x_adv - x)
        """
        self.model.eval()
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        if random_start:
            delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        else:
            delta = torch.zeros_like(x)
        delta = delta.clamp(-epsilon, epsilon)
        delta.requires_grad_(True)

        for _ in range(num_steps):
            logits = self.model(x + delta)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()

            with torch.no_grad():
                delta.data = delta.data + step_size * delta.grad.sign()
                delta.data = delta.data.clamp(-epsilon, epsilon)
                # Keep perturbed image in valid pixel range
                delta.data = (x + delta.data).clamp(0, 1) - x

            delta.grad.zero_()

        x_adv = (x + delta).clamp(0, 1).detach()
        return x_adv, delta.detach()
