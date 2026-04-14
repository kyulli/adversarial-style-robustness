"""
Input Preprocessing defense.
Applies smoothing/denoising to inputs at inference time to remove adversarial perturbations.
"""

import torch
import torch.nn.functional as F
import numpy as np


def gaussian_blur(x: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5) -> torch.Tensor:
    """
    Apply Gaussian blur to a batch of images.

    Args:
        x: [N, C, H, W] float tensor in [0, 1]
        sigma: Gaussian sigma
        kernel_size: Must be odd

    Returns:
        Blurred images, same shape
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Build 2D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel_2d = g.outer(g)  # [k, k]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(x.shape[1], 1, 1, 1)
    kernel_2d = kernel_2d.to(x.device)

    padding = kernel_size // 2
    blurred = F.conv2d(x, kernel_2d, padding=padding, groups=x.shape[1])
    return blurred.clamp(0, 1)


def quantize(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Reduce bit depth of image (input quantization defense).

    Args:
        x: [N, C, H, W] float tensor in [0, 1]
        bits: Number of quantization levels per channel

    Returns:
        Quantized images
    """
    levels = 2 ** bits
    return (x * levels).floor() / levels


class PreprocessingDefense:
    """
    Wraps a model with input preprocessing to defend against adversarial examples.
    """

    def __init__(self, model, device='cuda',
                 use_gaussian=True, gaussian_sigma=1.0,
                 use_quantization=False, quantization_bits=8):
        self.model = model
        self.device = device
        self.use_gaussian = use_gaussian
        self.gaussian_sigma = gaussian_sigma
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing to a batch of images."""
        if self.use_quantization:
            x = quantize(x, self.quantization_bits)
        if self.use_gaussian:
            x = gaussian_blur(x, self.gaussian_sigma)
        return x

    def predict(self, x: torch.Tensor):
        """Preprocess then classify."""
        self.model.eval()
        with torch.no_grad():
            x_clean = self.preprocess(x.to(self.device))
            logits = self.model(x_clean)
            return logits.argmax(dim=1), logits

    def __call__(self, x: torch.Tensor):
        """Forward pass (used during evaluation)."""
        x_clean = self.preprocess(x)
        return self.model(x_clean)
