"""
Wrapper for defended models (adversarially trained or preprocessing-defended).
Provides a consistent interface for evaluation.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baseline_model import StyleClassifier
from defenses.preprocessing import PreprocessingDefense


def load_model(checkpoint_path: str, backbone='resnet18', num_classes=27, device='cuda'):
    """Load a StyleClassifier from a saved checkpoint."""
    model = StyleClassifier(backbone=backbone, num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def load_preprocessing_defended_model(checkpoint_path: str, device='cuda',
                                       gaussian_sigma=1.0, quantization_bits=8,
                                       use_gaussian=True, use_quantization=False):
    """Load a model wrapped with preprocessing defense."""
    model = load_model(checkpoint_path, device=device)
    defended = PreprocessingDefense(
        model, device=device,
        use_gaussian=use_gaussian, gaussian_sigma=gaussian_sigma,
        use_quantization=use_quantization, quantization_bits=quantization_bits
    )
    return defended
