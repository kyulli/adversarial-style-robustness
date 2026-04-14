"""
Model utility functions.
"""

import torch
from pathlib import Path


def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, path: str, metadata: dict = None):
    """Save model weights and optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'state_dict': model.state_dict()}
    if metadata:
        payload['metadata'] = metadata
    torch.save(payload, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, path: str, device='cuda'):
    """Load model weights from a checkpoint."""
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and 'state_dict' in payload:
        model.load_state_dict(payload['state_dict'])
        return payload.get('metadata', {})
    else:
        # Legacy: plain state_dict
        model.load_state_dict(payload)
        return {}


def evaluate_accuracy(model, dataloader, device='cuda', top_k=(1, 5)):
    """
    Evaluate model accuracy on a dataloader.

    Returns:
        dict with top-k accuracies
    """
    model.eval()
    counts = {k: 0 for k in top_k}
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            batch_size = labels.size(0)
            total += batch_size

            for k in top_k:
                _, topk_preds = logits.topk(k, dim=1)
                correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
                counts[k] += correct.any(dim=1).sum().item()

    return {f'top{k}': counts[k] / total for k in top_k}
