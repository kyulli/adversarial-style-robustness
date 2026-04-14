"""
Preprocessing utilities for WikiArt dataset.
Handles image validation, resizing, and dataset statistics.
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json


def validate_images(data_dir: str, fix_corrupted: bool = False):
    """
    Check all images in the dataset for corruption or unreadable files.

    Args:
        data_dir: Path to WikiArt directory
        fix_corrupted: If True, delete corrupted images
    """
    data_path = Path(data_dir)
    corrupted = []
    valid = 0

    for img_path in tqdm(list(data_path.rglob("*.jpg")) + list(data_path.rglob("*.png")),
                         desc="Validating images"):
        try:
            with Image.open(img_path) as img:
                img.verify()
            valid += 1
        except Exception as e:
            corrupted.append(str(img_path))
            if fix_corrupted:
                img_path.unlink()

    print(f"\nValidation complete: {valid} valid, {len(corrupted)} corrupted.")
    if corrupted:
        print("Corrupted files:")
        for f in corrupted:
            print(f"  {f}")
    return corrupted


def compute_dataset_stats(data_dir: str, sample_size: int = 1000, save_path: str = None):
    """
    Compute mean and std of the dataset for normalization.

    Args:
        data_dir: Path to WikiArt directory
        sample_size: Number of images to sample for statistics
        save_path: If provided, save stats as JSON

    Returns:
        dict with 'mean' and 'std' (each a list of 3 floats for RGB)
    """
    from torchvision import transforms
    import torch

    data_path = Path(data_dir)
    all_images = list(data_path.rglob("*.jpg")) + list(data_path.rglob("*.png"))

    # Sample subset for efficiency
    if len(all_images) > sample_size:
        indices = np.random.choice(len(all_images), sample_size, replace=False)
        sampled = [all_images[i] for i in indices]
    else:
        sampled = all_images

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    pixel_sum = torch.zeros(3)
    pixel_sq_sum = torch.zeros(3)
    count = 0

    for img_path in tqdm(sampled, desc="Computing stats"):
        try:
            with Image.open(img_path).convert("RGB") as img:
                tensor = transform(img)  # [3, H, W]
                pixel_sum += tensor.sum(dim=[1, 2])
                pixel_sq_sum += (tensor ** 2).sum(dim=[1, 2])
                count += tensor.shape[1] * tensor.shape[2]
        except Exception:
            continue

    mean = (pixel_sum / count).tolist()
    std = ((pixel_sq_sum / count - torch.tensor(mean) ** 2) ** 0.5).tolist()

    stats = {"mean": mean, "std": std, "num_samples": len(sampled)}
    print(f"\nDataset stats (from {len(sampled)} images):")
    print(f"  Mean: {[f'{m:.4f}' for m in mean]}")
    print(f"  Std:  {[f'{s:.4f}' for s in std]}")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved to {save_path}")

    return stats


def print_class_distribution(data_dir: str):
    """Print number of images per style class."""
    data_path = Path(data_dir)
    style_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    print(f"\nClass distribution in {data_dir}:")
    print(f"{'Style':<40} {'Images':>8}")
    print("-" * 50)
    total = 0
    for d in style_dirs:
        count = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
        total += count
        print(f"{d.name:<40} {count:>8}")
    print("-" * 50)
    print(f"{'TOTAL':<40} {total:>8}")
    print(f"Number of classes: {len(style_dirs)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess WikiArt dataset")
    parser.add_argument("--data_dir", type=str, default="./data/wikiart")
    parser.add_argument("--validate", action="store_true", help="Validate images")
    parser.add_argument("--fix", action="store_true", help="Delete corrupted images")
    parser.add_argument("--stats", action="store_true", help="Compute dataset stats")
    parser.add_argument("--distribution", action="store_true", help="Print class distribution")
    args = parser.parse_args()

    if args.validate:
        validate_images(args.data_dir, fix_corrupted=args.fix)
    if args.stats:
        compute_dataset_stats(args.data_dir, save_path="./data/dataset_stats.json")
    if args.distribution:
        print_class_distribution(args.data_dir)
