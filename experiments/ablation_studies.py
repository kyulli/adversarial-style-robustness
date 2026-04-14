"""
Ablation Studies: compare defense mechanisms and attack parameters.

Runs evaluate_robustness across multiple model checkpoints and
produces a consolidated comparison table + plots.

Usage:
    python experiments/ablation_studies.py \
        --baseline_path ./results/baseline/best_model.pth \
        --adv_path ./results/adversarial_training/best_model.pth \
        --aug_path ./results/augmentation/best_model.pth \
        --data_dir ./data/wikiart \
        --output_dir ./results/ablations
"""

import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import get_dataloaders, WikiArtDataset
from models.baseline_model import StyleClassifier
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from attacks.utils import compute_perturbation_magnitude


def quick_eval(model, loader, attack_cls, epsilon, device, max_batches=30):
    """Quick evaluation of attack success rate at a given epsilon."""
    model.eval()
    attack = attack_cls(model, device)
    correct_adv, total = 0, 0

    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        x_adv, _ = attack.generate(images, labels, epsilon=epsilon)
        with torch.no_grad():
            preds = model(x_adv).argmax(1)
        correct_adv += (preds == labels).sum().item()
        total += labels.size(0)

    return 1 - correct_adv / total  # attack success rate


def robustness_curve(model_paths: dict, loader, attack_cls, epsilon_range,
                     device, max_batches=30):
    """
    Compute robustness curves for multiple models.

    Returns:
        DataFrame indexed by epsilon, columns = model names
    """
    results = {name: [] for name in model_paths}

    for epsilon in epsilon_range:
        print(f"  epsilon={epsilon:.3f}")
        for name, path in model_paths.items():
            model = StyleClassifier(backbone='resnet18', num_classes=27, pretrained=False)
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)
            asr = quick_eval(model, loader, attack_cls, epsilon, device, max_batches)
            results[name].append(asr)

    df = pd.DataFrame(results, index=epsilon_range)
    df.index.name = 'epsilon'
    return df


def per_style_asr(model, loader, attack_cls, epsilon, device, class_names, max_batches=50):
    """Compute per-style attack success rate."""
    model.eval()
    attack = attack_cls(model, device)
    per_style = {c: {'correct': 0, 'total': 0} for c in class_names}

    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        x_adv, _ = attack.generate(images, labels, epsilon=epsilon)
        with torch.no_grad():
            preds = model(x_adv).argmax(1)
        for j in range(images.size(0)):
            c = class_names[labels[j].item()]
            per_style[c]['total'] += 1
            per_style[c]['correct'] += int(preds[j] == labels[j])

    asr_dict = {}
    for c, v in per_style.items():
        if v['total'] > 0:
            asr_dict[c] = 1 - v['correct'] / v['total']
    return asr_dict


def plot_robustness_curves(df: pd.DataFrame, attack_name: str, output_dir: Path):
    """Plot robustness curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in df.columns:
        ax.plot(df.index, df[col], marker='o', label=col)
    ax.set_xlabel('Epsilon (perturbation magnitude)')
    ax.set_ylabel('Attack Success Rate')
    ax.set_title(f'Robustness vs. Perturbation ({attack_name.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / f'robustness_curve_{attack_name}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_per_style_heatmap(results_dict: dict, attack_name: str, epsilon: float, output_dir: Path):
    """Heatmap of per-style ASR across defense methods."""
    all_styles = sorted(set(s for r in results_dict.values() for s in r))
    df = pd.DataFrame(
        {name: [results_dict[name].get(s, float('nan')) for s in all_styles]
         for name in results_dict},
        index=all_styles
    )
    fig, ax = plt.subplots(figsize=(max(6, len(results_dict) * 2), max(8, len(all_styles) * 0.4)))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
                vmin=0, vmax=1, cbar_kws={'label': 'Attack Success Rate'})
    ax.set_title(f'Per-Style Attack Success Rate ({attack_name.upper()}, ε={epsilon})')
    ax.set_xlabel('Defense')
    ax.set_ylabel('Style')
    plt.tight_layout()
    path = output_dir / f'per_style_heatmap_{attack_name}_eps{epsilon}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Ablation studies')
    parser.add_argument('--baseline_path', type=str, required=True)
    parser.add_argument('--adv_path', type=str, default=None)
    parser.add_argument('--aug_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='./data/wikiart')
    parser.add_argument('--output_dir', type=str, default='./results/ablations')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_batches', type=int, default=30)
    parser.add_argument('--epsilon', type=float, default=0.03,
                        help='Epsilon used for per-style analysis')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model path dict
    model_paths = {'Baseline': args.baseline_path}
    if args.adv_path:
        model_paths['Adversarial Training'] = args.adv_path
    if args.aug_path:
        model_paths['Data Augmentation'] = args.aug_path

    # Data
    _, _, test_loader, num_classes = get_dataloaders(
        args.data_dir, batch_size=args.batch_size)
    test_dataset = WikiArtDataset(args.data_dir, split='test')
    class_names = test_dataset.classes

    epsilon_range = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1]

    for attack_name, attack_cls in [('fgsm', FGSM), ('pgd', PGD)]:
        print(f"\n=== {attack_name.upper()} Robustness Curves ===")
        df_curves = robustness_curve(model_paths, test_loader, attack_cls,
                                     epsilon_range, device, args.max_batches)
        df_curves.to_csv(output_dir / f'curves_{attack_name}.csv')
        plot_robustness_curves(df_curves, attack_name, output_dir)
        print(df_curves.to_string())

        print(f"\n=== {attack_name.upper()} Per-Style Analysis (ε={args.epsilon}) ===")
        per_style_results = {}
        for name, path in model_paths.items():
            model = StyleClassifier(backbone='resnet18', num_classes=27, pretrained=False)
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)
            per_style_results[name] = per_style_asr(
                model, test_loader, attack_cls, args.epsilon,
                device, class_names, args.max_batches)
        plot_per_style_heatmap(per_style_results, attack_name, args.epsilon, output_dir)

    print(f"\nAll ablation results saved to {output_dir}")


if __name__ == '__main__':
    main()
