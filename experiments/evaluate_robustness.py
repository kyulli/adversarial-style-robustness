"""
Robustness evaluation script.
Generates adversarial examples, measures attack success rates,
and saves quantitative metrics per style.

Usage:
    python experiments/evaluate_robustness.py \
        --model_path ./results/baseline/best_model.pth \
        --data_dir ./data/wikiart \
        --attack fgsm pgd \
        --output_dir ./results/evaluation
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import get_dataloaders
from models.baseline_model import StyleClassifier
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from attacks.utils import compute_perturbation_magnitude, attack_success_rate, psnr


def evaluate_attack(model, loader, attack, epsilon_list, device, class_names, max_batches=None):
    """
    Run an attack across multiple epsilon values on the test set.

    Returns:
        DataFrame with columns: epsilon, overall_accuracy, attack_success_rate,
                                 mean_l2, mean_linf, psnr, and per-style metrics
    """
    rows = []

    for epsilon in epsilon_list:
        print(f"  epsilon={epsilon:.3f}")
        correct_adv = 0
        correct_clean = 0
        total = 0
        l2_list, linf_list, psnr_list = [], [], []
        per_style_correct = {c: 0 for c in class_names}
        per_style_adv_correct = {c: 0 for c in class_names}
        per_style_total = {c: 0 for c in class_names}

        model.eval()
        for batch_idx, (images, labels) in enumerate(tqdm(loader, leave=False)):
            if max_batches and batch_idx >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)

            # Clean predictions
            with torch.no_grad():
                clean_logits = model(images)
                clean_preds = clean_logits.argmax(1)

            # Adversarial examples
            if isinstance(attack, FGSM):
                x_adv, pert = attack.generate(images, labels, epsilon=epsilon)
            elif isinstance(attack, PGD):
                x_adv, pert = attack.generate(images, labels, epsilon=epsilon)
            else:
                raise ValueError("Unknown attack type")

            with torch.no_grad():
                adv_logits = model(x_adv)
                adv_preds = adv_logits.argmax(1)

            # Aggregate
            correct_clean += (clean_preds == labels).sum().item()
            correct_adv += (adv_preds == labels).sum().item()
            total += images.size(0)

            l2_list.extend(compute_perturbation_magnitude(pert, 'l2').cpu().tolist())
            linf_list.extend(compute_perturbation_magnitude(pert, 'linf').cpu().tolist())

            # PSNR on CPU for memory
            psnr_list.extend([
                psnr(images[i:i+1].cpu(), x_adv[i:i+1].cpu())
                for i in range(images.size(0))
            ])

            # Per-style breakdown
            for i in range(images.size(0)):
                c = class_names[labels[i].item()]
                per_style_total[c] += 1
                per_style_correct[c] += int(clean_preds[i] == labels[i])
                per_style_adv_correct[c] += int(adv_preds[i] == labels[i])

        row = {
            'epsilon': epsilon,
            'clean_accuracy': correct_clean / total,
            'adv_accuracy': correct_adv / total,
            'attack_success_rate': 1 - correct_adv / total,
            'mean_l2': np.mean(l2_list),
            'mean_linf': np.mean(linf_list),
            'mean_psnr': np.mean([p for p in psnr_list if np.isfinite(p)]),
        }
        # Per-style attack success rate
        for c in class_names:
            if per_style_total[c] > 0:
                row[f'asr_{c}'] = 1 - per_style_adv_correct[c] / per_style_total[c]
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model robustness')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data/wikiart')
    parser.add_argument('--attack', nargs='+', default=['fgsm', 'pgd'],
                        choices=['fgsm', 'pgd'])
    parser.add_argument('--epsilon', nargs='+', type=float,
                        default=[0.01, 0.02, 0.03, 0.05, 0.1])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='./results/evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit batches per epsilon for faster debug runs')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    _, _, test_loader, num_classes = get_dataloaders(
        args.data_dir, batch_size=args.batch_size)

    # Get class names from dataset
    from data_loader import WikiArtDataset
    test_dataset = WikiArtDataset(args.data_dir, split='test')
    class_names = test_dataset.classes

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = StyleClassifier(backbone='resnet18', num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    all_results = {}

    for atk_name in args.attack:
        print(f"\n=== Attack: {atk_name.upper()} ===")
        if atk_name == 'fgsm':
            attack = FGSM(model, device)
        elif atk_name == 'pgd':
            attack = PGD(model, device)

        df = evaluate_attack(model, test_loader, attack, args.epsilon,
                              device, class_names, max_batches=args.max_batches)

        csv_path = output_dir / f'{atk_name}_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        print(df[['epsilon', 'clean_accuracy', 'adv_accuracy', 'attack_success_rate',
                   'mean_linf', 'mean_psnr']].to_string(index=False))
        all_results[atk_name] = df.to_dict(orient='records')

    # Combined summary
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
