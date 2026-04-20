"""
Train defended models (adversarial training or augmentation-based).

Usage:
    # Adversarial training
    python experiments/train_defended.py \
        --defense adversarial_training \
        --data_dir ./data/wikiart \
        --epochs 50 --output_dir ./results/adversarial_training

    # Data augmentation defense
    python experiments/train_defended.py \
        --defense data_augmentation \
        --data_dir ./data/wikiart \
        --epochs 50 --output_dir ./results/augmentation
"""

import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import get_dataloaders
from defenses.data_augmentation import get_augmented_transform
from models.baseline_model import StyleClassifier, count_parameters


def main():
    parser = argparse.ArgumentParser(description='Train defended model')
    parser.add_argument('--defense', type=str, required=True,
                        choices=['adversarial_training', 'data_augmentation'])
    parser.add_argument('--data_dir', type=str, default='./data/wikiart')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--architecture', type=str, default='resnet18')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    # Adversarial training specific
    parser.add_argument('--attack_method', type=str, default='pgd',
                        choices=['fgsm', 'pgd'])
    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--step_size', type=float, default=0.001)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Fraction of adversarial examples per batch')
    args = parser.parse_args()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    output_dir = args.output_dir or f'./results/{args.defense}'

    # Load data
    if args.defense == 'data_augmentation':
        # Use augmented transform for training loader
        aug_transform = get_augmented_transform()
        train_loader, val_loader, _, num_classes = get_dataloaders(
            args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
        # Re-build train dataset with augmented transform
        from data_loader import WikiArtDataset
        from torch.utils.data import DataLoader
        train_dataset = WikiArtDataset(args.data_dir, transform=aug_transform, split='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
    else:
        train_loader, val_loader, _, num_classes = get_dataloaders(
            args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"Number of classes: {num_classes}")

    # Create model
    model = StyleClassifier(backbone=args.architecture,
                             num_classes=num_classes, pretrained=True)
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    if args.defense == 'adversarial_training':
        from defenses.adversarial_training import AdversarialTrainer
        trainer = AdversarialTrainer(model, device=device, output_dir=output_dir)
        trainer.fit(
            train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            attack_method=args.attack_method,
            epsilon=args.epsilon, step_size=args.step_size, num_steps=args.num_steps,
            alpha=args.alpha
        )

    elif args.defense == 'data_augmentation':
        from defenses.data_augmentation import AugmentationTrainer
        trainer = AugmentationTrainer(model, device=device, output_dir=output_dir)
        trainer.fit(
            train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay
        )

    print(f"\nDone. Model saved to {output_dir}/best_model.pth")


if __name__ == '__main__':
    main()
