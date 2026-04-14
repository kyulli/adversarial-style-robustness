"""
Visualization script: adversarial examples, saliency maps, perturbation magnitudes.

Usage:
    python experiments/generate_visualizations.py \
        --model_path ./results/baseline/best_model.pth \
        --data_dir ./data/wikiart \
        --output_dir ./results/visualizations
"""

import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import get_dataloaders, WikiArtDataset
from models.baseline_model import StyleClassifier
from attacks.fgsm import FGSM
from attacks.pgd import PGD


def denormalize(tensor):
    """Undo ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def visualize_adversarial_examples(model, images, labels, class_names, attack, epsilon,
                                    attack_name, output_dir, n_show=4):
    """Show clean vs adversarial image pairs with predictions."""
    model.eval()
    x_adv, pert = attack.generate(images, labels, epsilon=epsilon)

    with torch.no_grad():
        clean_preds = model(images).argmax(1).cpu()
        adv_preds = model(x_adv).argmax(1).cpu()

    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
    if n_show == 1:
        axes = axes[None, :]

    for i in range(min(n_show, images.size(0))):
        clean_img = denormalize(images[i]).permute(1, 2, 0).numpy()
        adv_img = denormalize(x_adv[i]).permute(1, 2, 0).numpy()
        pert_img = (pert[i].cpu().permute(1, 2, 0).numpy() * 10 + 0.5).clip(0, 1)

        true_label = class_names[labels[i].item()]
        clean_pred = class_names[clean_preds[i].item()]
        adv_pred = class_names[adv_preds[i].item()]

        axes[i, 0].imshow(clean_img)
        axes[i, 0].set_title(f'Clean\nTrue: {true_label}\nPred: {clean_pred}', fontsize=8)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(adv_img)
        color = 'red' if adv_pred != true_label else 'green'
        axes[i, 1].set_title(f'Adversarial (ε={epsilon})\nPred: {adv_pred}',
                              fontsize=8, color=color)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pert_img)
        axes[i, 2].set_title(f'Perturbation (×10)', fontsize=8)
        axes[i, 2].axis('off')

    plt.suptitle(f'{attack_name.upper()} Adversarial Examples (ε={epsilon})', fontsize=12)
    plt.tight_layout()
    path = output_dir / f'adv_examples_{attack_name}_eps{epsilon}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def visualize_saliency_map(model, images, labels, class_names, output_dir, n_show=4):
    """Visualize vanilla gradient saliency maps."""
    model.eval()
    images_req = images.clone().requires_grad_(True)
    logits = model(images_req)
    # Sum of correct-class logits
    scores = logits[range(images.size(0)), labels].sum()
    scores.backward()

    saliency = images_req.grad.abs().max(dim=1)[0].cpu()  # [N, H, W]

    fig, axes = plt.subplots(n_show, 2, figsize=(8, 4 * n_show))
    if n_show == 1:
        axes = axes[None, :]

    for i in range(min(n_show, images.size(0))):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        sal = saliency[i].numpy()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'{class_names[labels[i].item()]}', fontsize=9)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(sal, cmap='hot')
        axes[i, 1].set_title('Saliency Map', fontsize=9)
        axes[i, 1].axis('off')

    plt.suptitle('Gradient Saliency Maps', fontsize=12)
    plt.tight_layout()
    path = output_dir / 'saliency_maps.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data/wikiart')
    parser.add_argument('--output_dir', type=str, default='./results/visualizations')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_show', type=int, default=4)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    (output_dir / 'adversarial_examples').mkdir(parents=True, exist_ok=True)
    (output_dir / 'saliency_maps').mkdir(parents=True, exist_ok=True)

    # Data
    _, _, test_loader, num_classes = get_dataloaders(
        args.data_dir, batch_size=args.batch_size)
    test_dataset = WikiArtDataset(args.data_dir, split='test')
    class_names = test_dataset.classes

    # Model
    model = StyleClassifier(backbone='resnet18', num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    # Get one sample batch
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Adversarial examples for multiple epsilons
    for attack_cls, name in [(FGSM, 'fgsm'), (PGD, 'pgd')]:
        attack = attack_cls(model, device)
        for eps in [0.01, 0.03, 0.1]:
            visualize_adversarial_examples(
                model, images, labels, class_names, attack, eps, name,
                output_dir / 'adversarial_examples', n_show=args.n_show)

    # Saliency maps
    visualize_saliency_map(model, images, labels, class_names,
                            output_dir / 'saliency_maps', n_show=args.n_show)

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()
