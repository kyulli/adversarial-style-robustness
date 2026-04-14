"""
Data Augmentation defense.
Trains a standard model with aggressive augmentations to improve robustness.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import json


def get_augmented_transform(image_size=224,
                             random_crop=True,
                             random_rotation=True,
                             color_jitter=True,
                             gaussian_blur=False):
    """
    Build a torchvision transform pipeline with augmentations.

    Returns:
        transform: Composed transform
    """
    ops = [transforms.Resize((image_size + 32, image_size + 32))]

    if random_crop:
        ops.append(transforms.RandomCrop(image_size))
    else:
        ops.append(transforms.CenterCrop(image_size))

    ops.append(transforms.RandomHorizontalFlip())

    if random_rotation:
        ops.append(transforms.RandomRotation(15))

    if color_jitter:
        ops.append(transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))

    if gaussian_blur:
        ops.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


class AugmentationTrainer:
    """Trains with heavy data augmentation as a robustness defense."""

    def __init__(self, model, device='cuda', output_dir='./results/augmentation'):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def _run_epoch(self, loader, optimizer, criterion, training=True):
        self.model.train() if training else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, labels in tqdm(loader, desc='Train' if training else 'Val', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                if training:
                    optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                if training:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += images.size(0)
        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs=50, lr=0.001,
            weight_decay=1e-4, early_stopping_patience=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        best_val_acc, patience_counter = 0.0, 0

        print("Augmentation-based training...")
        for epoch in range(epochs):
            train_loss, train_acc = self._run_epoch(train_loader, optimizer, criterion, True)
            val_loss, val_acc = self._run_epoch(val_loader, optimizer, criterion, False)
            scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs}  "
                  f"Train loss={train_loss:.4f} acc={train_acc:.4f}  "
                  f"Val loss={val_loss:.4f} acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.output_dir / 'best_model.pth')
                print("  ✓ Best model saved")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        return self.history
