"""
Adversarial Training defense.
Madry et al., 2018: mix adversarial examples into each training batch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path
import json
import sys
import os

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacks.pgd import PGD
from attacks.fgsm import FGSM


class AdversarialTrainer:
    """
    Trains a model with adversarial examples mixed into each batch.

    alpha: fraction of the batch replaced with adversarial examples.
    """

    def __init__(self, model, device='cuda', output_dir='./results/adversarial_training'):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def _make_attacker(self, method, epsilon, step_size, num_steps):
        if method == 'pgd':
            return PGD(self.model, self.device), dict(
                epsilon=epsilon, step_size=step_size, num_steps=num_steps)
        elif method == 'fgsm':
            return FGSM(self.model, self.device), dict(epsilon=epsilon)
        else:
            raise ValueError(f"Unknown attack method: {method}")

    def train_epoch(self, loader, optimizer, criterion,
                    attack_method='pgd', epsilon=0.03,
                    step_size=0.001, num_steps=10, alpha=0.5):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        attacker, atk_kwargs = self._make_attacker(attack_method, epsilon, step_size, num_steps)

        for images, labels in tqdm(loader, desc='Adv Train', leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            B = images.size(0)

            # Generate adversarial examples for alpha fraction of the batch
            n_adv = int(B * alpha)
            if n_adv > 0:
                self.model.eval()
                x_adv, _ = attacker.generate(images[:n_adv], labels[:n_adv], **atk_kwargs)
                self.model.train()
                mixed = torch.cat([x_adv.detach(), images[n_adv:]], dim=0)
            else:
                mixed = images

            optimizer.zero_grad()
            logits = self.model(mixed)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            correct += (logits.argmax(1) == labels).sum().item()
            total += B

        return total_loss / total, correct / total

    def validate(self, loader, criterion):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(loader, desc='Val', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += images.size(0)
        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=1e-4,
            attack_method='pgd', epsilon=0.03, step_size=0.001, num_steps=10, alpha=0.5,
            early_stopping_patience=10):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        best_val_acc, patience_counter = 0.0, 0

        print(f"Adversarial training: method={attack_method}, epsilon={epsilon}, alpha={alpha}")
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion,
                attack_method, epsilon, step_size, num_steps, alpha)
            val_loss, val_acc = self.validate(val_loader, criterion)
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
