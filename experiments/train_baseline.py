"""
Training script for baseline style classification model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Import project modules
from data_loader import get_dataloaders
from baseline_model import StyleClassifier, count_parameters


class Trainer:
    """Trainer class for model training."""
    
    def __init__(self, model, device='cuda', output_dir='./results/baseline'):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = self.model(images)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            _, predictions = torch.max(logits, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = total_loss / total
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, dataloader, criterion):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss = criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item() * images.size(0)
                _, predictions = torch.max(logits, 1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        epoch_loss = total_loss / total
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy
    
    def fit(self, train_loader, val_loader, epochs=50, learning_rate=0.001,
            weight_decay=1e-4, scheduler='cosine', early_stopping_patience=10):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs (int): Number of epochs
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
            scheduler (str): Learning rate scheduler type
            early_stopping_patience (int): Patience for early stopping
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if scheduler == 'cosine':
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            lr_scheduler = None
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_accuracy = self.train_epoch(train_loader, optimizer, criterion)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            
            # Validate
            val_loss, val_accuracy = self.validate(val_loader, criterion)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Update learning rate
            if lr_scheduler:
                lr_scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                self.save_model('best_model.pth')
                print("✓ Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        self.save_history()
        return self.history
    
    def save_model(self, filename='model.pth'):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, filename='best_model.pth'):
        """Load model checkpoint."""
        path = self.output_dir / filename
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
    
    def save_history(self):
        """Save training history."""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"History saved to {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Train baseline style classification model')
    parser.add_argument('--data_dir', type=str, default='./data/wikiart',
                        help='Path to WikiArt dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--architecture', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'efficientnet_b0'],
                        help='Model architecture')
    parser.add_argument('--output_dir', type=str, default='./results/baseline',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print(f"Creating {args.architecture} model...")
    model = StyleClassifier(backbone=args.architecture, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Train
    trainer = Trainer(model, device=device, output_dir=args.output_dir)
    trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    trainer.load_model('best_model.pth')
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy = trainer.validate(test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")


if __name__ == '__main__':
    main()
