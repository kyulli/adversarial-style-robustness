"""
Data loading and preprocessing module for WikiArt dataset.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


class WikiArtDataset(Dataset):
    """Custom dataset class for WikiArt images and style labels."""
    
    def __init__(self, root_dir, transform=None, split='train', train_split=0.7, val_split=0.15, seed=42):
        """
        Args:
            root_dir (str): Path to WikiArt directory containing style subdirectories
            transform: Optional transforms to be applied on images
            split (str): 'train', 'val', or 'test'
            train_split (float): Proportion of data for training
            val_split (float): Proportion of data for validation
            seed (int): Random seed for reproducibility
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.seed = seed
        
        # Get all style classes
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Collect all images
        self.images = []
        self.labels = []
        
        np.random.seed(seed)
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.root_dir / class_name
            image_files = sorted([f for f in class_dir.glob('*.jpg') if f.is_file()])
            
            # Shuffle and split
            np.random.shuffle(image_files)
            n = len(image_files)
            train_end = int(n * train_split)
            val_end = int(n * (train_split + val_split))
            
            if split == 'train':
                files_to_use = image_files[:train_end]
            elif split == 'val':
                files_to_use = image_files[train_end:val_end]
            else:  # test
                files_to_use = image_files[val_end:]
            
            self.images.extend(files_to_use)
            self.labels.extend([class_idx] * len(files_to_use))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx):
        """Get class name from index."""
        return self.classes[idx]


def get_dataloaders(data_dir, batch_size=32, num_workers=4, image_size=224,
                    train_split=0.7, val_split=0.15, seed=42):
    """
    Create train, val, and test dataloaders.
    
    Args:
        data_dir (str): Path to WikiArt directory
        batch_size (int): Batch size
        num_workers (int): Number of data loading workers
        image_size (int): Size to resize images to
        train_split (float): Proportion for training
        val_split (float): Proportion for validation
        seed (int): Random seed
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    train_dataset = WikiArtDataset(
        data_dir, transform=train_transform, split='train',
        train_split=train_split, val_split=val_split, seed=seed
    )
    
    val_dataset = WikiArtDataset(
        data_dir, transform=test_transform, split='val',
        train_split=train_split, val_split=val_split, seed=seed
    )
    
    test_dataset = WikiArtDataset(
        data_dir, transform=test_transform, split='test',
        train_split=train_split, val_split=val_split, seed=seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    num_classes = len(train_dataset.classes)
    
    return train_loader, val_loader, test_loader, num_classes


if __name__ == "__main__":
    # Example usage
    data_dir = "./data/wikiart"
    
    if os.path.exists(data_dir):
        train_loader, val_loader, test_loader, num_classes = get_dataloaders(
            data_dir, batch_size=32, num_workers=4
        )
        
        print(f"Number of classes: {num_classes}")
        print(f"Train loader length: {len(train_loader)}")
        print(f"Val loader length: {len(val_loader)}")
        print(f"Test loader length: {len(test_loader)}")
        
        # Sample batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
    else:
        print(f"Data directory {data_dir} not found. Please download WikiArt dataset first.")
