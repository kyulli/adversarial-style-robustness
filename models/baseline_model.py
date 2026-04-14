"""
Baseline model architecture for artistic style classification.
Uses pretrained ResNet-18 fine-tuned on WikiArt.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from torchvision.models import efficientnet_b0


def get_baseline_model(model_name='resnet18', num_classes=27, pretrained=True, device='cuda'):
    """
    Create a baseline model for style classification.
    
    Args:
        model_name (str): Model architecture ('resnet18', 'resnet50', 'efficientnet_b0')
        num_classes (int): Number of artistic styles (default 27 for WikiArt)
        pretrained (bool): Use pretrained ImageNet weights
        device (str): 'cuda' or 'cpu'
    
    Returns:
        model: PyTorch model ready for training
    """
    
    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained)
        # Replace final FC layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    elif model_name == 'resnet50':
        model = resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    model = model.to(device)
    return model


class StyleClassifier(nn.Module):
    """
    Wrapper class for style classification with utility methods.
    """
    
    def __init__(self, backbone='resnet18', num_classes=27, pretrained=True):
        super().__init__()
        self.backbone = get_baseline_model(backbone, num_classes, pretrained)
        self.num_classes = num_classes
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)
    
    def predict(self, x):
        """Get predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs
    
    def get_intermediate_features(self, x, layer='avgpool'):
        """
        Extract intermediate features from specified layer.
        Useful for visualization and analysis.
        
        Args:
            x: Input tensor
            layer (str): Layer name ('avgpool', 'layer4', etc.)
        
        Returns:
            features: Intermediate feature tensor
        """
        if isinstance(self.backbone, type(resnet18())):
            # For ResNet
            if layer == 'avgpool':
                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)
                x = self.backbone.avgpool(x)
                return x
        
        return None
    
    def freeze_backbone(self, freeze=True):
        """Freeze/unfreeze backbone parameters (useful for transfer learning)."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def unfreeze_last_layer(self):
        """Unfreeze the final classification layer."""
        for param in self.backbone.fc.parameters():
            param.requires_grad = True


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = StyleClassifier(backbone='resnet18', num_classes=27, pretrained=True)
    model = model.to(device)
    
    print(f"Model: {model}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    preds, probs = model.predict(dummy_input)
    print(f"Predictions shape: {preds.shape}")
    print(f"Probabilities shape: {probs.shape}")
