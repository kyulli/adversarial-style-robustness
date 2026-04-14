"""
Adversarial attack implementations: FGSM and PGD
"""

import torch
import torch.nn as nn


class FGSM:
    """Fast Gradient Sign Method attack."""
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: PyTorch model
            device (str): 'cuda' or 'cpu'
        """
        self.model = model
        self.device = device
    
    def generate(self, x, y, epsilon=0.03):
        """
        Generate adversarial examples using FGSM.
        
        Args:
            x: Input images [N, C, H, W]
            y: Target labels [N]
            epsilon (float): Perturbation magnitude
        
        Returns:
            x_adv: Adversarial examples
            perturbation: Added perturbation
        """
        x = x.clone().detach().to(self.device).requires_grad_(True)
        y = y.clone().detach().to(self.device)
        
        # Forward pass
        logits = self.model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        
        # Backward pass
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()
        
        # Compute perturbation
        grad = x.grad.sign()
        perturbation = epsilon * grad
        
        # Apply perturbation
        x_adv = (x + perturbation).clamp(0, 1).detach()
        
        return x_adv, perturbation.detach()


class PGD:
    """Projected Gradient Descent attack."""
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: PyTorch model
            device (str): 'cuda' or 'cpu'
        """
        self.model = model
        self.device = device
    
    def generate(self, x, y, epsilon=0.03, step_size=0.001, num_steps=20, random_start=True):
        """
        Generate adversarial examples using PGD.
        
        Args:
            x: Input images [N, C, H, W]
            y: Target labels [N]
            epsilon (float): Maximum perturbation magnitude
            step_size (float): Step size for each iteration
            num_steps (int): Number of optimization steps
            random_start (bool): Start from random perturbation
        
        Returns:
            x_adv: Adversarial examples
            perturbation: Final perturbation
        """
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)
        
        # Initialize perturbation
        if random_start:
            delta = torch.rand_like(x) * 2 * epsilon - epsilon
        else:
            delta = torch.zeros_like(x)
        
        delta.requires_grad = True
        
        # Iterative perturbation
        for step in range(num_steps):
            # Forward pass
            logits = self.model(x + delta)
            loss = nn.CrossEntropyLoss()(logits, y)
            
            # Backward pass
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
            
            # Update delta
            with torch.no_grad():
                grad = delta.grad.sign()
                delta.data = delta.data + step_size * grad
                
                # Projection: keep perturbation within epsilon ball
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                
                # Ensure image stays in valid range
                x_adv_temp = torch.clamp(x + delta, 0, 1)
                delta.data = x_adv_temp - x
            
            delta.requires_grad = True
        
        x_adv = torch.clamp(x + delta, 0, 1).detach()
        perturbation = delta.detach()
        
        return x_adv, perturbation


class AdversarialAttacker:
    """Unified interface for generating adversarial examples."""
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: PyTorch model
            device (str): 'cuda' or 'cpu'
        """
        self.model = model
        self.device = device
        self.fgsm = FGSM(model, device)
        self.pgd = PGD(model, device)
    
    def attack(self, x, y, method='fgsm', epsilon=0.03, **kwargs):
        """
        Generate adversarial examples.
        
        Args:
            x: Input images
            y: Target labels
            method (str): 'fgsm' or 'pgd'
            epsilon (float): Perturbation magnitude
            **kwargs: Additional arguments for specific attacks
        
        Returns:
            x_adv: Adversarial examples
            perturbation: Added perturbation
        """
        if method.lower() == 'fgsm':
            return self.fgsm.generate(x, y, epsilon=epsilon)
        elif method.lower() == 'pgd':
            step_size = kwargs.get('step_size', 0.001)
            num_steps = kwargs.get('num_steps', 20)
            return self.pgd.generate(x, y, epsilon=epsilon, step_size=step_size, num_steps=num_steps)
        else:
            raise ValueError(f"Unknown attack method: {method}")


def compute_perturbation_magnitude(perturbation, metric='linf'):
    """
    Compute perturbation magnitude.
    
    Args:
        perturbation: Perturbation tensor [N, C, H, W]
        metric (str): 'linf' (L-infinity), 'l2' (L2), or 'l1' (L1)
    
    Returns:
        magnitude: Perturbation magnitude per sample
    """
    batch_size = perturbation.shape[0]
    
    if metric == 'linf':
        # L-infinity norm (maximum absolute value)
        magnitude = perturbation.abs().view(batch_size, -1).max(dim=1)[0]
    elif metric == 'l2':
        # L2 norm
        magnitude = (perturbation ** 2).view(batch_size, -1).sum(dim=1).sqrt()
    elif metric == 'l1':
        # L1 norm
        magnitude = perturbation.abs().view(batch_size, -1).sum(dim=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return magnitude


if __name__ == "__main__":
    # Test attacks
    from baseline_model import StyleClassifier
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = StyleClassifier(backbone='resnet18', num_classes=27)
    model = model.to(device)
    model.eval()
    
    # Create dummy data
    x = torch.randn(4, 3, 224, 224).clamp(0, 1).to(device)
    y = torch.randint(0, 27, (4,)).to(device)
    
    # Test FGSM
    print("Testing FGSM...")
    attacker = AdversarialAttacker(model, device)
    x_adv_fgsm, pert_fgsm = attacker.attack(x, y, method='fgsm', epsilon=0.03)
    print(f"FGSM perturbation shape: {pert_fgsm.shape}")
    print(f"FGSM perturbation magnitude (L∞): {compute_perturbation_magnitude(pert_fgsm, 'linf')}")
    
    # Test PGD
    print("\nTesting PGD...")
    x_adv_pgd, pert_pgd = attacker.attack(x, y, method='pgd', epsilon=0.03, num_steps=20)
    print(f"PGD perturbation shape: {pert_pgd.shape}")
    print(f"PGD perturbation magnitude (L∞): {compute_perturbation_magnitude(pert_pgd, 'linf')}")
