"""
MixUp augmentation for smoother decision boundaries.

FIX-7 applied:
- Skip MixUp entirely if the batch contains IOL samples (minority class).
"""

import torch
import numpy as np


# FIX-7: IOL class index for minority class filtering
IOL_CLASS_IDX = 3


def should_skip_mixup(y: torch.Tensor) -> bool:
    """
    FIX-7: Skip MixUp if batch contains IOL class (minority class).

    Args:
        y: Target labels of shape (N,)

    Returns:
        True if batch contains IOL samples, False otherwise
    """
    return (y == IOL_CLASS_IDX).any().item()


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    Apply MixUp augmentation.

    Args:
        x: Input batch of shape (N, C, H, W)
        y: Target labels of shape (N,)
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        mixed_x: Mixed input batch
        y_a: Original labels
        y_b: Permuted labels
        lam: Mixing coefficient
    """
    batch_size = x.size(0)

    # Sample mixing coefficient from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Random permutation for pairing
    index = torch.randperm(batch_size, device=x.device)

    # Mix inputs
    mixed_x = lam * x + (1 - lam) * x[index]

    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for MixUp targets.

    Args:
        criterion: Loss function (CrossEntropyLoss or FocalLoss)
        pred: Model predictions (logits)
        y_a: First target batch (class indices)
        y_b: Second target batch (class indices)
        lam: Mixing coefficient (float or tensor)

    Returns:
        Mixed loss
    """
    if pred is None:
        raise ValueError("Model output (pred) cannot be None. Check model forward pass.")

    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    # Ensure lam is a Python float/int (not a tensor) so it doesn't affect gradients
    if isinstance(lam, torch.Tensor):
        lam = lam.item()
    return lam * loss_a + (1 - lam) * loss_b
