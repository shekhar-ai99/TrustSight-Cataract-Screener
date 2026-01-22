"""Focal Loss implementation for imbalanced classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        alpha: Weighting factor in range (0,1) to balance pos/neg examples or a list of weights for each class.
        gamma: Exponent of the modulating factor (1 - p_t)^gamma to balance easy/hard examples.
        label_smoothing: FIX-8: Label smoothing parameter (0.0 = no smoothing, 0.1 = standard).
        reduction: Specifies reduction to apply ('mean', 'sum', 'none').
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing  # FIX-8: Added label smoothing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits of shape (N, C) where C is number of classes.
            targets: Target class indices of shape (N,).
        
        Returns:
            Focal loss.
        """
        # FIX-8: Compute cross-entropy with label smoothing
        ce = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Apply focal weight: (1 - p_t)^gamma
        loss = ce * ((1 - p_t) ** self.gamma)
        
        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Index alpha tensor, detaching is fine (weights are not trainable)
                alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
