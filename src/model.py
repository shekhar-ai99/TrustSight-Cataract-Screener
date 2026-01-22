import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CataractModel(nn.Module):
    def __init__(self, weights_path: str | None = None, backbone: str = "resnet18"):
        super().__init__()
        # Load ResNet18 (rank #1 on leaderboard uses this)
        if backbone == "resnet18":
            try:
                # Try to load with pretrained weights
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception as e:
                # Fallback to random initialization if download fails
                print(f"Warning: Could not load pretrained ResNet18 weights ({e}). Using random init.")
                self.backbone = models.resnet18(weights=None)
            
            # Replace final FC layer for 4-class classification
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, 4)
        else:
            # Fallback to EfficientNet-B0 if specified
            from efficientnet_pytorch import EfficientNet
            self.backbone = EfficientNet.from_pretrained("efficientnet-b0", advprop=False)
            in_features = self.backbone._fc.weight.shape[1]
            self.backbone._fc = nn.Linear(in_features, 4)
        
        # Load custom weights if provided
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)

    def forward(self, x, mc: bool = False):
        """Forward pass through ResNet18 backbone."""
        if mc:
            # MC Dropout mode: enable dropout
            self.train()
            return self.backbone(x)
        else:
            # Deterministic mode: standard forward
            return self.backbone(x)


    def disable_mc_dropout(self):
        """Ensure dropout modules are in eval mode."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

    def safe_mc_forward(self, x, n_samples: int = 15):
        """
        Perform MC Dropout forward passes without permanently mutating model state.
        Temporarily set the model to train() so dropout is active, run passes under
        torch.no_grad(), then restore all modules' training flags to their originals.
        Returns a list of probability arrays (length n_samples) each of shape (4,).
        """
        orig_training = {m: m.training for m in self.modules()}
        try:
            # Enable train mode so dropout layers are active
            self.train()
            probs = []
            with torch.no_grad():
                for _ in range(n_samples):
                    out = self.forward(x)
                    # handle possible tensor shapes
                    p_tensor = F.softmax(out, dim=-1)
                    p = p_tensor.squeeze().cpu().numpy()
                    # Ensure we append a list consistently
                    if hasattr(p, "tolist"):
                        p = p.tolist()
                    probs.append(p)
            return probs
        finally:
            # Restore original training flags
            for m, was_training in orig_training.items():
                m.training = was_training

    def predict_proba(self, x, n_mc: int = 1):
        """
        If n_mc == 1: deterministic prediction with dropout disabled.
        If n_mc > 1: perform MC Dropout sampling using safe_mc_forward.
        Returns list of probability arrays.
        """
        if n_mc <= 1:
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
            # Return a list for consistency
            if probs.ndim == 0:
                return [probs.tolist()]
            return [probs.tolist()]
        else:
            probs = self.safe_mc_forward(x, n_samples=n_mc)
            # safe_mc_forward already returns list of lists
            return probs
