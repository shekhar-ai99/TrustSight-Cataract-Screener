import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class CataractModel(nn.Module):
    def __init__(self, weights_path: str | None = None):
        super().__init__()
        # Prefer local weights; avoid internet calls in all cases
        if weights_path and os.path.exists(weights_path):
            self.backbone = EfficientNet.from_name("efficientnet-b0")
            state = torch.load(weights_path, map_location="cpu")
            self.backbone.load_state_dict(state)
        else:
            # Create architecture without attempting to download pretrained weights
            self.backbone = EfficientNet.from_name("efficientnet-b0")

        in_features = getattr(self.backbone._fc, "in_features", None)
        if in_features is None:
            in_features = self.backbone._fc.weight.shape[1]
        self.backbone._fc = nn.Linear(in_features, 1)

    def forward(self, x, mc: bool = False):
        # Default behavior: just run backbone. MC behavior is handled by safe_mc_forward
        out = self.backbone(x)
        return out

    def enable_mc_dropout(self):
        """Force all dropout modules to train mode so they stay active during inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

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
        Returns a list of probabilities (floats) of length n_samples.
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
                    p_tensor = torch.sigmoid(out)
                    p = p_tensor.squeeze().cpu().numpy()
                    # Ensure we append a scalar or 1-d array consistently
                    if hasattr(p, "tolist"):
                        p = p.tolist()
                    probs.append(float(p) if isinstance(p, (float, int)) else p)
            return probs
        finally:
            # Restore original training flags
            for m, was_training in orig_training.items():
                m.training = was_training

    def predict_proba(self, x, n_mc: int = 1):
        """
        If n_mc == 1: deterministic prediction with dropout disabled.
        If n_mc > 1: perform MC Dropout sampling using safe_mc_forward.
        """
        if n_mc <= 1:
            with torch.no_grad():
                logits = self.forward(x)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            # Return a list for consistency when downstream expects iterable
            if isinstance(probs, (float, int)):
                return [float(probs)]
            return probs.tolist()
        else:
            probs = self.safe_mc_forward(x, n_samples=n_mc)
            # safe_mc_forward already returns list of floats
            return probs
