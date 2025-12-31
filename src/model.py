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
        # Default behavior unchanged: eval mode unless mc=True
        if mc:
            # enable dropout layers during inference but keep overall model in eval
            self.enable_mc_dropout()
        else:
            self.disable_mc_dropout()
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

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x, mc=False)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        return probs
