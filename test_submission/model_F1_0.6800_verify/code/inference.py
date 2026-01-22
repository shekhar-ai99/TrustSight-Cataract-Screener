import os
import pickle
import sys
import types
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

# Enforce deterministic behaviour for evaluator path
def _set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


_set_seed(42)

_model = None
CLASS_COUNT = 4
VECTOR_SIZE = 512 * 512 * 3
CLASS_NAMES = [
    "No Cataract",
    "Immature Cataract",
    "Mature Cataract",
    "IOL Inserted"
]

# Match training resolution (train uses 384x384); evaluator input is 512x512 flattened
_INFER_RES = 384
# Optional temperature scaling (default 1.0). If temp.txt exists beside model, use it.
_DEFAULT_TEMPERATURE = 1.0
_TEMPERATURE_PATHS = [
    os.path.join(os.path.dirname(__file__), "temp.txt"),
    "temp.txt",
]


def _load_temperature() -> float:
    for path in _TEMPERATURE_PATHS:
        if os.path.exists(path):
            try:
                value = float(open(path, "r", encoding="utf-8").read().strip())
                if value > 0:
                    return value
            except Exception:
                continue
    return _DEFAULT_TEMPERATURE


_TEMPERATURE = _load_temperature()

class CataractModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_name("efficientnet-b0")
        in_features = self.backbone._fc.weight.shape[1]
        self.backbone._fc = nn.Linear(in_features, CLASS_COUNT)

    def forward(self, x):
        return self.backbone(x)


def _disable_dropout(model):
    """Force dropout layers off for deterministic eval."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.eval()


class _UnpicklerWithClassMap(pickle.Unpickler):
    """Custom unpickler that remaps old 'model' module to CataractModel"""
    def find_class(self, module, name):
        if module == 'model' and name == 'CataractModel':
            return CataractModel
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if 'model' in module:
                return CataractModel
            raise
    
    def persistent_load(self, pid):
        """Handle persistent IDs (torch tensors stored separately)"""
        # Return a placeholder - torch.load will handle the actual loading
        return pid

def _load_model():
    global _model
    if _model is None:
        try:
            # Try to load from multiple possible locations
            model_path = os.path.join(os.path.dirname(__file__), "..", "model.pth")
            if not os.path.exists(model_path):
                model_path = "model.pth"

            # Expect a state_dict (safe for PyTorch >=2.6 evaluators)
            state = torch.load(model_path, map_location="cpu")
            if not isinstance(state, dict):
                raise RuntimeError(f"Expected state_dict, got {type(state)}")

            model_instance = CataractModel()
            model_instance.load_state_dict(state, strict=False)
            model_instance.eval()
            _disable_dropout(model_instance)
            _model = model_instance
        except Exception as e:
            raise RuntimeError(f"Failed to load model.pth: {e}")


def predict(batch):
    """
    Input:
        batch: pd.DataFrame (with 'image_vector' column) | np.ndarray | list | torch.Tensor
               DataFrame: must have 'image_vector' column
               Array: shape = (N, 786432) or (786432,)
    Output:
        np.ndarray of shape (N,) with class label strings
    """
    _load_model()
    torch.set_grad_enabled(False)

    # --- Case 1: Evaluator input (DataFrame) ---
    if hasattr(batch, "columns"):
        if "image_vector" not in batch.columns:
            raise ValueError("DataFrame must contain 'image_vector' column")
        vectors = batch["image_vector"].tolist()
        batch = np.array(vectors, dtype=np.float32)

    # --- Case 2: Raw array / tensor fallback ---
    if isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu().numpy()

    if isinstance(batch, list):
        batch = np.array(batch, dtype=np.float32)

    if not isinstance(batch, np.ndarray):
        raise TypeError("Input batch must be DataFrame, list, numpy array, or torch tensor")

    # Handle single sample
    if batch.ndim == 1:
        batch = batch.reshape(1, -1)

    # Validate vector size (explicit check before reshape)
    expected_size = batch.shape[1]
    if expected_size != VECTOR_SIZE:
        raise ValueError(
            f"image_vector length {expected_size} cannot be reshaped "
            f"to (3, 512, 512). Expected raw pixel vector of length {VECTOR_SIZE}."
        )

    batch = torch.from_numpy(batch.astype(np.float32))
    batch = batch.view(-1, 3, 512, 512)

    # Align to training resolution for consistent spatial stats
    if _INFER_RES != 512:
        batch = F.interpolate(batch, size=(_INFER_RES, _INFER_RES), mode="bilinear", align_corners=False)

    # Detect pixel scale: if values look like 0-255, scale to [0,1]
    max_val = batch.max()
    if max_val > 1.5:
        batch = batch / 255.0

    # Normalize (match training)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    batch = (batch - mean) / std

    # Get calibrated probabilities (temperature-scaled) and class predictions
    with torch.no_grad():
        logits = _model(batch)
        if _TEMPERATURE != 1.0:
            logits = logits / _TEMPERATURE
        # Slight positive bias to cataract classes (indices 1,2,3) to favor sensitivity
        bias = torch.zeros_like(logits)
        bias[:, 1:] += 0.05
        logits = logits + bias
        probs = torch.softmax(logits, dim=1)

    # Confidence-aware prediction with class-specific thresholds
    probs_np = probs.cpu().numpy()
    class_indices = []
    
    for i in range(probs_np.shape[0]):
        prob_sample = probs_np[i]
        max_prob = prob_sample.max()
        pred_idx = prob_sample.argmax()
        
        # Class-specific confidence thresholds to reduce overconfident errors
        # IOL Inserted (idx=3) has lower threshold due to class imbalance
        if pred_idx == 3 and prob_sample[3] < 0.35:
            # Fallback to second-best prediction for low-confidence IOL
            prob_sample_copy = prob_sample.copy()
            prob_sample_copy[3] = 0
            pred_idx = prob_sample_copy.argmax()
        elif max_prob < 0.40:
            # For very low confidence across all classes, prefer No Cataract (idx=0)
            # as the conservative screening default
            pred_idx = 0
        
        class_indices.append(pred_idx)
    
    class_indices = np.array(class_indices)

    # Map indices to class label strings
    labels = np.array([CLASS_NAMES[i] for i in class_indices])

    return labels