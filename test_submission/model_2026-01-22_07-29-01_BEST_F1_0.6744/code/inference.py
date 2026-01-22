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

# Per-class confidence thresholds (tunable for each class)
# Format: {class_idx: threshold}
# If pred_prob[class_idx] < threshold, fallback to second-best
_CLASS_THRESHOLDS = {
    0: 0.40,  # No Cataract
    1: 0.45,  # Immature Cataract
    2: 0.40,  # Mature Cataract
    3: 0.60,  # IOL Inserted (high threshold due to class imbalance)
}

# Global fallback confidence threshold
_GLOBAL_THRESHOLD = 0.40


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


def _apply_tta(batch_tensor):
    """
    Test-Time Augmentation: apply multiple transformations and average predictions.
    
    Augmentations: original, horizontal flip, vertical flip, horizontal+vertical flip
    
    Args:
        batch_tensor: Preprocessed tensor of shape (N, 3, H, W)
    
    Returns:
        Ensemble softmax probabilities from all augmentations
    """
    augmented_probs = []
    
    # Original
    with torch.no_grad():
        logits = _model(batch_tensor)
        if _TEMPERATURE != 1.0:
            logits = logits / _TEMPERATURE
        bias = torch.zeros_like(logits)
        bias[:, 1:] += 0.05
        logits = logits + bias
        probs = torch.softmax(logits, dim=1)
        augmented_probs.append(probs)
    
    # Horizontal flip
    batch_h_flip = torch.flip(batch_tensor, dims=[3])
    with torch.no_grad():
        logits = _model(batch_h_flip)
        if _TEMPERATURE != 1.0:
            logits = logits / _TEMPERATURE
        bias = torch.zeros_like(logits)
        bias[:, 1:] += 0.05
        logits = logits + bias
        probs = torch.softmax(logits, dim=1)
        augmented_probs.append(probs)
    
    # Vertical flip
    batch_v_flip = torch.flip(batch_tensor, dims=[2])
    with torch.no_grad():
        logits = _model(batch_v_flip)
        if _TEMPERATURE != 1.0:
            logits = logits / _TEMPERATURE
        bias = torch.zeros_like(logits)
        bias[:, 1:] += 0.05
        logits = logits + bias
        probs = torch.softmax(logits, dim=1)
        augmented_probs.append(probs)
    
    # Horizontal + Vertical flip
    batch_hv_flip = torch.flip(batch_tensor, dims=[2, 3])
    with torch.no_grad():
        logits = _model(batch_hv_flip)
        if _TEMPERATURE != 1.0:
            logits = logits / _TEMPERATURE
        bias = torch.zeros_like(logits)
        bias[:, 1:] += 0.05
        logits = logits + bias
        probs = torch.softmax(logits, dim=1)
        augmented_probs.append(probs)
    
    # Average all augmented predictions
    ensemble_probs = torch.stack(augmented_probs, dim=0).mean(dim=0)
    return ensemble_probs


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
        # Parse comma-separated strings if needed
        parsed_vectors = []
        for v in vectors:
            if isinstance(v, str):
                # Parse comma-separated string to float array
                v = np.fromstring(v, sep=',', dtype=np.float32)
            else:
                v = np.array(v, dtype=np.float32)
            parsed_vectors.append(v)
        batch = np.array(parsed_vectors, dtype=np.float32)

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

    # Test-Time Augmentation: ensemble across 4 flipped variants
    probs = _apply_tta(batch)
    probs_np = probs.cpu().numpy()
    class_indices = []
    
    for i in range(probs_np.shape[0]):
        prob_sample = probs_np[i]
        max_prob = prob_sample.max()
        pred_idx = prob_sample.argmax()
        
        # Apply per-class threshold
        class_threshold = _CLASS_THRESHOLDS.get(pred_idx, _GLOBAL_THRESHOLD)
        if prob_sample[pred_idx] < class_threshold:
            # Fallback to second-best prediction if below threshold
            prob_sample_copy = prob_sample.copy()
            prob_sample_copy[pred_idx] = 0
            pred_idx = prob_sample_copy.argmax()
        
        # Global confidence fallback: very low confidence → default to No Cataract
        if max_prob < _GLOBAL_THRESHOLD:
            pred_idx = 0
        
        class_indices.append(pred_idx)
    
    class_indices = np.array(class_indices)

    # Map indices to class label strings
    labels = np.array([CLASS_NAMES[i] for i in class_indices])

    return labels