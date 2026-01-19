import torch
import numpy as np
import pandas as pd
import os
import pickle
import zipfile
import sys
import types
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

_model = None
CLASS_COUNT = 4
VECTOR_SIZE = 512 * 512 * 3
CLASS_NAMES = [
    "No Cataract",
    "Immature Cataract",
    "Mature Cataract",
    "IOL Inserted"
]

class CataractModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_name("efficientnet-b0")
        in_features = self.backbone._fc.weight.shape[1]
        self.backbone._fc = nn.Linear(in_features, CLASS_COUNT)

    def forward(self, x):
        return self.backbone(x)

def _enable_mc_dropout(model):
    """Enable dropout layers during inference."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


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
            # 1. Relative to this file's directory (for extracted submission)
            model_path = os.path.join(os.path.dirname(__file__), "..", "model.pth")
            
            # 2. Current working directory (fallback)
            if not os.path.exists(model_path):
                model_path = "model.pth"
            
            # Load model using torch.load
            # If it fails due to missing 'model' module, create an empty module
            try:
                _model = torch.load(model_path, map_location="cpu")
            except ModuleNotFoundError as e:
                if 'model' in str(e):
                    # Create a fake 'model' module in sys.modules to help unpickling
                    import sys
                    import types
                    
                    # Create fake module
                    fake_model_module = types.ModuleType('model')
                    fake_model_module.CataractModel = CataractModel
                    sys.modules['model'] = fake_model_module
                    
                    # Try again
                    _model = torch.load(model_path, map_location="cpu")
                else:
                    raise
            
            # If loaded object is a state_dict (dict), create model and load state
            if isinstance(_model, dict):
                model_instance = CataractModel()
                model_instance.load_state_dict(_model, strict=False)
                _model = model_instance
            
            _model.eval()
            _enable_mc_dropout(_model)
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

    # Normalize (match training)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    batch = (batch - mean) / std

    # Get class predictions (argmax instead of probabilities)
    with torch.no_grad():
        logits = _model(batch)
    
    class_indices = torch.argmax(logits, dim=1).cpu().numpy()

    # Map indices to class label strings
    labels = np.array([CLASS_NAMES[i] for i in class_indices])

    return labels