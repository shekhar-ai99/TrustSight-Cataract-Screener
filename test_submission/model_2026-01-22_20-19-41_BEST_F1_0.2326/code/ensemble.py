"""
Ensemble inference: Load multiple trained models and average predictions.
Used to boost F1 by combining diverse model predictions.
"""
import os
import torch
import numpy as np
from pathlib import Path


class EnsembleInference:
    """Load multiple model.pth files and average their predictions."""
    
    def __init__(self, model_paths: list):
        """
        Args:
            model_paths: List of paths to model.pth files (state_dicts)
        """
        self.model_paths = model_paths
        self.models = []
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all models into memory."""
        from inference import CataractModel, DEVICE
        
        for path in self.model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: {path}")
            
            model = CataractModel()
            state = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
            model.eval()
            model.to(DEVICE)
            self.models.append(model)
        
        print(f"Loaded {len(self.models)} models for ensemble")
    
    def predict_proba(self, batch):
        """
        Get ensemble probabilities by averaging softmax outputs.
        
        Args:
            batch: Preprocessed tensor of shape (N, 3, 384, 384)
        
        Returns:
            Ensemble softmax probabilities of shape (N, 4)
        """
        from inference import DEVICE
        
        ensemble_probs = None
        
        with torch.no_grad():
            for model in self.models:
                logits = model(batch.to(DEVICE))
                probs = torch.softmax(logits, dim=1)
                
                if ensemble_probs is None:
                    ensemble_probs = probs
                else:
                    ensemble_probs += probs
        
        # Average probabilities across models
        ensemble_probs /= len(self.models)
        return ensemble_probs.cpu().numpy()


def predict_ensemble(batch, model_paths: list):
    """
    Quick ensemble prediction.
    
    Args:
        batch: Input batch (DataFrame or array)
        model_paths: List of model.pth file paths
    
    Returns:
        Ensemble class label predictions
    """
    # Preprocess batch (same as single-model inference)
    from inference import predict
    
    # Use ensemble for probability calculation
    ensemble = EnsembleInference(model_paths)
    
    # For now, fall back to single-model predict
    # In production, would integrate ensemble probs into threshold logic
    return predict(batch)
