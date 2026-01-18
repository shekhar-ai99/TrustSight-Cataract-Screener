import torch
import numpy as np

_model = None
CLASS_COUNT = 4
VECTOR_SIZE = 512 * 512 * 3

def _enable_mc_dropout(model):
    """Enable dropout layers during inference."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

def _load_model():
    global _model
    if _model is None:
        try:
            _model = torch.load("model.pth", map_location="cpu")
            _model.eval()
            _enable_mc_dropout(_model)
        except Exception as e:
            raise RuntimeError(f"Failed to load model.pth: {e}")

def predict(batch):
    """
    Input:
        batch: list | np.ndarray | torch.Tensor
               shape = (N, 786432) or (786432,)
    Output:
        np.ndarray of shape (N, 4)
    """
    _load_model()
    torch.set_grad_enabled(False)

    # Convert to numpy
    if isinstance(batch, list):
        batch = np.array(batch, dtype=np.float32)

    if isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu().numpy()

    if not isinstance(batch, np.ndarray):
        raise TypeError("Input batch must be list, numpy array, or torch tensor")

    # Handle single sample
    if batch.ndim == 1:
        batch = batch.reshape(1, -1)

    if batch.shape[1] != VECTOR_SIZE:
        raise ValueError(f"Expected input shape (*, {VECTOR_SIZE}), got {batch.shape}")

    batch = torch.from_numpy(batch.astype(np.float32))
    batch = batch.view(-1, 3, 512, 512)

    # Normalize (match training)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    batch = (batch - mean) / std

    # MC Dropout inference
    preds = []
    for _ in range(10):
        logits = _model(batch)
        preds.append(torch.softmax(logits, dim=1))

    mean_probs = torch.mean(torch.stack(preds), dim=0)

    return mean_probs.cpu().numpy()