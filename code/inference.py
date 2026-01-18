import torch
import numpy as np
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import cv2
import albumentations as A

# Deterministic setup
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

# Model
class CataractModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_name("efficientnet-b0")
        in_features = self.backbone._fc.weight.shape[1]
        self.backbone._fc = nn.Linear(in_features, 4)

    def forward(self, x, mc: bool = False):
        out = self.backbone(x)
        return out

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def disable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

    def safe_mc_forward(self, x, n_samples: int = 15):
        orig_training = {m: m.training for m in self.modules()}
        try:
            self.enable_mc_dropout()
            probs = []
            for _ in range(n_samples):
                with torch.no_grad():
                    logit = self.forward(x)
                    prob = torch.softmax(logit, dim=-1).squeeze().cpu().numpy().tolist()
                    probs.append(prob)
        finally:
            for m, train in orig_training.items():
                m.train(train)
        return probs

# Preprocess
# (handled inline in predict)

# IQA
def check_image_quality(img):
    try:
        if img is None or img.size == 0:
            return "REJECT", "INVALID_IMAGE"
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 100:
            return "REJECT", "BLURRY"
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mean_s = np.mean(hsv[:, :, 1])
        if mean_s < 30:
            return "REJECT", "LOW_CONTRAST"
        return "ACCEPT", ""
    except:
        return "REJECT", "ERROR"

# MC inference
def mc_inference(model, img_tensor, n_mc=15, seed=42):
    torch.manual_seed(seed)
    return model.safe_mc_forward(img_tensor, n_samples=n_mc)

# Class labels
CLASS_NAMES = ["No Cataract", "Immature Cataract", "Mature Cataract", "IOL Inserted"]

# Global model
_model = None

def _load_model():
    global _model
    if _model is None:
        _model = CataractModel()
        _model.backbone.load_state_dict(torch.load("final_model.pth", map_location="cpu"))
        _model.eval()

def predict(batch):
    """
    Input: batch (list of image paths or numpy array of images)
    Output: list of dicts with 'predicted_class', 'class_probs', 'confidence', 'uncertainty'
    """
    _load_model()
    if isinstance(batch, torch.Tensor):
        batch = batch.cpu().numpy()
    results = []
    for item in batch:
        if isinstance(item, str):
            img = cv2.imread(item)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = item  # assume np.ndarray (H, W, C)
        status, reason = check_image_quality(img)
        if status == "REJECT":
            results.append({
                "predicted_class": "REJECT",
                "class_probs": {name: 0.0 for name in CLASS_NAMES},
                "confidence": 0.0,
                "uncertainty": "HIGH"
            })
            continue
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToFloat(max_value=255.0)
        ])
        transformed = transform(image=img)
        img_arr = transformed["image"]
        img_tensor = torch.tensor(img_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        mc_probs = mc_inference(_model, img_tensor, n_mc=15, seed=42)
        mean_probs = np.mean(mc_probs, axis=0)
        pred_class = np.argmax(mean_probs)
        predicted_class = CLASS_NAMES[pred_class]
        confidence = float(np.max(mean_probs))
        # Uncertainty as variance of the predicted class prob
        pred_class_probs = [sample[pred_class] for sample in mc_probs]
        variance = np.var(pred_class_probs)
        if variance < 0.01:
            uncertainty = "LOW"
        elif variance < 0.05:
            uncertainty = "MEDIUM"
        else:
            uncertainty = "HIGH"
        class_probs = {CLASS_NAMES[i]: float(mean_probs[i]) for i in range(4)}
        results.append({
            "predicted_class": predicted_class,
            "class_probs": class_probs,
            "confidence": confidence,
            "uncertainty": uncertainty
        })
    return results