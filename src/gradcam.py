import numpy as np
import cv2
import torch

def _find_last_conv(module):
    last_conv = None
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    return last_conv

def generate_gradcam(model, image_tensor, target_class=1, out_path="outputs/gradcam_image.jpg"):
    """Generate Grad-CAM heatmap for the given model and image tensor.

    Args:
        model: torch model
        image_tensor: torch.Tensor, shape (1,C,H,W)
        target_class: int (for binary single-output model, ignored)
        out_path: where to save overlay PNG

    Returns:
        heatmap: numpy array (H,W,3) BGR
    """
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # Find target convolutional layer
    target_layer = None
    # Try common attribute names
    for attr in ["_conv_head", "conv_head", "features"]:
        if hasattr(model.backbone, attr):
            target_layer = getattr(model.backbone, attr)
            break
    if target_layer is None:
        target_layer = _find_last_conv(model.backbone)

    if target_layer is None:
        raise RuntimeError("Could not find convolutional layer for Grad-CAM")

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    # Forward
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad = True
    out = model(image_tensor, mc=False)
    # For binary single-logit model use sigmoid
    score = torch.sigmoid(out).squeeze()
    # Backward on the score scalar
    model.zero_grad()
    if score.ndim > 0:
        score = score.mean()
    score.backward(retain_graph=False)

    h1.remove()
    h2.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks did not fire")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # Normalize cam
    cam -= cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize to image size
    cam_img = (cam * 255).astype('uint8')
    H, W = cam_img.shape
    # If input size differs, resize to input spatial dims
    input_H = image_tensor.shape[2]
    input_W = image_tensor.shape[3]
    cam_resized = cv2.resize(cam_img, (input_W, input_H))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

    # Overlay on original image reconstructed from tensor
    # Convert tensor back to RGB image
    img_np = image_tensor.squeeze(0).cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    # Un-normalize (approx)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np * 255.0, 0, 255).astype('uint8')
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    # Ensure outputs directory exists
    import os
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, overlay)

    return heatmap
