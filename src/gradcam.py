"""Grad-CAM for the NanoMeasureNet regressor.

Standard Grad-CAM is defined for classification logits; for a scalar
regression output we use the model's own predicted diameter as the target
scalar whose gradient w.r.t. the last conv feature maps we backprop -- i.e.
"which pixels, if they changed, would most change the predicted diameter."
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x_single):
        """x_single: (1, C, H, W) tensor, requires no pre-set grad. Returns
        (cam, predicted_value) where cam is a (H, W) numpy array in [0, 1]."""
        self.model.eval()
        x_single = x_single.clone().requires_grad_(True)
        pred = self.model(x_single)  # scalar (batch size 1)
        self.model.zero_grad()
        pred.backward(retain_graph=False)

        grads = self.gradients[0]       # (C, h, w)
        acts = self.activations[0]      # (C, h, w)
        weights = grads.mean(dim=(1, 2))  # (C,) global-average-pooled gradient

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
        for c, w in enumerate(weights):
            cam += w * acts[c]
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_resized = F.interpolate(cam[None, None], size=x_single.shape[-2:],
                                      mode="bilinear", align_corners=False)[0, 0]
        return cam_resized.detach().cpu().numpy(), float(pred.item())

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def overlay_cam_on_image(image_gray_01, cam, alpha=0.45):
    """image_gray_01: (H, W) float array in [0,1]. cam: (H, W) in [0,1].
    Returns an (H, W, 3) RGB uint8 image with a jet-colormap CAM overlay."""
    import matplotlib.cm as cm
    base = np.stack([image_gray_01] * 3, axis=-1)
    heat = cm.jet(cam)[:, :, :3]
    blended = (1 - alpha) * base + alpha * heat
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)
