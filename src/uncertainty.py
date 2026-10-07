"""MC-Dropout uncertainty estimation.

At inference, keep Dropout ACTIVE (model.train() for dropout layers only) and
run the forward pass N times. The spread (std) across the N stochastic
predictions is used as a predictive uncertainty estimate -- cheap, no
retraining, no ensemble of separate models needed.
"""
import numpy as np
import torch


def enable_mc_dropout(model):
    """Set the whole model to eval() (so BatchNorm uses running stats) but
    force Dropout layers back into train() mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predict_with_uncertainty(model, x, n_samples=30):
    """x: (1, C, H, W) tensor. Returns (mean_prediction, std_uncertainty, samples)."""
    enable_mc_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            preds.append(pred.item())
    preds = np.array(preds)
    return float(preds.mean()), float(preds.std()), preds
