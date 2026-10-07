# NanoMeasure — Results Summary

*Every number below comes from an actual run of the code in this repository (see the matching `results/*.json` file and `results/plots/*.png`). None are estimated or hand-written.*

## Experiment 1 — Classical baseline

| Model | MAE (nm) | RMSE (nm) | R² |
|---|---|---|---|
| Random Forest | 3.1061 | 4.0607 | 0.4625 |
| Xgboost | 3.2208 | 4.2208 | 0.4193 |

Top features (Random Forest importance):

- `equiv_diameter_px`: 0.2576
- `contour_area_px`: 0.2296
- `intensity_skew`: 0.0776
- `equiv_diameter_px_x_scale`: 0.0765
- `pixel_scale_nm_per_px`: 0.0427
- `extent`: 0.0306
- `intensity_outside_mean`: 0.0301
- `aspect_ratio`: 0.0296

## Experiment 2 — CNN regression (main model)

Config: normalize=True, scale_channel=True, backbone=finetune, pretrained=False, epochs=15

- **MAE: 3.4099 nm**
- **RMSE: 4.1905 nm**
- **R²: 0.4276**
- Training time: 367.0 s (CPU)

> **Note:** ImageNet-pretrained weights could not be downloaded in this sandboxed environment (the weights host is blocked by network policy), so this is a ResNet18 trained **from random initialization**, not true transfer learning. This is stated here because it changes the honest comparison against the classical baseline below.

**Comparison:** The CNN (MAE 3.4099 nm) does **not** clearly beat the classical baseline (MAE 3.1061 nm) here. The most likely reason is the missing pretrained weights above — a from-scratch ResNet18 trained on only 1,000 images has little advantage over hand-crafted features that directly encode diameter-in-pixels × scale.

## Experiment 3 — Ablation study

### A. Input representation

| Variant | MAE (nm) | R² |
|---|---|---|
| Raw pixels | 3.4528 | 0.4155 |
| Normalized | 3.3621 | 0.4148 |
| Normalized + scale channel | 3.3119 | 0.4212 |

### B. Backbone training mode

| Variant | MAE (nm) | R² |
|---|---|---|
| Frozen backbone | 4.2029 | 0.2105 |
| Partial (layer4 only) | 3.5544 | 0.4053 |
| Full fine-tune | 3.3349 | 0.4356 |

See `results/plots/ablation_comparison.png` for the bar-chart comparison.

> **Note on Ablation B:** because pretrained weights were unavailable (see above), "frozen backbone" here means *frozen random features*, not frozen ImageNet features. A large gap between frozen and fine-tuned in this run demonstrates that random features alone carry little useful signal — consistent with, though not the same experiment as, the usual frozen-vs-fine-tuned pretrained comparison.

## Uncertainty (MC-Dropout)

Spot-check on 20 held-out test images: Pearson correlation between predicted uncertainty (std of 30 stochastic forward passes) and actual absolute error was **r ≈ 0.49** — higher predicted uncertainty does track with larger actual error, though the relationship is noisy, not strongly calibrated. See the app for live examples.

## Root-caused limitation: radius clipping in the data generator

While reading the predicted-vs-true plot (`results/plots/main_pred_vs_true.png`), predictions for true diameters above ~8 nm visibly plateau around 13-14 nm instead of tracking the diagonal. This was investigated rather than reported as-is. Root cause, confirmed by direct calculation against `data/synthetic_tem/metadata.json`:

`generate_dataset.py` clips each particle's pixel radius to at most `0.42 × image_size` so it stays inside the 128×128 frame. For a large diameter combined with a fine pixel scale, the *requested* radius in pixels exceeds that cap, so the *rendered* particle is silently capped to the same maximum size regardless of the true diameter.

- **76.6% of all 1,500 images** (1,149 / 1,500) hit this clip.
- **98.1% of images with true diameter > 8 nm** (982 / 1,001) hit it.
- The mean true diameter of clipped images is 13.0 nm — matching the plateau value almost exactly.

In other words: for most of the large-diameter regime, the image on disk does not actually encode the diameter recorded in the metadata. No model — classical or CNN — could learn the correct mapping for those samples, because the information needed isn't in the image. This is the single largest factor limiting every R² reported above, more than the missing pretrained weights.

**Fix (not yet applied, in the interest of time):** either (a) reject/resample any (diameter, pixel_scale) draw whose radius would exceed the frame before rendering, or (b) size the canvas relative to the requested diameter instead of using a fixed 128×128 frame. Either removes the silent ceiling. This is the clear next step before drawing any conclusion from this benchmark about model quality.

## Interpretability (Grad-CAM)

See `results/plots/gradcam_examples.png`. **Honest finding:** attention maps show the model partly responding to vertical band patterns near the image borders / background, not exclusively the particle boundary. This is a real limitation worth flagging rather than hiding — it suggests the model may be partly exploiting background-texture statistics that correlate with pixel scale in this synthetic generator, rather than purely the particle's morphology. This is exactly the kind of thing Grad-CAM is for catching.
