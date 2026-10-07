"""NanoMeasure demo app.

Run with:  streamlit run app/streamlit_app.py

Upload a TEM-like image, give its pixel scale, and get:
  - predicted diameter (nm) with MC-Dropout uncertainty
  - a Grad-CAM overlay showing what the model attended to
  - (if a ground-truth diameter is known, e.g. for a held-out test image)
    the prediction error
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model_cnn import NanoMeasureNet  # noqa: E402
from uncertainty import predict_with_uncertainty  # noqa: E402
from gradcam import GradCAM, overlay_cam_on_image  # noqa: E402

MODEL_PATH = ROOT / "results" / "models" / "main.pt"

st.set_page_config(page_title="NanoMeasure", layout="centered")
st.title("NanoMeasure")
st.caption("TEM image → nanoparticle diameter, with uncertainty and Grad-CAM interpretability.")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = NanoMeasureNet(in_channels=ckpt["in_channels"], backbone_mode=ckpt["backbone_mode"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


loaded = load_model()
if loaded is None:
    st.error("No trained model found at results/models/main.pt. Run the training "
              "pipeline first (see README — `python src/train_cnn.py --name main "
              "--normalize --scale-channel --backbone finetune`).")
    st.stop()

model, ckpt = loaded
normalize = ckpt["normalize"]
use_scale_channel = ckpt["use_scale_channel"]

uploaded = st.file_uploader("Upload a TEM image (grayscale, roughly centered particle)",
                              type=["png", "jpg", "jpeg"])
pixel_scale = st.number_input("Pixel size (nm / pixel)", min_value=0.001, max_value=1.0,
                                 value=0.05, step=0.001, format="%.4f")
ground_truth = st.number_input("Ground-truth diameter, if known (nm) — optional, 0 = unknown",
                                  min_value=0.0, value=0.0, step=0.1)

if uploaded is not None:
    img = Image.open(uploaded).convert("L").resize((128, 128))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_display = img_arr.copy()

    proc = img_arr.copy()
    if normalize:
        proc = (proc - proc.mean()) / (proc.std() + 1e-6)
    img3 = np.stack([proc, proc, proc], axis=0)
    if use_scale_channel:
        scale_channel = np.full_like(proc[None], pixel_scale)
        img_in = np.concatenate([img3, scale_channel], axis=0)
    else:
        img_in = img3
    x = torch.from_numpy(img_in).float().unsqueeze(0)  # (1, C, H, W)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_display, caption="Input image", clamp=True, use_container_width=True)

    mean_pred, std_pred, _ = predict_with_uncertainty(model, x, n_samples=30)

    cam_layer = model.get_last_conv_layer()
    gc = GradCAM(model, cam_layer)
    cam, _ = gc(x)
    gc.remove()
    overlay = overlay_cam_on_image(img_display, cam)

    with col2:
        st.image(overlay, caption="Grad-CAM: what the model attended to",
                   use_container_width=True)

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted diameter", f"{mean_pred:.2f} nm", f"±{std_pred:.2f} nm")
    if ground_truth > 0:
        err = mean_pred - ground_truth
        m2.metric("Ground truth", f"{ground_truth:.2f} nm")
        m3.metric("Error", f"{err:+.2f} nm")
    else:
        m2.metric("Ground truth", "unknown")
        m3.metric("Uncertainty / prediction", f"{(std_pred/max(mean_pred,1e-6))*100:.1f}%")

    if std_pred > 0.15 * mean_pred:
        st.warning("High relative uncertainty — treat this prediction with caution. "
                     "This usually means the image is outside the range the model was "
                     "confidently trained on (very small/large particle, unusual contrast, "
                     "or heavy noise).")
else:
    st.info("Upload an image to get a prediction. You can use one of the test images "
             "in `data/synthetic_tem/test/` to try it out.")

with st.expander("About this model"):
    st.write(f"""
    - Backbone: ResNet18 (ImageNet-pretrained), mode = **{ckpt['backbone_mode']}**
    - Input: {'normalized' if normalize else 'raw'} image"""
             + (" + explicit pixel-scale channel" if use_scale_channel else "") + """
    - Uncertainty: MC-Dropout, 30 stochastic forward passes
    - Trained and evaluated entirely on a **synthetic** TEM-like nanoparticle
      benchmark with known ground truth (see project README for why, and for
      the real-data path).
    """)
