"""Classical (non-deep-learning) feature extraction for the baseline model.

Features fall into 5 groups, matching the project spec:
  - edge density       : fraction of pixels on a Canny edge map
  - contour statistics  : largest-contour area/perimeter/circularity/solidity
  - texture             : GLCM contrast/homogeneity/energy (particle interior)
  - intensity statistics: mean/std/skew/kurtosis, inside vs. outside contour
  - shape descriptors    : equivalent diameter (px), aspect ratio, extent

All features are computed in PIXELS where relevant, with pixel_scale passed
in separately so the model can (and must) combine shape-in-pixels with the
image's physical scale to recover a physical diameter -- this is the same
scale-awareness issue flagged for the CNN.
"""
import cv2
import numpy as np
from scipy import stats as sstats
from skimage.feature import graycomatrix, graycoprops


def _largest_contour(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def extract_features(img_u8, pixel_scale):
    """img_u8: 2D uint8 grayscale array. pixel_scale: nm/pixel (float).
    Returns a dict of named scalar features."""
    feats = {}
    img = img_u8.astype(np.float32)

    # --- intensity statistics (whole image) ---
    feats["intensity_mean"] = float(img.mean())
    feats["intensity_std"] = float(img.std())
    feats["intensity_skew"] = float(sstats.skew(img.ravel()))
    feats["intensity_kurtosis"] = float(sstats.kurtosis(img.ravel()))

    # --- segment particle via Otsu threshold (particle is the darker blob) ---
    _, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # --- edge density ---
    edges = cv2.Canny(img_u8, 50, 150)
    feats["edge_density"] = float((edges > 0).mean())

    contour = _largest_contour(binary)
    if contour is not None and cv2.contourArea(contour) > 4:
        area_px = cv2.contourArea(contour)
        perim_px = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else area_px
        x, y, w, h = cv2.boundingRect(contour)

        feats["contour_area_px"] = float(area_px)
        feats["contour_perimeter_px"] = float(perim_px)
        feats["circularity"] = float(4 * np.pi * area_px / (perim_px ** 2 + 1e-6))
        feats["solidity"] = float(area_px / (hull_area + 1e-6))
        feats["equiv_diameter_px"] = float(2 * np.sqrt(area_px / np.pi))
        feats["aspect_ratio"] = float(w / (h + 1e-6))
        feats["extent"] = float(area_px / (w * h + 1e-6))

        # intensity inside vs outside the particle mask
        mask = np.zeros_like(img_u8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        inside = img[mask > 0]
        outside = img[mask == 0]
        feats["intensity_inside_mean"] = float(inside.mean()) if inside.size else 0.0
        feats["intensity_outside_mean"] = float(outside.mean()) if outside.size else 0.0
        feats["contrast_in_out"] = feats["intensity_outside_mean"] - feats["intensity_inside_mean"]
    else:
        for k in ["contour_area_px", "contour_perimeter_px", "circularity", "solidity",
                  "equiv_diameter_px", "aspect_ratio", "extent",
                  "intensity_inside_mean", "intensity_outside_mean", "contrast_in_out"]:
            feats[k] = 0.0

    # --- texture (GLCM) on the full image, downscaled gray levels for speed ---
    img_q = (img_u8 // 32).astype(np.uint8)  # quantize to 8 levels
    glcm = graycomatrix(img_q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2],
                          levels=8, symmetric=True, normed=True)
    feats["glcm_contrast"] = float(graycoprops(glcm, "contrast").mean())
    feats["glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity").mean())
    feats["glcm_energy"] = float(graycoprops(glcm, "energy").mean())
    feats["glcm_correlation"] = float(graycoprops(glcm, "correlation").mean())

    # --- the explicit scale input, and a pixel-based size feature combined
    # with it, so the model isn't forced to re-discover "diameter = size_px *
    # scale" on its own ---
    feats["pixel_scale_nm_per_px"] = float(pixel_scale)
    feats["equiv_diameter_px_x_scale"] = feats["equiv_diameter_px"] * pixel_scale

    return feats


FEATURE_NAMES = None  # populated lazily from the first extraction for a stable column order


def features_to_vector(feats):
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        FEATURE_NAMES = sorted(feats.keys())
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=float)
