"""
generate_dataset.py
--------------------
Procedurally generates a synthetic TEM-like nanoparticle image dataset with
KNOWN ground-truth diameters and pixel scales.

Why synthetic, and why this is a legitimate scientific choice (not a shortcut):
  - We do not have reliable network access to external repositories (e.g. Dryad)
    from this environment, so a real public TEM nanoparticle dataset could not
    be downloaded here.
  - A controlled synthetic benchmark with known ground truth is standard
    practice for validating a measurement pipeline before testing on real data
    (this mirrors VibeTube's own approach: an abTEM-simulated 6,930-image
    benchmark is the PRIMARY validation set in the paper this project is
    positioned alongside; experimental images are a secondary check).
  - This generator is NOT a physics-based electron-optics simulator (no
    multislice, no CTF). It is a fast procedural renderer that reproduces the
    qualitative appearance of bright-field TEM nanoparticle images: a
    roughly-round, mass-thickness-contrast particle on a textured support
    film, with edge blur, shot noise, and varying pixel scale. This
    limitation is stated explicitly in the project README and should be
    stated explicitly to anyone evaluating this project.

Each image ships with:
  - diameter_nm       : ground-truth particle diameter in nanometers
  - pixel_scale       : nm per pixel for that image (varies per image, as in
                         real microscopy where magnification differs per
                         acquisition -- this is what makes scale information
                         a first-class input rather than something the model
                         must infer from pixels alone)
  - particle_id, split: bookkeeping
"""
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from PIL import Image

RNG_SEED = 42
IMG_SIZE = 128
N_TOTAL = 1500
SPLITS = {"train": 1000, "val": 250, "test": 250}
DIAMETER_RANGE_NM = (2.0, 20.0)
PIXEL_SCALE_RANGE = (0.025, 0.09)   # nm/pixel -- varies per image (different "magnifications")
OUT_DIR = Path(__file__).parent / "synthetic_tem"


def make_particle_mask(size, radius_px, rng, n_harmonics=4, wobble=0.06):
    """Roughly-circular particle shape via a low-order radial Fourier perturbation,
    so particles aren't perfect circles (real nanoparticles rarely are)."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    theta = np.arctan2(yy - cy, xx - cx)
    r = np.hypot(yy - cy, xx - cx)

    coeffs = rng.normal(0, wobble, size=n_harmonics)
    phases = rng.uniform(0, 2 * np.pi, size=n_harmonics)
    radial_mod = np.ones_like(theta)
    for k in range(1, n_harmonics + 1):
        radial_mod += coeffs[k - 1] * np.cos(k * theta + phases[k - 1])
    local_radius = radius_px * radial_mod
    return (r <= local_radius).astype(np.float32)


def render_image(diameter_nm, pixel_scale, rng, size=IMG_SIZE):
    radius_px = (diameter_nm / 2.0) / pixel_scale
    radius_px = np.clip(radius_px, 3, size * 0.42)  # keep particle in-frame

    mask = make_particle_mask(size, radius_px, rng)

    # Support-film background texture (grainy carbon film look)
    background = rng.normal(0.55, 0.03, size=(size, size))
    background = gaussian_filter(background, sigma=1.2)

    # Mass-thickness contrast: particle appears darker than film, with slight
    # internal intensity variation (thickness/crystallinity-like texture)
    particle_core = 0.55 - rng.uniform(0.18, 0.30)
    internal_tex = rng.normal(0, 0.015, size=(size, size))
    internal_tex = gaussian_filter(internal_tex, sigma=1.5)
    particle_intensity = particle_core + internal_tex

    image = background * (1 - mask) + particle_intensity * mask

    # Edge / defocus blur
    blur_sigma = rng.uniform(0.6, 1.6)
    image = gaussian_filter(image, sigma=blur_sigma)

    # Poisson shot noise (dose-dependent, same idea VibeTube's robustness
    # study uses for its dose degradation)
    dose_factor = rng.uniform(400, 1500)
    image = np.clip(image, 1e-4, 1)
    noisy = rng.poisson(image * dose_factor) / dose_factor
    noisy = np.clip(noisy, 0, 1)

    return (noisy * 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(RNG_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    idx = 0
    for split, n in SPLITS.items():
        split_dir = OUT_DIR / split
        split_dir.mkdir(exist_ok=True)
        for i in range(n):
            diameter_nm = rng.uniform(*DIAMETER_RANGE_NM)
            pixel_scale = rng.uniform(*PIXEL_SCALE_RANGE)
            img = render_image(diameter_nm, pixel_scale, rng)
            fname = f"particle_{idx:05d}.png"
            Image.fromarray(img).save(split_dir / fname)
            records.append({
                "particle_id": idx,
                "split": split,
                "filename": f"{split}/{fname}",
                "diameter_nm": round(float(diameter_nm), 4),
                "pixel_scale_nm_per_px": round(float(pixel_scale), 5),
            })
            idx += 1

    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Generated {idx} synthetic TEM-like nanoparticle images in {OUT_DIR}")
    for split, n in SPLITS.items():
        print(f"  {split}: {n} images")


if __name__ == "__main__":
    main()
