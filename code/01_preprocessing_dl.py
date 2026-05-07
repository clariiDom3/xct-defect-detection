"""
01_preprocessing_v2.py
======================
Step 1 — Convert every image AND its two separate GT masks to polar
coordinates.

This is the V2 version of 01_preprocessing.py, updated to:
  - Import config_dl (which re-exports all config_v2 paths).
  - Handle TWO separate binary masks per image (sample + pores) instead
    of a single combined mask with label values 0/1/2.
  - Write polar outputs to the V2 folder structure:
      polar/images/           <stem>_polar.png
      polar/masks/sample/     <stem>_polar_sample.png   (binary 0/255)
      polar/masks/pores/      <stem>_polar_pores.png    (binary 0/255)
      polar/vis/              <stem>_polar_vis.png      (QA, --vis flag)

The four polar transforms (warp_polar → radial crop → transpose →
tight row crop) are applied with IDENTICAL parameters to the image
and BOTH masks so all three remain perfectly pixel-aligned.

All transform logic is unchanged from 01_preprocessing.py — only the
import, path references, and mask I/O have been updated.

Usage
-----
  python 01_preprocessing_v2.py           # with QA images
  python 01_preprocessing_v2.py --no-vis  # skip QA images

Place this file in:
  DL/CBAM/01_preprocessing_v2.py
config_dl.py must be in the same folder.

Dependencies
------------
  pip install scikit-image opencv-python matplotlib tqdm numpy
"""

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.transform import warp_polar
from tqdm import tqdm

import config_dl as cfg   # replaces the old 'import config'

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# Centre detection  (unchanged from 01_preprocessing.py)
# ══════════════════════════════════════════════════════════════════════════════

def detect_center(gray: np.ndarray):
    """
    Find the geometric centre + outer radius of the ring for warp_polar.

    Strategy: minEnclosingCircle on the largest bright outer contour.
    Falls back to centroid of the inner dark hole, then image centre.

    Returns (row, col, radius, method_name).
    """
    h, w = gray.shape

    # Primary: minEnclosingCircle on largest bright outer contour
    try:
        blurred = cv2.GaussianBlur(gray, (15, 15), 3)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest      = max(contours, key=cv2.contourArea)
            (ex, ey), er = cv2.minEnclosingCircle(largest)
            row = int(np.clip(round(ey), 0, h - 1))
            col = int(np.clip(round(ex), 0, w - 1))
            return row, col, float(er), "minEnclosingCircle"
    except Exception:
        pass

    # Fallback A: centroid of inner dark hole
    try:
        thresh = threshold_otsu(gray)
        dark   = (gray <= thresh).astype(np.uint8)
        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
        interior = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i][:5]
            if not (x == 0 or y == 0 or x + bw >= w or y + bh >= h):
                interior.append((area, i))
        if interior:
            interior.sort(reverse=True)
            cx, cy     = centroids[interior[0][1]]
            row_c, col_c = int(round(cy)), int(round(cx))
            rad        = float(min(row_c, h - row_c, col_c, w - col_c))
            return row_c, col_c, rad, "dark-hole"
    except Exception:
        pass

    # Fallback B: image centre
    return h // 2, w // 2, float(min(h, w)) / 2.0, "image-center-fallback"


# ══════════════════════════════════════════════════════════════════════════════
# Crop helpers  (unchanged from 01_preprocessing.py)
# ══════════════════════════════════════════════════════════════════════════════

def find_radial_crop_bounds(col_sums: np.ndarray,
                            fraction: float = cfg.CROP_FRACTION):
    """
    Two-sided radial crop bounds from the column-sum profile.
    left_col  = first col where col-sum > fraction × max  (inner void edge)
    right_col = last  col where col-sum > fraction × max  (outer edge)
    Returns (left_col, right_col) inclusive.
    """
    threshold = col_sums.max() * fraction
    hits      = np.where(col_sums > threshold)[0]
    if len(hits) == 0:
        return 0, len(col_sums) - 1
    return int(hits[0]), int(hits[-1])


def find_tight_row_bounds(img_transposed: np.ndarray,
                          padding: float = 0.02,
                          smooth_sigma: float = 5.0):
    """
    Locate the top and bottom of the material band using gradient-based
    edge detection on the per-row mean profile.

    Gradient approach is scale-invariant and avoids the Otsu problem of
    cutting into the material when the ring is thick or bright.

    Returns (top_row, bottom_row) with a small padding.
    """
    from scipy.ndimage import gaussian_filter1d

    n_rows  = img_transposed.shape[0]
    pad_px  = max(1, int(n_rows * padding))
    profile = img_transposed.mean(axis=1)
    smooth  = gaussian_filter1d(profile.astype(np.float64), sigma=smooth_sigma)
    grad    = np.gradient(smooth)

    top_row    = int(np.argmax(grad))
    bottom_row = int(np.argmin(grad))

    if top_row >= bottom_row:
        # Fallback: half-max threshold
        half_max = smooth.max() / 2.0
        hits     = np.where(smooth > half_max)[0]
        if len(hits) >= 2:
            top_row, bottom_row = int(hits[0]), int(hits[-1])
        else:
            top_row, bottom_row = 0, n_rows - 1

    top    = max(0,          top_row    - pad_px)
    bottom = min(n_rows - 1, bottom_row + pad_px)
    return top, bottom


# ══════════════════════════════════════════════════════════════════════════════
# Polar transform  (unchanged logic, accepts multiple masks now)
# ══════════════════════════════════════════════════════════════════════════════

def to_polar(image: np.ndarray,
             center: tuple,
             radius: float,
             output_shape: tuple,
             is_mask: bool = False) -> np.ndarray:
    """
    warp_polar wrapper.
    Images  -> bicubic  (order=3): smooth intensity gradients preserved.
    Masks   -> nearest  (order=0): binary values 0/255 preserved exactly.
    """
    return warp_polar(
        image,
        center         = center,
        radius         = radius,
        output_shape   = output_shape,
        order          = 0 if is_mask else 3,
        preserve_range = True,
    )


def polar_transform(raw_gray:    np.ndarray,
                    sample_mask: np.ndarray | None,
                    pore_mask:   np.ndarray | None):
    """
    Full Step-1 pipeline for one image + two masks.

    All four transforms (warp_polar, radial crop, transpose, row crop)
    are applied with IDENTICAL parameters to the image and BOTH masks.
    This guarantees pixel-level alignment between image and GT masks
    throughout the rest of the pipeline.

    Parameters
    ----------
    raw_gray    : uint8 or uint16 grayscale image
    sample_mask : binary uint8 (0/255) or None
    pore_mask   : binary uint8 (0/255) or None

    Returns
    -------
    polar_img       : float64 [0,1]  shape (material_H, perimeter_W)
    polar_sample    : uint8   binary shape (material_H, perimeter_W) or None
    polar_pores     : uint8   binary shape (material_H, perimeter_W) or None
    det_method      : str     which centre-detection method was used
    """
    h, w = raw_gray.shape

    # ── Centre + radius ───────────────────────────────────────────────────────
    row_c, col_c, ring_radius, det_method = detect_center(raw_gray)
    center    = (row_c, col_c)
    n_angles  = int(2 * np.pi * ring_radius)   # ≈ circumference in pixels
    n_radii   = int(ring_radius)               # outer ring edge
    out_shape = (n_angles, n_radii)
    radius    = ring_radius

    # ── Step 1: normalise + warp_polar ────────────────────────────────────────
    img = raw_gray.astype(np.float64)
    if img.max() > 0:
        img = img / img.max()

    polar_img = to_polar(img, center, radius, out_shape, is_mask=False)

    # Warp both masks if provided
    def _warp_mask(m):
        if m is None:
            return None
        return to_polar(m.astype(np.float64), center, radius,
                        out_shape, is_mask=True)

    ps = _warp_mask(sample_mask)
    pp = _warp_mask(pore_mask)

    # ── Step 2: two-sided radial crop (columns = radial axis) ─────────────────
    col_sums            = np.sum(polar_img, axis=0)
    left_col, right_col = find_radial_crop_bounds(col_sums)
    polar_img           = polar_img[:, left_col : right_col + 1]
    if ps is not None: ps = ps[:, left_col : right_col + 1]
    if pp is not None: pp = pp[:, left_col : right_col + 1]

    # ── Step 3: transpose → horizontal layout ─────────────────────────────────
    polar_img = polar_img.T
    if ps is not None: ps = ps.T
    if pp is not None: pp = pp.T

    # ── Step 4: tight row crop (removes dark background rows) ─────────────────
    top, bottom = find_tight_row_bounds(polar_img)
    polar_img   = polar_img[top : bottom + 1, :]
    if ps is not None: ps = ps[top : bottom + 1, :]
    if pp is not None: pp = pp[top : bottom + 1, :]

    # Round mask values and cast to uint8 binary (0/255)
    def _finalise_mask(m):
        if m is None:
            return None
        return (np.round(m) > 0.5).astype(np.uint8) * 255

    return polar_img, _finalise_mask(ps), _finalise_mask(pp), det_method


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation  (updated for two masks)
# ══════════════════════════════════════════════════════════════════════════════

def save_vis(raw_gray:     np.ndarray,
             polar_img:    np.ndarray,
             polar_sample: np.ndarray | None,
             polar_pores:  np.ndarray | None,
             path:         Path):
    """
    QA figure — 4 panels:
      1. Original image
      2. Polar image (tightly cropped)
      3. Polar + sample mask overlay  (green)   — if available
      4. Polar + pore mask overlay    (red)     — if available
    """
    panels = [("Original", raw_gray, None, None)]
    panels.append(("Polar (cropped)\n"
                   f"{polar_img.shape[0]}×{polar_img.shape[1]}",
                   polar_img, None, None))
    if polar_sample is not None:
        panels.append(("Polar + sample GT", polar_img, polar_sample, "Greens"))
    if polar_pores is not None:
        panels.append(("Polar + pore GT",   polar_img, polar_pores,  "Reds"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (title, base, overlay, cmap) in zip(axes, panels):
        ax.imshow(base, cmap="gray", aspect="auto")
        if overlay is not None:
            ax.imshow(overlay > 0, cmap=cmap, alpha=0.45, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(save_vis_flag: bool = True):
    # ── Create output directories ─────────────────────────────────────────────
    for d in (cfg.POLAR_IMAGES_DIR,
              cfg.POLAR_SAMPLE_MASKS_DIR,
              cfg.POLAR_PORE_MASKS_DIR,
              cfg.POLAR_VIS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ── Find all images ───────────────────────────────────────────────────────
    image_paths = sorted(cfg.IMAGES_DIR.glob("*.png"))
    if not image_paths:
        print(f"[ERROR] No PNG images found in {cfg.IMAGES_DIR}")
        return

    print(f"[INFO] Found {len(image_paths)} image(s) in {cfg.IMAGES_DIR}")
    skipped = 0

    for img_path in tqdm(image_paths, desc="Polar transform", unit="img"):
        stem = img_path.stem   # e.g. "1491"

        # ── Load image (handles 8-bit and 16-bit) ─────────────────────────────
        raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            tqdm.write(f"  [WARN] Cannot read {img_path.name}, skipping.")
            skipped += 1
            continue

        # Normalise 16-bit to 8-bit for centre detection + polar transform
        if raw.dtype != np.uint8:
            lo, hi = float(raw.min()), float(raw.max())
            raw = ((raw.astype(np.float32) - lo) / (hi - lo + 1e-9) * 255
                   ).clip(0, 255).astype(np.uint8)

        # ── Load GT masks (V2 separate binary masks) ──────────────────────────
        # Sample mask:  V2/masks/sample/<stem>_sample.png
        # Pore mask:    V2/masks/pores/<stem>_pores.png
        s_path = cfg.SAMPLE_MASKS_DIR / f"{stem}_sample.png"
        p_path = cfg.PORE_MASKS_DIR   / f"{stem}_pores.png"

        sample_mask = (cv2.imread(str(s_path), cv2.IMREAD_GRAYSCALE)
                       if s_path.exists() else None)
        pore_mask   = (cv2.imread(str(p_path), cv2.IMREAD_GRAYSCALE)
                       if p_path.exists() else None)

        if sample_mask is None and pore_mask is None:
            tqdm.write(f"  [INFO] No GT masks found for {stem} "
                       f"— polar image will be saved without masks.")

        # ── Apply polar transform (identical params to image + both masks) ────
        polar_img, polar_sample, polar_pores, det_method = polar_transform(
            raw, sample_mask, pore_mask
        )

        if det_method != "minEnclosingCircle":
            tqdm.write(f"  [WARN] {stem}: centre via {det_method}")

        # ── Save polar image (uint8 0-255) ────────────────────────────────────
        cv2.imwrite(
            str(cfg.POLAR_IMAGES_DIR / f"{stem}_polar.png"),
            (polar_img * 255).clip(0, 255).astype(np.uint8),
        )

        # ── Save polar masks (binary 0/255) ───────────────────────────────────
        if polar_sample is not None:
            cv2.imwrite(
                str(cfg.POLAR_SAMPLE_MASKS_DIR / f"{stem}_polar_sample.png"),
                polar_sample,
            )
        if polar_pores is not None:
            cv2.imwrite(
                str(cfg.POLAR_PORE_MASKS_DIR / f"{stem}_polar_pores.png"),
                polar_pores,
            )

        # ── QA visualisation ──────────────────────────────────────────────────
        if save_vis_flag:
            save_vis(raw, polar_img, polar_sample, polar_pores,
                     cfg.POLAR_VIS_DIR / f"{stem}_polar_vis.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[DONE]")
    print(f"  Polar images        -> {cfg.POLAR_IMAGES_DIR}")
    print(f"  Polar sample masks  -> {cfg.POLAR_SAMPLE_MASKS_DIR}")
    print(f"  Polar pore masks    -> {cfg.POLAR_PORE_MASKS_DIR}")
    if save_vis_flag:
        print(f"  QA visuals          -> {cfg.POLAR_VIS_DIR}")
    if skipped:
        print(f"  Skipped (unreadable): {skipped}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="V2 polar preprocessing — converts images and both GT masks."
    )
    ap.add_argument("--no-vis", action="store_true",
                    help="Skip saving QA visualisation images.")
    args = ap.parse_args()
    main(save_vis_flag=not args.no_vis)
