"""
01_cbam_inference.py
====================
Runs CBAM inference on all raw .tif slices and saves predictions in
RING SPACE (same geometry as the original images and the CV masks).

Uses the EXACT same preprocessing chain as 01_preprocessing_dl.py:
  warp_polar (skimage) -> radial crop -> transpose -> row crop

This guarantees the patches seen by the model during inference are
identical in format to the patches seen during training.

Seamless inverse mapping
------------------------
After CBAM predicts probabilities in the cropped-transposed polar space,
we map each pixel in the ring image analytically back to its polar
coordinates using atan2, then bilinear-interpolate the probability.

For a ring pixel at (py, px):
  dx      = px - col_c
  dy      = py - row_c
  angle   = atan2(dy, dx)  mapped to [0, 2pi)
  r       = sqrt(dx^2 + dy^2)

  In the (n_angles, n_radii) pre-crop polar image:
    angle_idx = angle / (2*pi) * n_angles
    r_idx     = r / ring_radius * n_radii

  After radial crop (remove cols 0..left_col-1 and right_col+1..end):
    r_idx_cropped = r_idx - left_col

  After transpose (row<->col swap):
    map_row = r_idx_cropped     (radial axis becomes row)
    map_col = angle_idx         (angular axis becomes col)

  After row crop (remove rows 0..top-1 and bottom+1..end):
    map_row_final = map_row - top

  Read: prediction[map_row_final, map_col] via bilinear interpolation.

This is a continuous mapping with no seam, no gaps, no morphological
guessing. It is the correct inverse of the skimage warp_polar transform.

Usage
-----
  python 01_cbam_inference.py

Place in:  reconstruction/
Requires:  config_recon.py in the same folder
           DL/CBAM/model.py and DL/CBAM/cbam.py reachable via sys.path

Dependencies
------------
  pip install torch scikit-image scipy opencv-python numpy tqdm
"""

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from skimage.filters import threshold_otsu
from skimage.transform import warp_polar
from tqdm import tqdm

import config_recon as cfg

sys.path.insert(0, str(cfg.CBAM_DIR))
warnings.filterwarnings("ignore")


# Load helpers

def load_uint8(path: Path) -> np.ndarray:
    """Load any bit-depth grayscale image and return uint8 [0,255]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img.astype(np.float32) - lo) / (hi - lo + 1e-9) * 255
               ).clip(0, 255).astype(np.uint8)
    return img


# Exact preprocessing chain  (copied from 01_preprocessing_dl.py)

def detect_center(gray: np.ndarray):
    """
    Identical to 01_preprocessing_dl.py:detect_center.
    Returns (row_c, col_c, ring_radius, method).
    """
    h, w = gray.shape

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
            cx2, cy2    = centroids[interior[0][1]]
            row_c, col_c = int(round(cy2)), int(round(cx2))
            rad          = float(min(row_c, h - row_c, col_c, w - col_c))
            return row_c, col_c, rad, "dark-hole"
    except Exception:
        pass

    return h // 2, w // 2, float(min(h, w)) / 2.0, "image-center-fallback"


def find_radial_crop_bounds(col_sums: np.ndarray,
                             fraction: float = cfg.CROP_FRACTION):
    """Identical to 01_preprocessing_dl.py."""
    threshold = col_sums.max() * fraction
    hits      = np.where(col_sums > threshold)[0]
    if len(hits) == 0:
        return 0, len(col_sums) - 1
    return int(hits[0]), int(hits[-1])


def find_tight_row_bounds(img_transposed: np.ndarray,
                           padding: float = 0.02,
                           smooth_sigma: float = 5.0):
    """Identical to 01_preprocessing_dl.py."""
    n_rows  = img_transposed.shape[0]
    pad_px  = max(1, int(n_rows * padding))
    profile = img_transposed.mean(axis=1)
    smooth  = gaussian_filter1d(profile.astype(np.float64), sigma=smooth_sigma)
    grad    = np.gradient(smooth)

    top_row    = int(np.argmax(grad))
    bottom_row = int(np.argmin(grad))

    if top_row >= bottom_row:
        half_max = smooth.max() / 2.0
        hits     = np.where(smooth > half_max)[0]
        if len(hits) >= 2:
            top_row, bottom_row = int(hits[0]), int(hits[-1])
        else:
            top_row, bottom_row = 0, n_rows - 1

    top    = max(0,          top_row    - pad_px)
    bottom = min(n_rows - 1, bottom_row + pad_px)
    return top, bottom


def forward_transform(gray: np.ndarray):
    """
    Apply the exact same 4-step preprocessing as 01_preprocessing_dl.py.

    Returns
    -------
    polar_img : float64 [0,1] shape (material_H, n_angles)
                Exactly what the model was trained on.
    params    : dict with all transform parameters needed for the inverse.
    """
    h, w = gray.shape

    # Step 0: centre detection
    row_c, col_c, ring_radius, method = detect_center(gray)
    n_angles = int(2 * np.pi * ring_radius)
    n_radii  = int(ring_radius)

    # Step 1: normalise + warp_polar
    img = gray.astype(np.float64)
    if img.max() > 0:
        img = img / img.max()

    polar = warp_polar(
        img,
        center       = (row_c, col_c),
        radius       = ring_radius,
        output_shape = (n_angles, n_radii),
        order        = 3,
        preserve_range = True,
    )
    # polar shape: (n_angles, n_radii)

    # Step 2: radial crop
    col_sums = np.sum(polar, axis=0)
    left_col, right_col = find_radial_crop_bounds(col_sums)
    polar = polar[:, left_col : right_col + 1]
    # polar shape: (n_angles, n_radii_cropped)

    # Step 3: transpose
    polar = polar.T
    # polar shape: (n_radii_cropped, n_angles)

    # Step 4: row crop
    top, bottom = find_tight_row_bounds(polar)
    polar = polar[top : bottom + 1, :]
    # polar shape: (material_H, n_angles)

    params = dict(
        row_c       = row_c,
        col_c       = col_c,
        ring_radius = ring_radius,
        n_angles    = n_angles,
        n_radii     = n_radii,
        left_col    = left_col,
        right_col   = right_col,
        top         = top,
        bottom      = bottom,
        orig_h      = h,
        orig_w      = w,
    )
    return polar, params


# Seamless analytical inverse mapping

def seamless_inverse(prob_map: np.ndarray, params: dict) -> np.ndarray:
    """
    Map each pixel (py, px) in the original ring image to its coordinates
    in the CBAM prediction map using the analytical inverse of the
    preprocessing transform.

    The mapping is derived by tracing each ring pixel through all 4
    preprocessing steps in reverse:

      (py, px) in ring
        -> angle, r in polar
        -> (angle_idx, r_idx) in (n_angles, n_radii) pre-crop polar
        -> (angle_idx, r_idx - left_col) after radial crop
        -> (r_idx - left_col, angle_idx) after transpose
        -> (r_idx - left_col - top, angle_idx) after row crop
        = (map_row, map_col) in the prediction image

    Bilinear interpolation via cv2.remap reads the probability at that
    fractional coordinate. No seams, no gaps, no morphological filling.

    Parameters
    ----------
    prob_map : float32 (material_H, n_angles)  — CBAM probability output
    params   : dict from forward_transform()

    Returns
    -------
    ring_prob : float32 (orig_h, orig_w)  — probability in ring space
    """
    row_c       = params["row_c"]
    col_c       = params["col_c"]
    ring_radius = params["ring_radius"]
    n_angles    = params["n_angles"]
    n_radii     = params["n_radii"]
    left_col    = params["left_col"]
    top         = params["top"]
    orig_h      = params["orig_h"]
    orig_w      = params["orig_w"]

    pred_h, pred_w = prob_map.shape   # (material_H, n_angles)

    # Build pixel coordinate grids for the ring image
    py_grid, px_grid = np.mgrid[0:orig_h, 0:orig_w].astype(np.float64)

    # Displacements from ring centre
    # Note: skimage center=(row, col), so col_c is X and row_c is Y
    dx = px_grid - col_c
    dy = py_grid - row_c

    # Polar coordinates
    angle = np.arctan2(dy, dx)                    # (-pi, pi]
    angle = (angle + 2.0 * np.pi) % (2.0 * np.pi)  # [0, 2pi)
    r     = np.sqrt(dx ** 2 + dy ** 2)

    # Map to (n_angles, n_radii) pre-crop polar image coordinates
    angle_idx_f = (angle / (2.0 * np.pi)) * n_angles   # col in pred (after transpose)
    r_idx_f     = (r / ring_radius)       * n_radii     # row in pred (after transpose)

    # Apply radial crop offset (left_col)
    r_idx_cropped_f = r_idx_f - left_col

    # After transpose: row=r, col=angle.  After row crop: subtract top.
    map_row = (r_idx_cropped_f - top).astype(np.float32)   # row in pred
    map_col = angle_idx_f.astype(np.float32)               # col in pred

    # Bilinear interpolation: remap reads pred[map_row, map_col]
    ring_prob = cv2.remap(
        prob_map.astype(np.float32),
        map_col,                      # x-map  (columns)
        map_row,                      # y-map  (rows)
        interpolation = cv2.INTER_LINEAR,
        borderMode    = cv2.BORDER_CONSTANT,
        borderValue   = 0.0,
    )

    # Zero pixels outside the ring radius
    ring_prob[r > ring_radius] = 0.0
    return ring_prob


# Patch inference

def gaussian_window(size: int) -> np.ndarray:
    sigma  = size / 6.0
    coords = np.arange(size) - size / 2.0
    g1d    = np.exp(-0.5 * (coords / sigma) ** 2)
    w      = np.outer(g1d, g1d)
    return (w / w.max()).astype(np.float32)


def extract_patches(img: np.ndarray, ps: int, st: int):
    H, W = img.shape
    patches = []
    for r in range(0, max(1, H - ps + 1), st):
        for c in range(0, max(1, W - ps + 1), st):
            patches.append((img[r:r+ps, c:c+ps], r, c))
    return patches


def run_patches(polar: np.ndarray,
                model,
                device: torch.device,
                window: np.ndarray,
                ps: int,
                st: int,
                norm_mean: float = 0.485,
                norm_std:  float = 0.229):
    """
    Slice the polar image into overlapping ps×ps patches, run CBAM,
    and reassemble with Gaussian blending.

    The polar image may be smaller than ps in the height (radial) axis
    since the material band after cropping can be thin. In that case we
    pad to ps, run inference, then remove the padding from the result.

    Returns (s_prob, p_prob) float32, same shape as input polar.
    """
    orig_H, orig_W = polar.shape

    # Pad height to at least ps so we always have at least one full patch
    pad_h = max(0, ps - orig_H)
    pad_w = max(0, ps - orig_W)
    if pad_h or pad_w:
        polar = np.pad(polar, ((0, pad_h), (0, pad_w)), mode="reflect")
    Ph, Pw = polar.shape

    s_acc = np.zeros((Ph, Pw), dtype=np.float32)
    s_wt  = np.zeros((Ph, Pw), dtype=np.float32)
    p_acc = np.zeros((Ph, Pw), dtype=np.float32)
    p_wt  = np.zeros((Ph, Pw), dtype=np.float32)

    patches = extract_patches(polar, ps, st)
    MINI    = 16

    with torch.no_grad():
        for start in range(0, len(patches), MINI):
            batch = patches[start : start + MINI]
            imgs  = []
            for patch, r, c in batch:
                # Pad patch to exactly ps×ps if it landed at the edge
                ph, pw = patch.shape
                if ph < ps or pw < ps:
                    patch = np.pad(patch, ((0, ps-ph), (0, ps-pw)),
                                   mode="reflect")
                t = torch.from_numpy(
                    ((patch.astype(np.float32) - norm_mean) / norm_std)
                ).unsqueeze(0)
                imgs.append(t)

            bt         = torch.stack(imgs).to(device)
            sl, pl     = model(bt)
            s_prob_b   = torch.sigmoid(sl).squeeze(1).cpu().numpy()
            p_prob_b   = torch.sigmoid(pl).squeeze(1).cpu().numpy()

            for (_, r, c), sp, pp in zip(batch, s_prob_b, p_prob_b):
                s_acc[r:r+ps, c:c+ps] += sp * window
                s_wt [r:r+ps, c:c+ps] += window
                p_acc[r:r+ps, c:c+ps] += pp * window
                p_wt [r:r+ps, c:c+ps] += window

    s_map = np.where(s_wt > 1e-6, s_acc / s_wt, 0.0)
    p_map = np.where(p_wt > 1e-6, p_acc / p_wt, 0.0)

    # Remove padding
    return s_map[:orig_H, :orig_W], p_map[:orig_H, :orig_W]


# Infer one slice

def infer_slice(gray: np.ndarray,
                model,
                device: torch.device,
                window: np.ndarray) -> tuple:
    """
    Full inference pipeline for one ring image.

    1. Apply exact forward transform (same as training preprocessing).
    2. Run CBAM patch inference with Gaussian blending.
    3. Apply seamless analytical inverse mapping back to ring space.
    4. Threshold and clean up.

    Returns (sample_mask, pore_mask) as uint8 (0/255).
    """
    ps = cfg.PATCH_SIZE
    st = cfg.PATCH_STRIDE

    # 1. Forward transform (identical to training)
    polar, params = forward_transform(gray)
    # polar: float64 [0,1], shape (material_H, n_angles)

    # 2. Patch inference
    # The polar strip is circular in the angular (column) direction:
    # angle 0 and angle 2pi are the same physical point on the ring.
    # Without wrapping, columns near 0 and n_angles-1 are only covered
    # by the edge of one patch and receive very low Gaussian weight,
    # producing a low-probability gap that maps back to 3 o'clock.
    model.eval()
    polar_f = polar.astype(np.float32)
    pad_w   = ps   # one patch width is enough
    polar_padded = np.concatenate(
        [polar_f[:, -pad_w:], polar_f, polar_f[:, :pad_w]], axis=1
    )
    s_map_pad, p_map_pad = run_patches(
        polar_padded, model, device, window, ps, st
    )
    # Strip the circular padding — keep only the original n_angles columns
    s_map = s_map_pad[:, pad_w : pad_w + polar_f.shape[1]]
    p_map = p_map_pad[:, pad_w : pad_w + polar_f.shape[1]]
    # s_map, p_map: float32 [0,1], shape (material_H, n_angles)

    # 3. Seamless analytical inverse → ring space
    s_ring = seamless_inverse(s_map, params)   # float32 (orig_h, orig_w)
    p_ring = seamless_inverse(p_map, params)

    # 4. Threshold
    sample_mask = (s_ring >= cfg.SAMPLE_THRESHOLD).astype(np.uint8) * 255
    pore_mask   = (p_ring >= cfg.PORE_THRESHOLD  ).astype(np.uint8) * 255

    # Minimum area filter: remove pore noise
    MIN_PORE_PX = 20
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(pore_mask)
    clean = np.zeros_like(pore_mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_PORE_PX:
            clean[lbl == i] = 255
    pore_mask = clean

    # Pores must be inside sample
    pore_mask = cv2.bitwise_and(pore_mask, pore_mask, mask=sample_mask)

    return sample_mask, pore_mask


# Main

def main():
    cfg.CBAM_SAMPLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CBAM_PORE_OUT_DIR.mkdir(  parents=True, exist_ok=True)

    # Load model
    if not cfg.CBAM_CKPT.exists():
        raise FileNotFoundError(
            f"CBAM checkpoint not found: {cfg.CBAM_CKPT}\n"
            "Run python train.py in DL/CBAM/ first."
        )

    from model import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = build_model(pretrained=False).to(device)
    ck    = torch.load(cfg.CBAM_CKPT, map_location=device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"[INFO] CBAM loaded  "
          f"(val IoU-sample={ck['val_iou_s']:.4f}  "
          f"val IoU-pores={ck['val_iou_p']:.4f})")

    window = gaussian_window(cfg.PATCH_SIZE)

    # Collect slices
    slice_paths = sorted(
        cfg.SLICES_DIR.glob(f"{cfg.SLICE_PREFIX}*{cfg.SLICE_SUFFIX}")
    )
    if not slice_paths:
        raise FileNotFoundError(
            f"No slices found in {cfg.SLICES_DIR}\n"
            f"Looking for: {cfg.SLICE_PREFIX}*{cfg.SLICE_SUFFIX}"
        )
    print(f"[INFO] Found {len(slice_paths)} slices.")

    # Resume: skip already-processed slices
    to_process = []
    for p in slice_paths:
        stem  = p.stem
        s_out = cfg.CBAM_SAMPLE_OUT_DIR / f"{stem}_sample.png"
        p_out = cfg.CBAM_PORE_OUT_DIR   / f"{stem}_pores.png"
        if not s_out.exists() or not p_out.exists():
            to_process.append(p)

    if not to_process:
        print("[INFO] All slices already processed.")
        return
    '''test'''
    #to_process = to_process[:3]
    '''test'''
    print(f"[INFO] {len(to_process)} slices to process "
          f"({len(slice_paths) - len(to_process)} already done).")

    errors = []
    for slice_path in tqdm(to_process, desc="CBAM inference", unit="slice"):
        stem = slice_path.stem
        try:
            gray = load_uint8(slice_path)
            sample_mask, pore_mask = infer_slice(gray, model, device, window)
            cv2.imwrite(str(cfg.CBAM_SAMPLE_OUT_DIR / f"{stem}_sample.png"),
                        sample_mask)
            cv2.imwrite(str(cfg.CBAM_PORE_OUT_DIR   / f"{stem}_pores.png"),
                        pore_mask)
        except Exception as e:
            tqdm.write(f"  [ERROR] {stem}: {e}")
            errors.append(stem)

    print(f"\n[DONE]  Processed: {len(to_process) - len(errors)}"
          f"  Errors: {len(errors)}")
    if errors:
        print("  Failed:", errors)
    print(f"  Sample -> {cfg.CBAM_SAMPLE_OUT_DIR}")
    print(f"  Pores  -> {cfg.CBAM_PORE_OUT_DIR}")


if __name__ == "__main__":
    main()
