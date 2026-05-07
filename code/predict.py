"""
predict.py
==========
Runs inference on all polar images using the trained two-stage
CBAM segmentation pipeline.

Two-stage inference
-------------------
  Stage 1  :  image patch (1 ch)                  -> sample probability map
  Stage 2  :  image patch (1 ch) +
              Stage-1 sample probability (1 ch)    -> pore probability map

Patch reassembly — Gaussian blending
-------------------------------------
Each polar image is divided into overlapping patches (PATCH_SIZE,
PATCH_STRIDE from config_dl.py).  Every pixel may be covered by
multiple overlapping patches.  Naive averaging of overlapping
predictions creates visible seams at patch boundaries because the
model is less confident near edges.

Gaussian blending fixes this:
  - Each patch prediction is multiplied by a 2-D Gaussian weight
    window (sigma = PATCH_SIZE / 6, peak at centre).
  - Weighted predictions are accumulated into a full-size probability
    map together with the accumulated weights.
  - Final probability = sum(weighted_preds) / sum(weights)

This gives maximum weight to the patch centre (most reliable) and
smoothly down-weights the edges, making boundary artefacts invisible.

Example — 512×2048 polar image, PATCH_SIZE=256, PATCH_STRIDE=128:
  Rows of patches : 3
  Cols of patches : 15
  Total patches   : 45  (processed as micro-batches for speed)
  Output          : two full-resolution probability maps, thresholded
                    to binary masks and saved as uint8 PNG (0/255)

Outputs  (saved to config_dl.PRED_SAMPLE_DIR / PRED_PORE_DIR)
  <stem>_pred_sample.png   binary 0/255
  <stem>_pred_pores.png    binary 0/255

Usage
-----
  python predict.py                   # run on all polar images
  python predict.py --no-vis          # skip QA overlay images
  python predict.py --threshold-s 0.5 --threshold-p 0.4
                                      # override sigmoid thresholds

Dependencies
------------
  pip install torch opencv-python numpy tqdm
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_dl as cfg
from dataset import PolarInferenceDataset
# model is imported inside predict() after checkpoint check


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian weight window
# ══════════════════════════════════════════════════════════════════════════════

def _gaussian_window(size: int) -> np.ndarray:
    """
    2-D Gaussian weight window of shape (size, size).

    Peak = 1.0 at centre, falls smoothly to near-zero at edges.
    sigma = size / 6  ensures the weights reach ~0.01 at the corners,
    giving very low weight to edge predictions while keeping full weight
    at the centre of each patch.

    Used during patch reassembly so boundary artefacts are invisible.
    """
    sigma  = size / 6.0
    coords = np.arange(size) - size / 2.0
    gauss1d = np.exp(-0.5 * (coords / sigma) ** 2)
    window  = np.outer(gauss1d, gauss1d)   # (size, size)
    return (window / window.max()).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Reassembly helper
# ══════════════════════════════════════════════════════════════════════════════

def _make_accumulators(n_images: int,
                       metas:    list,
                       device:   str = "cpu"):
    """
    Pre-allocate one (prob_sum, weight_sum) pair per image for Gaussian
    blending reassembly.  Arrays are indexed by img_idx from the meta dict.

    Returns
    -------
    prob_acc   : list of np.float32 arrays, shape (pad_H, pad_W)
    weight_acc : list of np.float32 arrays, shape (pad_H, pad_W)
    shapes     : list of (orig_H, orig_W) for final crop
    """
    # Determine padded shape for each image from the first patch of that image
    seen     = {}
    shapes   = {}
    for m in metas:
        idx = m["img_idx"]
        if idx not in seen:
            seen[idx]   = (m["pad_H"],  m["pad_W"])
            shapes[idx] = (m["orig_H"], m["orig_W"])

    prob_acc   = [np.zeros(seen[i],   dtype=np.float32) for i in range(n_images)]
    weight_acc = [np.zeros(seen[i],   dtype=np.float32) for i in range(n_images)]
    orig_shapes = [shapes[i]                              for i in range(n_images)]
    return prob_acc, weight_acc, orig_shapes


def _accumulate(prob_acc:   list,
                weight_acc: list,
                probs:      np.ndarray,
                metas:      list,
                window:     np.ndarray):
    """
    Add one batch of patch predictions into the accumulators.

    probs : (B, H, W) float32 probability values in [0, 1]
    metas : list of B meta dicts from PolarInferenceDataset
    """
    for prob, meta in zip(probs, metas):
        idx = meta["img_idx"]
        r   = meta["row_start"]
        c   = meta["col_start"]
        ps  = cfg.PATCH_SIZE
        prob_acc  [idx][r:r + ps, c:c + ps] += prob   * window
        weight_acc[idx][r:r + ps, c:c + ps] +=          window


def _finalise(prob_acc:    list,
              weight_acc:  list,
              orig_shapes: list,
              threshold:   float) -> list:
    """
    Divide accumulated weighted probabilities by accumulated weights,
    crop to original image size, apply threshold, return binary masks.

    Returns list of uint8 (0/255) numpy arrays, one per image.
    """
    results = []
    for prob_sum, w_sum, (oh, ow) in zip(prob_acc, weight_acc, orig_shapes):
        prob     = np.where(w_sum > 1e-6, prob_sum / w_sum, 0.0)
        prob     = prob[:oh, :ow]                           # crop pad
        binary   = (prob >= threshold).astype(np.uint8) * 255
        results.append(binary)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main inference function
# ══════════════════════════════════════════════════════════════════════════════

def predict(threshold_s: float = cfg.SAMPLE_THRESHOLD,
            threshold_p: float = cfg.PORE_THRESHOLD,
            save_vis:    bool  = True,
            batch_size:  int   = cfg.BATCH_SIZE * 2):
    """
    Run two-stage inference on every polar image found in
    cfg.POLAR_IMAGES_DIR and save binary prediction masks.

    Parameters
    ----------
    threshold_s : sigmoid threshold for sample mask (default 0.50)
    threshold_p : sigmoid threshold for pore mask   (default 0.40)
    save_vis    : save a colour QA overlay image alongside each prediction
    batch_size  : patches per forward pass (higher = faster, more VRAM)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] Device: {device}")

    # checkpoint check is handled in the load block below

    # ── Load model ────────────────────────────────────────────────────────────
    from model import build_model
    CKPT_PATH = cfg.CBAM_DIR / "checkpoints" / "best.pth"
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}\n"
            "Run  python train.py  first."
        )
    model = build_model(pretrained=False).to(device)
    ck    = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"  Model loaded  "
          f"(val IoU-sample={ck['val_iou_s']:.4f}  "
          f"val IoU-pores={ck['val_iou_p']:.4f})")

    # ── Build inference dataset ───────────────────────────────────────────────
    img_paths = sorted(cfg.POLAR_IMAGES_DIR.glob("*_polar.png"))
    if not img_paths:
        print(f"[ERROR] No polar images in {cfg.POLAR_IMAGES_DIR}")
        return

    print(f"[predict] {len(img_paths)} image(s) found.")

    dataset = PolarInferenceDataset(img_paths,
                                    patch_size=cfg.PATCH_SIZE,
                                    stride=cfg.PATCH_STRIDE)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0,
                         collate_fn=lambda b: (
                             torch.stack([x[0] for x in b]),
                             [x[1] for x in b]
                         ))

    n       = len(img_paths)
    window  = _gaussian_window(cfg.PATCH_SIZE)

    # Allocate accumulators for sample and pore separately
    all_metas = [m for _, m in dataset]
    s_prob,  s_wt,  orig_shapes = _make_accumulators(n, all_metas)
    p_prob,  p_wt,  _           = _make_accumulators(n, all_metas)

    # ── Forward pass ─────────────────────────────────────────────────────────
    print("[predict] Running forward pass ...")
    with torch.no_grad():
        for patches, metas in tqdm(loader, desc="Patches", unit="batch"):
            patches = patches.to(device)    # (B, 1, H, W)

            # Single forward pass -> both heads
            s_logit, p_logit = model(patches)
            s_prob_batch = torch.sigmoid(s_logit).squeeze(1).cpu().numpy()
            p_prob_batch = torch.sigmoid(p_logit).squeeze(1).cpu().numpy()

            # Accumulate with Gaussian weights
            _accumulate(s_prob, s_wt, s_prob_batch, metas, window)
            _accumulate(p_prob, p_wt, p_prob_batch, metas, window)

    # ── Finalise + save ───────────────────────────────────────────────────────
    sample_masks = _finalise(s_prob, s_wt, orig_shapes, threshold_s)
    pore_masks   = _finalise(p_prob, p_wt, orig_shapes, threshold_p)

    cfg.PRED_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PRED_PORE_DIR.mkdir(  parents=True, exist_ok=True)
    if save_vis:
        cfg.VIS_DIR.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(img_paths):
        stem      = img_path.stem.replace("_polar", "")
        s_mask    = sample_masks[idx]
        p_mask    = pore_masks[idx]

        cv2.imwrite(str(cfg.PRED_SAMPLE_DIR / f"{stem}_pred_sample.png"), s_mask)
        cv2.imwrite(str(cfg.PRED_PORE_DIR   / f"{stem}_pred_pores.png"),  p_mask)

        if save_vis:
            gt_s_path = cfg.POLAR_SAMPLE_MASKS_DIR / f"{stem}_polar_sample.png"
            gt_p_path = cfg.POLAR_PORE_MASKS_DIR   / f"{stem}_polar_pores.png"
            gt_s = (cv2.imread(str(gt_s_path), cv2.IMREAD_GRAYSCALE)
                    if gt_s_path.exists() else None)
            gt_p = (cv2.imread(str(gt_p_path), cv2.IMREAD_GRAYSCALE)
                    if gt_p_path.exists() else None)
            _save_vis(img_path, gt_s, s_mask, gt_p, p_mask,
                      cfg.VIS_DIR / f"{stem}_pred_vis.png")

    print(f"\n[DONE]")
    print(f"  Sample predictions -> {cfg.PRED_SAMPLE_DIR}")
    print(f"  Pore predictions   -> {cfg.PRED_PORE_DIR}")
    if save_vis:
        print(f"  QA overlays        -> {cfg.VIS_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# QA visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _save_vis(polar_path:    Path,
              gt_sample:     np.ndarray | None,
              pred_sample:   np.ndarray,
              gt_pores:      np.ndarray | None,
              pred_pores:    np.ndarray,
              out_path:      Path):
    """
    5-row vertically stacked QA image.  Each row is the full polar image
    width so the spatial layout of the ring is clearly visible.

      Row 0 — Original polar image             (grayscale)
      Row 1 — GT sample mask over polar         (green,  alpha 0.45)
      Row 2 — Predicted sample mask over polar  (teal,   alpha 0.45)
      Row 3 — GT pore mask over polar           (red,    alpha 0.65)
      Row 4 — Predicted pore mask over polar    (orange, alpha 0.65)

    Rows 1 and 3 show "GT not available" if GT masks were not found.
    A label is burned into the top-left corner of every row so you can
    tell them apart without a legend.
    """
    raw = cv2.imread(str(polar_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return

    # Normalise to uint8
    raw = raw.astype(np.float32)
    lo, hi = raw.min(), raw.max()
    raw8 = ((raw - lo) / (hi - lo + 1e-9) * 255).clip(0, 255).astype(np.uint8)
    h, w  = raw8.shape

    def _align(mask):
        """Resize mask to (h, w) if needed."""
        if mask is None:
            return None
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    def blend(mask, bgr, alpha):
        """Overlay a binary mask onto raw8 with a solid colour."""
        out = cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR).astype(np.float32)
        m   = mask > 0
        for c, v in enumerate(bgr):
            out[:, :, c][m] = out[:, :, c][m] * (1 - alpha) + v * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    def empty_row(label):
        """Grey row with a text label — used when GT is unavailable."""
        row = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.putText(row, label, (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 2000),
                    (180, 180, 180), 2, cv2.LINE_AA)
        return row

    def add_label(img, text):
        """Tiny label — dark outline + white fill, no background box."""
        out  = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, text, (4, 13), font, 0.28, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, text, (4, 13), font, 0.28, (240, 240, 240), 1, cv2.LINE_AA)
        return out

    gt_s   = _align(gt_sample)
    pred_s = _align(pred_sample)
    gt_p   = _align(gt_pores)
    pred_p = _align(pred_pores)

    # ── Build 5 rows ──────────────────────────────────────────────────────────
    row0 = add_label(cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR),
                     "1 | Original polar image")

    row1 = (add_label(blend(gt_s,   ( 80, 200,  0), 0.45),
                      "2 | GT sample mask (green)")
            if gt_s is not None
            else empty_row("2 | GT sample mask — not available"))

    row2 = add_label(blend(pred_s, (212, 200,  0), 0.45),
                     "3 | Predicted sample mask (teal)")

    row3 = (add_label(blend(gt_p,   ( 48,  48, 255), 0.65),
                      "4 | GT pore mask (red)")
            if gt_p is not None
            else empty_row("4 | GT pore mask — not available"))

    row4 = add_label(blend(pred_p, (  0, 140, 255), 0.65),
                     "5 | Predicted pore mask (orange)")

    # ── Thin separator between rows ───────────────────────────────────────────
    sep = np.full((4, w, 3), 60, dtype=np.uint8)

    stacked = np.vstack([row0, sep, row1, sep, row2, sep, row3, sep, row4])
    cv2.imwrite(str(out_path), stacked)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Two-stage CBAM inference with Gaussian patch blending."
    )
    ap.add_argument("--no-vis",       action="store_true",
                    help="Skip QA overlay images.")
    ap.add_argument("--threshold-s",  type=float, default=cfg.SAMPLE_THRESHOLD,
                    help=f"Sample sigmoid threshold (default {cfg.SAMPLE_THRESHOLD})")
    ap.add_argument("--threshold-p",  type=float, default=cfg.PORE_THRESHOLD,
                    help=f"Pore sigmoid threshold   (default {cfg.PORE_THRESHOLD})")
    ap.add_argument("--batch-size",   type=int,   default=cfg.BATCH_SIZE * 2,
                    help="Patches per forward pass (default BATCH_SIZE×2)")
    args = ap.parse_args()

    predict(
        threshold_s = args.threshold_s,
        threshold_p = args.threshold_p,
        save_vis    = not args.no_vis,
        batch_size  = args.batch_size,
    )
