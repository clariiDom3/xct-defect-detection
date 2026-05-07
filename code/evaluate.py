"""
evaluate.py
===========
Evaluates prediction masks against GT masks and produces CSV reports
and a summary printed to the console.

What it compares
----------------
  Predicted sample mask  (PRED_SAMPLE_DIR/<stem>_pred_sample.png)
  vs
  GT sample mask         (POLAR_SAMPLE_MASKS_DIR/<stem>_polar_sample.png)

  Predicted pore mask    (PRED_PORE_DIR/<stem>_pred_pores.png)
  vs
  GT pore mask           (POLAR_PORE_MASKS_DIR/<stem>_polar_pores.png)

  Both comparisons are at the FULL IMAGE level (not patch level).
  predict.py has already reassembled patches back into full images.

Metrics computed per image
--------------------------
  IoU       Intersection over Union  (= Jaccard index)
  Dice      2×TP / (2×TP + FP + FN)  (= F1 over pixels)
  Precision TP / (TP + FP)
  Recall    TP / (TP + FN)
  F1        same as Dice for binary segmentation
  TP, FP, FN, TN   raw pixel counts

Outputs  (saved to config_dl.EVAL_DIR)
-------
  evaluation_sample.csv     per-image metrics for sample detection
  evaluation_pores.csv      per-image metrics for pore detection
  evaluation_summary.csv    mean ± std across all images, both classes

Usage
-----
  python evaluate.py

Run AFTER predict.py has produced the prediction masks.

Dependencies
------------
  pip install numpy pandas opencv-python tqdm
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config_dl as cfg


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute pixel-level binary segmentation metrics.

    Both inputs are expected as uint8 (0 = negative, >0 = positive).
    All pixel counts and ratios are computed at full image resolution.

    Returns a dict with keys:
      TP, FP, FN, TN  — raw pixel counts
      IoU             — Intersection over Union
      Dice            — Dice coefficient (= F1)
      Precision       — TP / (TP + FP)
      Recall          — TP / (TP + FN)
      F1              — same as Dice for binary segmentation
    """
    p   = (pred > 0)
    g   = (gt   > 0)
    TP  = int(np.logical_and( p,  g).sum())
    FP  = int(np.logical_and( p, ~g).sum())
    FN  = int(np.logical_and(~p,  g).sum())
    TN  = int(np.logical_and(~p, ~g).sum())
    eps = 1e-9

    iou  = TP / (TP + FP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    prec = TP / (TP + FP + eps)
    rec  = TP / (TP + FN + eps)

    # Edge case: GT is empty (no foreground pixels in this image)
    # If pred is also empty → perfect by convention.
    # If pred has positives  → zero precision (all false alarms).
    if not g.any():
        iou = rec = dice = (1.0 if not p.any() else 0.0)
        prec = (1.0 if not p.any() else 0.0)

    return dict(
        TP=TP, FP=FP, FN=FN, TN=TN,
        IoU      = round(iou,  4),
        Dice     = round(dice, 4),
        Precision= round(prec, 4),
        Recall   = round(rec,  4),
        F1       = round(dice, 4),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def evaluate():
    cfg.EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect images that have both prediction and GT ───────────────────────
    pred_sample_paths = sorted(cfg.PRED_SAMPLE_DIR.glob("*_pred_sample.png"))
    if not pred_sample_paths:
        print(f"[ERROR] No prediction masks found in {cfg.PRED_SAMPLE_DIR}\n"
              "        Run  python predict.py  first.")
        return

    rows_sample = []
    rows_pores  = []

    for pred_s_path in tqdm(pred_sample_paths, desc="Evaluating", unit="img"):
        # Derive stem and sibling paths
        # pred_s_path:  <stem>_pred_sample.png
        stem = pred_s_path.name.replace("_pred_sample.png", "")

        pred_p_path = cfg.PRED_PORE_DIR / f"{stem}_pred_pores.png"
        gt_s_path   = cfg.POLAR_SAMPLE_MASKS_DIR / f"{stem}_polar_sample.png"
        gt_p_path   = cfg.POLAR_PORE_MASKS_DIR   / f"{stem}_polar_pores.png"

        # ── Load prediction masks ─────────────────────────────────────────────
        pred_s = cv2.imread(str(pred_s_path), cv2.IMREAD_GRAYSCALE)
        if pred_s is None:
            tqdm.write(f"  [WARN] Cannot read {pred_s_path.name}")
            continue

        pred_p = (cv2.imread(str(pred_p_path), cv2.IMREAD_GRAYSCALE)
                  if pred_p_path.exists() else None)

        # ── Load GT masks ─────────────────────────────────────────────────────
        gt_s = (cv2.imread(str(gt_s_path), cv2.IMREAD_GRAYSCALE)
                if gt_s_path.exists() else None)
        gt_p = (cv2.imread(str(gt_p_path), cv2.IMREAD_GRAYSCALE)
                if gt_p_path.exists() else None)

        # ── Resize GT to prediction size if they differ ───────────────────────
        # (can happen if the GT was generated from the original image and
        #  the polar transform changed the resolution slightly)
        def _align(src, ref):
            if src is None or ref is None:
                return src
            if src.shape != ref.shape:
                return cv2.resize(src, (ref.shape[1], ref.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            return src

        gt_s = _align(gt_s, pred_s)
        gt_p = _align(gt_p, pred_p)

        # ── Compute metrics ───────────────────────────────────────────────────
        if gt_s is not None:
            rows_sample.append({
                "image": stem,
                **pixel_metrics(pred_s, gt_s)
            })
        else:
            tqdm.write(f"  [INFO] No GT sample mask for {stem} — skipping sample eval.")

        if pred_p is not None and gt_p is not None:
            rows_pores.append({
                "image": stem,
                **pixel_metrics(pred_p, gt_p)
            })
        elif pred_p is None:
            tqdm.write(f"  [INFO] No pore prediction for {stem}.")
        else:
            tqdm.write(f"  [INFO] No GT pore mask for {stem} — skipping pore eval.")

    # ── Save per-image CSVs ───────────────────────────────────────────────────
    metric_cols = ["IoU", "Dice", "Precision", "Recall", "F1"]

    if not rows_sample and not rows_pores:
        print("[WARN] No evaluations completed — check that GT masks exist in:\n"
              f"  {cfg.POLAR_SAMPLE_MASKS_DIR}\n"
              f"  {cfg.POLAR_PORE_MASKS_DIR}")
        return

    summary_rows = []

    for label, rows in [("sample", rows_sample), ("pores", rows_pores)]:
        if not rows:
            print(f"  [WARN] No {label} evaluations — skipping.")
            continue

        df = pd.DataFrame(rows)
        out_path = cfg.EVAL_DIR / f"evaluation_{label}.csv"
        df.to_csv(out_path, index=False, float_format="%.4f")
        print(f"  {label} metrics -> {out_path}  ({len(df)} images)")

        row = {"class": label}
        for col in metric_cols:
            row[f"{col}_mean"] = round(df[col].mean(), 4)
            row[f"{col}_std"]  = round(df[col].std(),  4)
        summary_rows.append(row)

    # ── Save summary CSV ──────────────────────────────────────────────────────
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        sum_path = cfg.EVAL_DIR / "evaluation_summary.csv"
        df_sum.to_csv(sum_path, index=False, float_format="%.4f")

        # ── Pretty-print to console ───────────────────────────────────────────
        w = 58
        print("\n" + "=" * w)
        print("  EVALUATION SUMMARY — CBAM Single Model")
        print("=" * w)
        print(f"  {'Class':<10} {'IoU':>6} {'Dice':>6} "
              f"{'Prec':>6} {'Rec':>6} {'F1':>6}")
        print("  " + "-" * (w - 2))
        for _, r in df_sum.iterrows():
            print(f"  {r['class']:<10} "
                  f"{r['IoU_mean']:>6.4f} "
                  f"{r['Dice_mean']:>6.4f} "
                  f"{r['Precision_mean']:>6.4f} "
                  f"{r['Recall_mean']:>6.4f} "
                  f"{r['F1_mean']:>6.4f}")
        print("=" * w)
        print(f"\n  Summary -> {sum_path}")


if __name__ == "__main__":
    evaluate()
