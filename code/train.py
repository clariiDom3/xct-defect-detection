"""
train.py
========
Training script for single CBAMSegNet model.

The model predicts both sample and pore masks simultaneously from one
forward pass.  Two losses are computed and summed:

  total_loss = loss_sample + loss_pores

  loss_sample : BCE (uniform weights) + Dice
  loss_pores  : BCE (pos_weight=10)   + Dice
                High pos_weight because pores are rare (~5% of sample
                area) — without it the model collapses to predicting
                nothing for the pore head.

Training details
----------------
  - AdamW optimiser, cosine LR annealing (LR -> 0 over NUM_EPOCHS)
  - Early stopping on val pore IoU (the harder task; sample IoU
    saturates quickly and is less informative for stopping)
  - Gradient clipping (max_norm=1.0) prevents exploding gradients
  - One checkpoint saved: best val pore IoU -> CKPT_PATH

Usage
-----
  python train.py

Dependencies
------------
  pip install torch torchvision tqdm pandas
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_dl as cfg
from dataset import PolarPatchDataset
from model   import build_model, count_parameters


# ── Checkpoint path (single model) ───────────────────────────────────────────
CKPT_PATH = cfg.CBAM_DIR / "checkpoints" / "best.pth"


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    BCE-with-logits + Dice loss for one binary segmentation head.

    BCE  trains pixel accuracy (handles global class distribution).
    Dice trains overlap directly (invariant to class imbalance, optimises
         the same metric we report at evaluation time).

    pos_weight : upweights the positive class in BCE.
                 Set to ~10 for the pore head (rare class).
                 None for the sample head (class is roughly balanced).
    """

    def __init__(self,
                 pos_weight: torch.Tensor = None,
                 smooth:     float        = 1.0):
        super().__init__()
        self.bce    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.smooth = smooth

    def forward(self,
                logits:  torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        prob = torch.sigmoid(logits)
        inter = (prob * targets).sum()
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (
            prob.sum() + targets.sum() + self.smooth
        )
        # Equal weighting of BCE and Dice (cfg.BCE_WEIGHT / cfg.DICE_WEIGHT)
        return cfg.BCE_WEIGHT * bce_loss + cfg.DICE_WEIGHT * dice_loss


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def batch_iou(logits: torch.Tensor,
              targets: torch.Tensor,
              threshold: float) -> float:
    """Mean IoU over a batch (binary)."""
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).clamp(0, 1).sum(dim=(1, 2, 3))
    return ((inter + 1e-9) / (union + 1e-9)).mean().item()


# ══════════════════════════════════════════════════════════════════════════════
# Data split
# ══════════════════════════════════════════════════════════════════════════════

def split_image_paths():
    """
    Split polar image paths into train / val / test at the IMAGE level.
    Splitting by image (not by patch) prevents data leakage: patches from
    the same scan must not appear in both train and val.
    """
    paths = sorted(cfg.POLAR_IMAGES_DIR.glob("*_polar.png"))
    if not paths:
        raise FileNotFoundError(
            f"No polar images found in {cfg.POLAR_IMAGES_DIR}\n"
            "Run  python 01_preprocessing_v2.py  first."
        )
    rng = random.Random(cfg.SEED)
    rng.shuffle(paths)
    n      = len(paths)
    n_test = max(1, int(n * cfg.TEST_SPLIT))
    n_val  = max(1, int(n * cfg.VAL_SPLIT))
    test   = paths[:n_test]
    val    = paths[n_test: n_test + n_val]
    train  = paths[n_test + n_val:]
    print(f"[Split] {n} images -> "
          f"train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# One epoch
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model:        nn.Module,
              loader:       DataLoader,
              loss_sample:  CombinedLoss,
              loss_pores:   CombinedLoss,
              optimizer:    torch.optim.Optimizer,
              device:       torch.device,
              train:        bool = True) -> dict:
    """
    Run one full epoch (train or eval).

    Each batch contains:
      image_t  : (B, 1, H, W) normalised grayscale patch
      sample_t : (B, 1, H, W) binary GT sample mask
      pore_t   : (B, 1, H, W) binary GT pore mask

    Returns dict with keys: loss, loss_s, loss_p, iou_s, iou_p
    """
    model.train() if train else model.eval()

    tot_loss = tot_ls = tot_lp = tot_is = tot_ip = 0.0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for image_t, sample_t, pore_t in loader:
            image_t  = image_t.to(device)
            sample_t = sample_t.to(device)
            pore_t   = pore_t.to(device)

            sample_logit, pore_logit = model(image_t)

            ls = loss_sample(sample_logit, sample_t)
            lp = loss_pores (pore_logit,   pore_t)
            loss = ls + lp

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            tot_loss += loss.item()
            tot_ls   += ls.item()
            tot_lp   += lp.item()
            tot_is   += batch_iou(sample_logit, sample_t, cfg.SAMPLE_THRESHOLD)
            tot_ip   += batch_iou(pore_logit,   pore_t,   cfg.PORE_THRESHOLD)
            n        += 1

    nb = max(n, 1)
    return dict(loss   = tot_loss / nb,
                loss_s = tot_ls   / nb,
                loss_p = tot_lp   / nb,
                iou_s  = tot_is   / nb,
                iou_p  = tot_ip   / nb)


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train.py] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_paths, val_paths, _ = split_image_paths()

    print("\n[Datasets] Building training dataset ...")
    train_ds = PolarPatchDataset(train_paths, augment=True)
    print("[Datasets] Building validation dataset ...")
    val_ds   = PolarPatchDataset(val_paths,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(pretrained=True).to(device)
    print(f"\n[Model] Trainable parameters: {count_parameters(model):,}")

    # ── Loss functions ────────────────────────────────────────────────────────
    # Sample loss: uniform BCE weight (ring material is ~30% of image)
    loss_sample = CombinedLoss(pos_weight=None).to(device)

    # Pore loss: pos_weight=10 because pores are ~5% of sample area.
    # Without upweighting the pore head collapses to predicting nothing.
    pore_pw     = torch.tensor([cfg.PORE_POS_WEIGHT], device=device)
    loss_pores  = CombinedLoss(pos_weight=pore_pw).to(device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.LEARNING_RATE,
        weight_decay = cfg.WEIGHT_DECAY,
    )
    # Cosine annealing: LR decays from LEARNING_RATE to 1e-6 over NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    best_val_iou_p = 0.0   # early stopping monitors pore IoU (harder task)
    patience_ctr   = 0
    log_rows       = []

    print(f"\n{'='*70}")
    print(f"  Training CBAMSegNet  "
          f"(max {cfg.NUM_EPOCHS} epochs, patience {cfg.PATIENCE})")
    print(f"{'='*70}")
    header = (f"  {'Ep':>4} | "
              f"{'Tr loss':>8} {'Tr IoU-S':>9} {'Tr IoU-P':>9} | "
              f"{'Va loss':>8} {'Va IoU-S':>9} {'Va IoU-P':>9} | "
              f"{'LR':>8}")
    print(header)
    print("  " + "-" * 68)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        tr = run_epoch(model, train_loader, loss_sample, loss_pores,
                       optimizer, device, train=True)
        va = run_epoch(model, val_loader,   loss_sample, loss_pores,
                       optimizer, device, train=False)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  {epoch:>4} | "
            f"{tr['loss']:>8.4f} {tr['iou_s']:>9.4f} {tr['iou_p']:>9.4f} | "
            f"{va['loss']:>8.4f} {va['iou_s']:>9.4f} {va['iou_p']:>9.4f} | "
            f"{lr:>8.2e}   {time.time()-t0:.0f}s"
        )

        log_rows.append(dict(
            epoch      = epoch,
            train_loss = tr["loss"], train_iou_s = tr["iou_s"],
            train_iou_p= tr["iou_p"],
            val_loss   = va["loss"], val_iou_s   = va["iou_s"],
            val_iou_p  = va["iou_p"], lr         = lr,
        ))

        # ── Checkpoint (best val pore IoU) ────────────────────────────────────
        if va["iou_p"] > best_val_iou_p:
            best_val_iou_p = va["iou_p"]
            patience_ctr   = 0
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_iou_s":  va["iou_s"],
                "val_iou_p":  va["iou_p"],
                "optimizer":  optimizer.state_dict(),
            }, CKPT_PATH)
            print(f"        ✓ checkpoint saved  "
                  f"(val IoU-sample={va['iou_s']:.4f}  "
                  f"val IoU-pores={best_val_iou_p:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.PATIENCE:
                print(f"\n  Early stopping — "
                      f"{patience_ctr} epochs without pore IoU improvement.")
                break

    # ── Save training log ─────────────────────────────────────────────────────
    log_path = cfg.LOG_DIR / "training_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False, float_format="%.6f")

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"  Best val IoU — sample: {va['iou_s']:.4f}   pores: {best_val_iou_p:.4f}")
    print(f"  Checkpoint -> {CKPT_PATH}")
    print(f"  Log        -> {log_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
