"""
config_dl.py
============
All hyper-parameters and paths for the two-stage CBAM U-Net pipeline.
Extends config_v2.py — every V2 path is available by importing this file.

Two-stage architecture
----------------------
  Stage 1  :  polar patch  (1 ch)                 -> sample mask
  Stage 2  :  polar patch  (1 ch)
              + Stage-1 predicted sample mask (1 ch)   -> pore mask

Patch strategy
--------------
  Each polar image is divided into overlapping square patches before
  being fed to the model.  At inference the per-patch predictions are
  blended back into a full-resolution mask using a Gaussian weight
  window so that patch boundaries are invisible in the final output.

  Example for a 512x2048 polar image with PATCH_SIZE=256, PATCH_STRIDE=128:
    Along height: ceil((512-256)/128)+1 = 3 rows
    Along width:  ceil((2048-256)/128)+1 = 15 cols
    Total patches per image: 3*15 = 45
    Overlap: 50%  (stride = patch_size / 2)
"""

import sys
from pathlib import Path

# ── Roots — derived from this file's location so paths are always correct ──────
# config_dl.py lives in  DL/CBAM/  — all outputs stay inside that folder.
CBAM_DIR = Path(__file__).resolve().parent   # DL/CBAM/
DL_ROOT  = CBAM_DIR.parent                  # DL/

# Re-export everything from config_v2 (source images, GT masks, splits, etc.)
_v2_dir = CBAM_DIR.parent.parent / "V2"
sys.path.insert(0, str(_v2_dir))
import config_v2 as _v2
from config_v2 import *   # re-exports IMAGES_DIR, SAMPLE_MASKS_DIR, PORE_MASKS_DIR …

# Override the polar paths so polar outputs land in DL/CBAM/polar/ instead of V2/
POLAR_DIR              = CBAM_DIR / "polar"
POLAR_IMAGES_DIR       = POLAR_DIR / "images"
POLAR_SAMPLE_MASKS_DIR = POLAR_DIR / "masks" / "sample"
POLAR_PORE_MASKS_DIR   = POLAR_DIR / "masks" / "pores"
POLAR_VIS_DIR          = POLAR_DIR / "vis"

# Official CBAM repo (cloned inside CBAM_DIR)
# Clone with:  git clone https://github.com/Jongchan/attention-module.git
CBAM_REPO_DIR = CBAM_DIR / "attention-module"

# ── Patch parameters ──────────────────────────────────────────────────────────
PATCH_SIZE   = 256   # square patch side (pixels)
PATCH_STRIDE = 128   # step between patches; 50% overlap keeps boundary artefacts minimal

# ── Model checkpoints ─────────────────────────────────────────────────────────
STAGE1_CKPT  = CBAM_DIR / "checkpoints" / "stage1_best.pth"
STAGE2_CKPT  = CBAM_DIR / "checkpoints" / "stage2_best.pth"

# ── Training ──────────────────────────────────────────────────────────────────
# Stage 1 (sample) trains first, then Stage 2 (pores) is trained with the
# predicted sample mask from a frozen Stage-1 model.

BATCH_SIZE     = 8    # patches per batch (increase if VRAM allows)
NUM_EPOCHS     = 80   # max epochs; early stopping exits earlier
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
PATIENCE       = 15   # epochs without val-IoU improvement before stopping

# Loss weighting  (loss = BCE_WEIGHT * bce  +  DICE_WEIGHT * dice)
# Equal weights work well; increase DICE_WEIGHT if IoU matters more.
BCE_WEIGHT  = 1.0
DICE_WEIGHT = 1.0

# Positive-class weight for BCE.  Pores are rare (~5% of sample pixels),
# so we upweight them to prevent the model collapsing to "predict nothing".
# Set to None to use uniform weighting.
SAMPLE_POS_WEIGHT = None   # sample pixels are ~30% of image — no reweighting needed
PORE_POS_WEIGHT   = 10.0   # pore pixels are ~5% of sample area — upweight strongly

# ── U-Net encoder depth / channel widths ─────────────────────────────────────
# [enc_ch1, enc_ch2, enc_ch3, enc_ch4, bottleneck]
ENCODER_CHANNELS = [32, 64, 128, 256, 512]

# CBAM reduction ratio for channel attention (default 16 from paper)
CBAM_REDUCTION   = 16

# ── Data augmentation ─────────────────────────────────────────────────────────
# Applied only during training, not validation/test.
AUG_HFLIP_P       = 0.5   # horizontal flip probability
AUG_VFLIP_P       = 0.5   # vertical flip probability
AUG_ROTATE_LIMIT  = 15    # max rotation degrees
AUG_BRIGHTNESS    = 0.2   # brightness jitter range ± fraction
AUG_CONTRAST      = 0.2   # contrast jitter range ± fraction

# ── Input normalisation ───────────────────────────────────────────────────────
# Mean and std computed over the training split.
# If unknown, ImageNet defaults (0.485 / 0.229) work as a starting point.
NORM_MEAN = 0.485
NORM_STD  = 0.229

# ── Inference ─────────────────────────────────────────────────────────────────
# Threshold applied to sigmoid output to produce a binary mask.
SAMPLE_THRESHOLD = 0.50
PORE_THRESHOLD   = 0.40   # lower than 0.5 — pores are rare, favour recall

# ── Output directories ────────────────────────────────────────────────────────
CHARTS_DIR      = CBAM_DIR / "evaluation_charts"   # override V2 value
PRED_SAMPLE_DIR = CBAM_DIR / "predictions" / "sample"
PRED_PORE_DIR   = CBAM_DIR / "predictions" / "pores"
EVAL_DIR        = CBAM_DIR / "evaluation"
VIS_DIR         = CBAM_DIR / "vis"
LOG_DIR         = CBAM_DIR / "logs"

# ── Random seed ───────────────────────────────────────────────────────────────
SEED = 42
