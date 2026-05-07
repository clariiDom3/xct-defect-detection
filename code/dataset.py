"""
dataset.py
==========
Patch-based dataset for the two-stage CBAM U-Net pipeline.

Patch strategy
--------------
Each polar image (shape H × W) is divided into overlapping square patches
of size PATCH_SIZE with step PATCH_STRIDE.  Example:

  Polar image: 512 × 2048 px
  PATCH_SIZE=256, PATCH_STRIDE=128  (50% overlap)

  Rows of patches: ceil((512  - 256) / 128) + 1 =  4
  Cols of patches: ceil((2048 - 256) / 128) + 1 = 15
  Patches per image: 4 × 15 = 60

  The image is zero-padded on the right/bottom if its size is not an exact
  multiple of the stride, so every patch is always exactly PATCH_SIZE².

  At inference, overlapping predictions are averaged with a Gaussian
  weight window (see predict.py) to eliminate visible seams at boundaries.

Classes
-------
  PolarPatchDataset   — used for training and validation
    Each __getitem__ returns one patch triplet:
      image_patch   : (1, H, W) float tensor, normalised
      sample_patch  : (1, H, W) float binary mask (GT sample)
      pore_patch    : (1, H, W) float binary mask (GT pores)

  PolarInferenceDataset — used for predict.py
    Each __getitem__ returns one image patch + its (row, col, image_idx)
    coordinates so predict.py can reassemble patches into the full mask.
"""

import random
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config_dl as cfg


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_gray_uint8(path: Path) -> np.ndarray:
    """Load any PNG (8-bit or 16-bit) and return as uint8 via min-max norm."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img.astype(np.float32) - lo) / (hi - lo + 1e-9) * 255
               ).clip(0, 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _pad_to_multiple(img: np.ndarray,
                     patch_size: int,
                     stride: int) -> np.ndarray:
    """
    Zero-pad the image so its dimensions are compatible with the patch grid.
    The padded size is the smallest value >= img.shape[i] such that
    (size - patch_size) is exactly divisible by stride.
    """
    def _next(n):
        if n <= patch_size:
            return patch_size
        rem = (n - patch_size) % stride
        return n if rem == 0 else n + (stride - rem)

    ph = _next(img.shape[0])
    pw = _next(img.shape[1])
    padded = np.zeros((ph, pw), dtype=img.dtype)
    padded[:img.shape[0], :img.shape[1]] = img
    return padded


def _extract_patches(img: np.ndarray,
                     patch_size: int,
                     stride: int) -> Tuple[List[np.ndarray], List[Tuple]]:
    """
    Extract all overlapping patches from a 2-D array.

    Returns
    -------
    patches    : list of (patch_size, patch_size) uint8 arrays
    positions  : list of (row_start, col_start) tuples matching patches
    """
    patches, positions = [], []
    H, W = img.shape
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patches.append(img[r:r + patch_size, c:c + patch_size])
            positions.append((r, c))
    return patches, positions


def _to_tensor_image(patch: np.ndarray, mean: float, std: float) -> torch.Tensor:
    """
    Convert a uint8 (H, W) patch to a (1, H, W) float32 tensor, normalised.
    """
    x = patch.astype(np.float32) / 255.0
    x = (x - mean) / std
    return torch.from_numpy(x).unsqueeze(0)


def _to_tensor_mask(patch: np.ndarray) -> torch.Tensor:
    """
    Convert a uint8 binary mask (0/255) to a (1, H, W) float32 tensor (0/1).
    """
    return torch.from_numpy((patch > 127).astype(np.float32)).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation helpers (applied identically to image + both masks)
# ══════════════════════════════════════════════════════════════════════════════

def _augment(img: np.ndarray,
             sample_mask: np.ndarray,
             pore_mask: np.ndarray,
             seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply random spatial + photometric augmentation consistently to
    the image and both masks.

    Spatial transforms (flip, rotate) are applied identically to all three.
    Photometric transforms (brightness / contrast) are applied to the image
    only — masks are binary and must not be altered photometrically.
    """
    rng = np.random.RandomState(seed)

    # Horizontal flip
    if rng.rand() < cfg.AUG_HFLIP_P:
        img         = np.fliplr(img)
        sample_mask = np.fliplr(sample_mask)
        pore_mask   = np.fliplr(pore_mask)

    # Vertical flip
    if rng.rand() < cfg.AUG_VFLIP_P:
        img         = np.flipud(img)
        sample_mask = np.flipud(sample_mask)
        pore_mask   = np.flipud(pore_mask)

    # Random rotation
    angle = rng.uniform(-cfg.AUG_ROTATE_LIMIT, cfg.AUG_ROTATE_LIMIT)
    if abs(angle) > 0.5:
        H, W = img.shape
        M    = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        img         = cv2.warpAffine(img,         M, (W, H),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
        sample_mask = cv2.warpAffine(sample_mask, M, (W, H),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
        pore_mask   = cv2.warpAffine(pore_mask,   M, (W, H),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)

    # Brightness & contrast jitter (image only)
    alpha = 1.0 + rng.uniform(-cfg.AUG_CONTRAST,   cfg.AUG_CONTRAST)
    beta  = rng.uniform(-cfg.AUG_BRIGHTNESS * 255, cfg.AUG_BRIGHTNESS * 255)
    img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Ensure contiguity after flips
    return (np.ascontiguousarray(img),
            np.ascontiguousarray(sample_mask),
            np.ascontiguousarray(pore_mask))


# ══════════════════════════════════════════════════════════════════════════════
# Training / validation dataset
# ══════════════════════════════════════════════════════════════════════════════

class PolarPatchDataset(Dataset):
    """
    Patch-based dataset for training and validation.

    Each item is a triplet of patches extracted from one polar image:
      image_patch  : (1, PATCH_SIZE, PATCH_SIZE) normalised float32
      sample_patch : (1, PATCH_SIZE, PATCH_SIZE) binary float32 (0/1)
      pore_patch   : (1, PATCH_SIZE, PATCH_SIZE) binary float32 (0/1)

    Patch extraction
    ----------------
    For a polar image of shape (H, W) with PATCH_SIZE=256, PATCH_STRIDE=128:
      - The image is padded to the next compatible size.
      - Every (PATCH_SIZE × PATCH_SIZE) window is extracted at steps of
        PATCH_STRIDE in both dimensions.
      - Overlapping patches are averaged during inference (see predict.py).

    Only patches where the GT sample mask contains at least MIN_SAMPLE_FRAC
    foreground pixels are kept — blank background patches are useless.
    Separately, pore patches (where pore mask has any foreground) are
    oversampled to PORE_OVERSAMPLE_FACTOR × their natural count to
    counteract class imbalance at the patch level.

    Args
    ----
    image_paths  : list of polar image paths
    augment      : apply random augmentation (True for train, False for val)
    patch_size   : side length of each square patch (default PATCH_SIZE)
    stride       : step between patches (default PATCH_STRIDE)
    """

    # Minimum fraction of sample foreground pixels for a patch to be kept.
    # Discards fully-background patches that add nothing to training.
    MIN_SAMPLE_FRAC   = 0.01   # at least 1% of patch must be sample material

    # Patches with any pore pixels are repeated this many times to compensate
    # for the rarity of pores relative to background/sample patches.
    PORE_OVERSAMPLE_FACTOR = 4

    def __init__(
        self,
        image_paths: List[Path],
        augment: bool = False,
        patch_size: int = cfg.PATCH_SIZE,
        stride:     int = cfg.PATCH_STRIDE,
    ):
        self.augment     = augment
        self.patch_size  = patch_size
        self.stride      = stride
        self.mean        = cfg.NORM_MEAN
        self.std         = cfg.NORM_STD

        # Build the flat list of (img_patch, sample_patch, pore_patch) tuples.
        # Each entry stores numpy uint8 arrays (not tensors) to minimise RAM.
        self._items: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        print(f"[Dataset] Building patch index from {len(image_paths)} image(s) "
              f"(patch={patch_size}px, stride={stride}px, augment={augment})")

        for img_path in image_paths:
            stem = img_path.stem   # e.g. "1491_polar"
            orig = stem.replace("_polar", "")

            # Load polar image and GT masks
            img = _load_gray_uint8(img_path)

            s_path = cfg.POLAR_SAMPLE_MASKS_DIR / f"{orig}_polar_sample.png"
            p_path = cfg.POLAR_PORE_MASKS_DIR   / f"{orig}_polar_pores.png"

            if not s_path.exists() or not p_path.exists():
                print(f"  [WARN] GT masks not found for {orig}, skipping.")
                continue

            sample_mask = _load_gray_uint8(s_path)
            pore_mask   = _load_gray_uint8(p_path)

            # Pad all three arrays to the same compatible shape
            img_p    = _pad_to_multiple(img,         patch_size, stride)
            smask_p  = _pad_to_multiple(sample_mask, patch_size, stride)
            pmask_p  = _pad_to_multiple(pore_mask,   patch_size, stride)

            # Extract patches — positions are the same for image and both masks
            img_patches,    positions = _extract_patches(img_p,   patch_size, stride)
            smask_patches,  _         = _extract_patches(smask_p, patch_size, stride)
            pmask_patches,  _         = _extract_patches(pmask_p, patch_size, stride)

            n_total = len(img_patches)
            n_kept  = 0
            n_pore  = 0

            for ip, sp, pp in zip(img_patches, smask_patches, pmask_patches):
                # Discard patches with almost no sample material
                if (sp > 127).mean() < self.MIN_SAMPLE_FRAC:
                    continue

                self._items.append((ip, sp, pp))
                n_kept += 1

                # Oversample patches that contain at least one pore pixel
                if (pp > 127).any():
                    for _ in range(self.PORE_OVERSAMPLE_FACTOR - 1):
                        self._items.append((ip, sp, pp))
                    n_pore += 1

            print(f"  {orig}: {n_total} patches -> {n_kept} kept "
                  f"({n_pore} pore patches × {self.PORE_OVERSAMPLE_FACTOR}x oversampled)")

        print(f"[Dataset] Total items: {len(self._items)}\n")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_p, smask_p, pmask_p = self._items[idx]

        if self.augment:
            # Use the index as a seed so augmentation is reproducible per item
            img_p, smask_p, pmask_p = _augment(img_p, smask_p, pmask_p, seed=idx)

        image_t  = _to_tensor_image(img_p,   self.mean, self.std)
        sample_t = _to_tensor_mask(smask_p)
        pore_t   = _to_tensor_mask(pmask_p)

        return image_t, sample_t, pore_t


# ══════════════════════════════════════════════════════════════════════════════
# Inference dataset — returns patches + position info for reassembly
# ══════════════════════════════════════════════════════════════════════════════

class PolarInferenceDataset(Dataset):
    """
    Patch dataset for inference (no GT masks required).

    Each item returns:
      patch_tensor : (1, PATCH_SIZE, PATCH_SIZE) normalised float32
      meta         : dict with keys:
                       'img_idx'    — index into image_paths list
                       'row_start'  — top row of this patch in the padded image
                       'col_start'  — left col of this patch in the padded image
                       'orig_H'     — original (un-padded) image height
                       'orig_W'     — original (un-padded) image width
                       'pad_H'      — padded image height
                       'pad_W'      — padded image width

    predict.py uses meta to place each patch prediction back into a full
    probability map of size (pad_H, pad_W), then crops to (orig_H, orig_W).
    """

    def __init__(
        self,
        image_paths: List[Path],
        patch_size: int = cfg.PATCH_SIZE,
        stride:     int = cfg.PATCH_STRIDE,
    ):
        self.patch_size = patch_size
        self.stride     = stride
        self.mean       = cfg.NORM_MEAN
        self.std        = cfg.NORM_STD

        # Pre-load and pad all images; build flat patch list
        self._patches: List[np.ndarray]  = []
        self._metas:   List[dict]        = []

        for img_idx, img_path in enumerate(image_paths):
            img = _load_gray_uint8(img_path)
            oh, ow = img.shape
            img_p = _pad_to_multiple(img, patch_size, stride)
            ph, pw = img_p.shape

            patches, positions = _extract_patches(img_p, patch_size, stride)
            for patch, (r, c) in zip(patches, positions):
                self._patches.append(patch)
                self._metas.append({
                    "img_idx":   img_idx,
                    "row_start": r,
                    "col_start": c,
                    "orig_H":    oh,
                    "orig_W":    ow,
                    "pad_H":     ph,
                    "pad_W":     pw,
                })

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        patch  = self._patches[idx]
        tensor = _to_tensor_image(patch, self.mean, self.std)
        return tensor, self._metas[idx]
