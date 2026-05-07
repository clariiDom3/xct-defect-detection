"""
generate_masks_v2.py
====================
Generates TWO separate binary masks per image from polygon annotations:

  masks/sample/  <stem>_sample.png   255 = sample material, 0 = background
  masks/pores/   <stem>_pores.png    255 = pore,            0 = everything else
  masks/vis/     <stem>_overlay.png  colour QA overlay      (--vis only)

Why separate masks?
-------------------
A single combined mask (0/1/2) forces every downstream consumer to
remember the encoding.  Two independent binary masks are simpler to load,
visualise, and pass to any model or metric function.

Pores are physically inside the sample, so a pore pixel is 255 in BOTH
masks -- this is intentional and correct.

Overlap / hole handling
-----------------------
Each annotation is rendered on its own blank layer first.  Holes in one
polygon can only erase that annotation's own pixels, never a neighbour's.
Annotations of the same class are OR-merged onto a shared canvas so the
final mask is always their union.

Usage
-----
  python generate_masks_v2.py [--vis]

  --vis   also save a colour overlay for visual inspection

All paths come from config_v2.py.

Dependencies
------------
  pip install numpy pillow
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import config_v2 as cfg


# ── Class names as they appear in the annotation JSON ────────────────────────
SAMPLE_CLASS = "sample"
PORE_CLASS   = "pores"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_annotations(json_path: Path) -> list:
    print(f"[INFO] Loading annotations from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] {len(data)} image record(s) found.")
    return data


def get_image_size(image_path: Path):
    """Return (width, height) without fully decoding."""
    with Image.open(image_path) as img:
        return img.size


def render_annotation(ann: dict, width: int, height: int) -> np.ndarray:
    """
    Render a single polygon annotation onto an isolated uint8 layer.
      contour[0]  -> outer boundary -> filled with 255
      contour[1+] -> holes          -> filled with 0  (only on THIS layer)
    """
    layer = Image.new("L", (width, height), 0)
    draw  = ImageDraw.Draw(layer)
    for i, contour in enumerate(ann.get("value", {}).get("data", [])):
        if len(contour) < 3:
            continue
        flat = [coord for pt in contour for coord in pt]
        draw.polygon(flat, fill=(255 if i == 0 else 0))
    return np.array(layer, dtype=np.uint8)


def build_class_mask(annotations: list,
                     target_class: str,
                     width: int,
                     height: int) -> np.ndarray:
    """
    OR-merge all annotations of target_class into one binary mask.
    Zero pixels (holes) from a new annotation never erase existing pixels.
    """
    canvas = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        if ann.get("value", {}).get("class", "") != target_class:
            continue
        layer  = render_annotation(ann, width, height)
        canvas = np.where(layer > 0, layer, canvas)
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def save_overlay(image_path: Path,
                 sample_mask: np.ndarray,
                 pore_mask: np.ndarray,
                 out_path: Path):
    """
    Blend masks onto the original image.
      green  = sample material
      red    = pores  (drawn on top -- always visible over the green)
    """
    orig    = Image.open(image_path).convert("RGBA")
    w, h    = orig.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    for mask, color in [
        (sample_mask, (0,   200,  80, 120)),
        (pore_mask,   (220,  40,  40, 180)),
    ]:
        layer      = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        color_img  = Image.new("RGBA", (w, h), color)
        layer.paste(color_img, mask=Image.fromarray(mask).convert("L"))
        overlay    = Image.alpha_composite(overlay, layer)

    Image.alpha_composite(orig, overlay).convert("RGB").save(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# Core
# ══════════════════════════════════════════════════════════════════════════════

def generate_masks(save_vis: bool = False):
    # Create output dirs
    for d in (cfg.SAMPLE_MASKS_DIR, cfg.PORE_MASKS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if save_vis:
        cfg.MASK_VIS_DIR.mkdir(parents=True, exist_ok=True)

    records   = load_annotations(cfg.ANNOTATIONS_JSON)
    processed = 0
    skipped   = 0

    for rec in records:
        filename = rec.get("filename", "")
        if not filename:
            continue

        image_path = cfg.IMAGES_DIR / filename
        if not image_path.exists():
            print(f"[WARN] Image not found, skipping: {filename}")
            skipped += 1
            continue

        width, height = get_image_size(image_path)
        annotations   = rec.get("annotations", [])
        stem          = Path(filename).stem

        sample_mask = build_class_mask(annotations, SAMPLE_CLASS, width, height)
        pore_mask   = build_class_mask(annotations, PORE_CLASS,   width, height)

        Image.fromarray(sample_mask).save(cfg.SAMPLE_MASKS_DIR / f"{stem}_sample.png")
        Image.fromarray(pore_mask  ).save(cfg.PORE_MASKS_DIR   / f"{stem}_pores.png")

        if save_vis:
            save_overlay(image_path, sample_mask, pore_mask,
                         cfg.MASK_VIS_DIR / f"{stem}_overlay.png")

        processed += 1
        if processed % 50 == 0:
            print(f"  ... {processed} images done")

    print(f"\n[DONE] Processed: {processed}  |  Skipped: {skipped}")
    print(f"  Sample masks -> {cfg.SAMPLE_MASKS_DIR}")
    print(f"  Pore masks   -> {cfg.PORE_MASKS_DIR}")
    if save_vis:
        print(f"  Overlays     -> {cfg.MASK_VIS_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate separate sample and pore masks from XCT annotations."
    )
    ap.add_argument("--vis", action="store_true",
                    help="Also save colour overlay images for visual inspection.")
    args = ap.parse_args()
    generate_masks(save_vis=args.vis)
