# Data

## Overview

This directory contains the ground-truth annotation masks used for training and evaluating the CBAM segmentation model. The raw XCT scan images are not included in this repository due to file size constraints and must be obtained separately (see below).

## Obtaining the Raw Data

The raw XCT scan images (16-bit grayscale `.png` slices of additively manufactured metal samples) are stored locally and are not distributed with this repository. To reproduce the full pipeline:

1. Place the raw XCT slice images into the path expected by `config_dl.py` (configured via the `IMAGES_DIR` variable).
2. Place polygon annotation files (JSON format) into the `code/annotations/` directory.
3. Run `python generate_masks_v2.py --vis` from the `code/` directory to regenerate the binary masks from annotations.

## Directory Structure

```
data/
├── README.md            # This file
└── masks/
    ├── sample/          # Binary sample-material masks (0/255 uint8 PNG)
    │   ├── 413_sample.png
    │   ├── 483_sample.png
    │   ├── ...
    │   └── 1491_sample.png
    └── pores/           # Binary porosity-defect masks (0/255 uint8 PNG)
        ├── 413_pores.png
        ├── 483_pores.png
        ├── ...
        └── 1491_pores.png
```

## Mask Details

- **14 annotated slices** total, each with a paired sample mask and pore mask.
- **Sample masks** (`masks/sample/`): pixel value 255 indicates ring/sample material; 0 indicates background. Approximately 30% of pixels are foreground.
- **Pore masks** (`masks/pores/`): pixel value 255 indicates a porosity defect; 0 indicates everything else. Pores are physically located inside the sample region and constitute roughly 5% of the sample area.
- Both masks are binary uint8 PNGs at the same resolution as the corresponding raw XCT slice.
- A pore pixel is 255 in **both** the sample and pore masks (pores are a subset of the sample region).

## Preprocessing Pipeline

Before training, the raw images and masks are transformed through the following steps (handled by scripts in `code/`):

1. **Mask generation** (`generate_masks_v2.py`): Converts polygon annotations to binary masks.
2. **Polar transformation** (`01_preprocessing_dl.py`): Unwraps the circular sample geometry into rectangular polar coordinates. The same transform is applied identically to the image and both masks to maintain pixel alignment.
3. **Patch extraction** (handled at runtime by `dataset.py`): Polar images are divided into overlapping 256x256 patches with stride 128 during training.

## Data Split

The 14 images are split at the **image level** (not patch level) to prevent data leakage:
- **Train / Val / Test** split is performed by `train.py` using a fixed random seed (`SEED=42` in `config_dl.py`).
- Split ratios are configured via `TEST_SPLIT` and `VAL_SPLIT` in the config.
