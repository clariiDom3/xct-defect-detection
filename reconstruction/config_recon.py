"""
config_recon.py
===============
Single source of truth for the 3D reconstruction pipeline.
Place this file in:
  XCT data defects detection/reconstruction/

All other reconstruction scripts import from here.
"""

from pathlib import Path

# Root
BASE = Path(r"C:\Users\Aorus\OneDrive\Desktop\Research\XCT data defects detection")

# Suppress config_v2 JSON-not-found warning when imported via config_dl
import os
os.environ["SUPPRESS_V2_JSON_CHECK"] = "1"

# Input: raw .tif slices 
SLICES_DIR   = BASE / "Adj SETH-H image slices"
SLICE_SUFFIX = ".tif"          # file extension
SLICE_PREFIX = "clip_rot"      # files named clip_rot0000 … clip_rot1110

# CV approach: predictions already exist (pore + sample) 
CV_SAMPLE_DIR = BASE / "V2" / "results" / "sample_masks"
CV_PORE_DIR   = BASE / "V2" / "results" / "pore_masks"
# Naming convention: clip_rot0000_sample.png  /  clip_rot0000_pores.png

# CBAM model 
CBAM_DIR      = BASE / "DL" / "CBAM"
CBAM_CKPT     = CBAM_DIR / "checkpoints" / "best.pth"
CBAM_REPO_DIR = CBAM_DIR / "attention-module"   # for cbam.py import

# CBAM inference outputs (ring-space, same geometry as CV masks) 
CBAM_OUT_DIR        = BASE / "reconstruction" / "cbam_predictions"
CBAM_SAMPLE_OUT_DIR = CBAM_OUT_DIR / "sample"
CBAM_PORE_OUT_DIR   = CBAM_OUT_DIR / "pores"

# 3D reconstruction outputs 
RECON_DIR      = BASE / "reconstruction" / "3d_output"
MESH_DIR       = RECON_DIR / "meshes"       # .stl / .obj / .ply exports
VIS_DIR        = RECON_DIR / "screenshots"  # saved screenshots from viewer

# Voxel / physical scale 
# Each pixel represents this many micrometres in X and Y.
# Each slice step (Z direction) represents this many micrometres.
# If you know the scale bar: divide 500 (µm) by the number of pixels the
# scale bar spans in the image, then fill in below.
# Leave as 1.0 to work in pixel units — the 3D shape will still be correct.
VOXEL_SIZE_XY_UM = 1.0   # µm per pixel (X, Y)
VOXEL_SIZE_Z_UM  = 1.0   # µm per slice  (Z)

# Preprocessing parameters (must match 01_preprocessing_dl.py) 
CROP_FRACTION = 0.10   # fraction of max col-sum used for radial crop bounds

# Patch parameters (must match what CBAM was trained on) 
PATCH_SIZE   = 256
PATCH_STRIDE = 128

# Inference thresholds 
SAMPLE_THRESHOLD = 0.50
PORE_THRESHOLD   = 0.40

# 3D mesh smoothing 
# Number of Laplacian smoothing iterations applied to the mesh before export.
# Higher = smoother but less detail. 0 = no smoothing.
SMOOTH_ITERATIONS = 20

# Marching cubes iso-level 
# Binary volumes have values 0 and 1; 0.5 gives a clean surface.
ISO_LEVEL = 0.5

# Visualisation colours (RGB 0-255) 
COLOR_SAMPLE = (180, 160, 120)   # warm beige — ring material
COLOR_PORE   = (220,  80,  60)   # red        — pores
COLOR_BG     = ( 30,  30,  40)   # dark background for the viewer

# Memory / performance 
# Number of slices to process in one batch during CBAM inference.
# Reduce if you see memory errors (unlikely with 96 GB RAM).
CBAM_BATCH_SLICES = 50
