"""
02_reconstruct_3d.py
====================
Loads all prediction masks from both models, stacks them into 3D volumes,
generates meshes, and launches an interactive 3D viewer.

What this produces
------------------
For each model (CV / CBAM), for each class (sample / pores):
  - A 3D binary numpy volume  (.npy)  -- shape (n_slices, H, W)
  - A triangle mesh exported as .stl, .obj, and .ply
  - Screenshots of the interactive viewer  (.png)

Interactive viewer (PyVista)
----------------------------
Opens a local desktop window (no internet required) showing:
  - Ring material (sample) volume -- warm beige, semi-transparent
  - Pores volume                  -- red, opaque

Controls in the viewer window:
  Left-click + drag   : rotate
  Right-click + drag  : zoom
  Middle-click + drag : pan
  Q or close window   : quit

The viewer shows both models side-by-side so you can compare directly.

Usage
-----
  python 02_reconstruct_3d.py

Run AFTER:
  1. CV predictions exist in V2/results/
  2. 01_cbam_inference.py has been run

Place this file in:
  reconstruction/

Dependencies
------------
  pip install numpy opencv-python scikit-image pyvista tqdm
  (pyvista also needs vtk: pip install vtk)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from skimage.measure import marching_cubes

import config_recon as cfg

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("[WARN] PyVista not installed. Interactive viewer disabled.")
    print("       Install with:  pip install pyvista vtk")


# Slice loading helpers

def sorted_slices():
    """Return list of slice stems in numerical order."""
    paths = sorted(
        cfg.SLICES_DIR.glob(f"{cfg.SLICE_PREFIX}*{cfg.SLICE_SUFFIX}")
    )
    return [p.stem for p in paths]


def load_mask(path: Path) -> np.ndarray | None:
    """Load a binary mask PNG as a boolean 2D array."""
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img > 127


# Volume builder

def build_volume(stems: list,
                 mask_dir: Path,
                 suffix: str,
                 label: str) -> np.ndarray:
    """
    Stack 2D binary masks into a 3D numpy volume of shape (Z, H, W).

    Z = slice index (depth axis of the 3D object)
    H = image height (vertical in each slice)
    W = image width  (horizontal in each slice)

    Parameters
    ----------
    stems    : list of slice filename stems in order, e.g. "clip_rot0000"
    mask_dir : directory containing the mask PNGs
    suffix   : filename suffix, e.g. "_sample.png" or "_pores.png"
    label    : human-readable label for the progress bar
    """
    volume = None
    missing = 0

    for i, stem in enumerate(tqdm(stems, desc=f"Loading {label}", unit="slice")):
        mask_path = mask_dir / f"{stem}{suffix}"
        mask      = load_mask(mask_path)

        if mask is None:
            missing += 1
            # Use a blank slice if the mask is missing (keeps geometry consistent)
            if volume is not None:
                layer = np.zeros(volume.shape[1:], dtype=bool)
            else:
                # We don't know the shape yet — defer
                continue
        else:
            layer = mask

        if volume is None:
            # First valid slice — allocate the full volume
            # We use int16 to save RAM vs float32
            volume = np.zeros((len(stems), layer.shape[0], layer.shape[1]),
                               dtype=bool)

        volume[i] = layer

    if missing:
        print(f"  [WARN] {label}: {missing} missing slices (filled with zeros)")

    return volume


# Mesh generation

def volume_to_mesh(volume: np.ndarray,
                   voxel_size: tuple = (1.0, 1.0, 1.0)) -> pv.PolyData | None:
    """
    Convert a binary 3D volume to a PyVista triangle mesh using marching cubes.

    voxel_size : (z_um, y_um, x_um) spacing in micrometres (or pixels if 1.0)

    Applies Laplacian smoothing to reduce staircase artefacts from the
    voxel grid, then decimates (reduces triangle count) for interactive
    rendering performance.
    """
    if not HAS_PYVISTA:
        return None

    print("  Running marching cubes ... ", end="", flush=True)
    # Pad with one layer of zeros so the mesh is always closed at the edges
    padded = np.pad(volume.astype(np.float32), 1, constant_values=0)

    try:
        verts, faces, normals, _ = marching_cubes(
            padded,
            level       = cfg.ISO_LEVEL,
            spacing     = voxel_size,
            allow_degenerate=False,
        )
    except Exception as e:
        print(f"failed ({e})")
        return None

    print(f"done  ({len(faces):,} triangles)")

    n_faces    = len(faces)
    faces_flat = np.hstack([np.full((n_faces, 1), 3, dtype=int), faces])
    mesh       = pv.PolyData(verts, faces_flat.ravel())

    print(f"  Smoothing ({cfg.SMOOTH_ITERATIONS} iterations) ...", end=" ", flush=True)
    mesh = mesh.smooth(n_iter=cfg.SMOOTH_ITERATIONS, relaxation_factor=0.1)
    print("done")

    print("  Decimating ...", end=" ", flush=True)
    try:
        mesh = mesh.decimate(0.5)
        print(f"done  ({mesh.n_cells:,} triangles remaining)")
    except Exception:
        print("skipped")

    return mesh


def save_mesh(mesh, out_dir: Path, name: str):
    """Export mesh as .stl, .obj and .ply."""
    if mesh is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".stl", ".obj", ".ply"):
        out_path = out_dir / f"{name}{ext}"
        mesh.save(str(out_path))
        print(f"    Saved: {out_path.name}")


# Interactive viewer

def launch_viewer(meshes: dict, title: str = "XCT 3D Reconstruction"):
    """
    Open a side-by-side interactive PyVista window.

    meshes : dict mapping label -> (mesh, color_rgb, opacity)
             e.g. {"CV sample": (mesh_obj, (180,160,120), 0.6),
                   "CV pores" : (mesh_obj, (220,80,60),   1.0)}

    Controls
    --------
    Left drag    : rotate
    Right drag   : zoom
    Middle drag  : pan
    Q / close    : quit
    """
    if not HAS_PYVISTA:
        print("[SKIP] PyVista not available — cannot open viewer.")
        return

    valid = {k: v for k, v in meshes.items() if v[0] is not None}
    if not valid:
        print("[WARN] No valid meshes to display.")
        return

    n     = len(valid)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    pl = pv.Plotter(
        shape     = (nrows, ncols),
        title     = title,
        window_size = (1600, 900),
    )
    pl.set_background(
        [c / 255.0 for c in cfg.COLOR_BG]
    )

    for idx, (label, (mesh, color, opacity)) in enumerate(valid.items()):
        row = idx // ncols
        col = idx %  ncols
        pl.subplot(row, col)
        pl.add_mesh(
            mesh,
            color       = [c / 255.0 for c in color],
            opacity     = opacity,
            smooth_shading = True,
            show_edges  = False,
        )
        pl.add_text(label, font_size=12, position="upper_left")
        pl.add_axes()
        pl.reset_camera()

    print("\n[Viewer] Opening interactive window ...")
    print("  Left-drag: rotate  |  Right-drag: zoom  |  Middle-drag: pan")
    print("  Press Q or close the window to exit.\n")

    cfg.VIS_DIR.mkdir(parents=True, exist_ok=True)
    pl.show(auto_close=False)

    # Save screenshot after window closes
    screenshot_path = cfg.VIS_DIR / f"{title.replace(' ','_')}.png"
    pl.screenshot(str(screenshot_path))
    print(f"  Screenshot saved: {screenshot_path}")
    pl.close()


# Main

def main():
    cfg.RECON_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MESH_DIR.mkdir( parents=True, exist_ok=True)

    stems = sorted_slices()
    if not stems:
        raise FileNotFoundError(
            f"No slices found in {cfg.SLICES_DIR}\n"
            f"Check SLICE_PREFIX and SLICE_SUFFIX in config_recon.py"
        )
    print(f"[INFO] {len(stems)} slices found.")

    voxel = (
        cfg.VOXEL_SIZE_Z_UM,
        cfg.VOXEL_SIZE_XY_UM,
        cfg.VOXEL_SIZE_XY_UM,
    )

    # Load volumes 
    print("\n── CV approach ───────────────────────────────────────────────────")
    cv_sample_vol = build_volume(stems, cfg.CV_SAMPLE_DIR,
                                  "_sample.png", "CV sample")
    cv_pore_vol   = build_volume(stems, cfg.CV_PORE_DIR,
                                  "_pores.png",  "CV pores")

    print("\n── CBAM model ────────────────────────────────────────────────────")
    cbam_sample_vol = build_volume(stems, cfg.CBAM_SAMPLE_OUT_DIR,
                                    "_sample.png", "CBAM sample")
    cbam_pore_vol   = build_volume(stems, cfg.CBAM_PORE_OUT_DIR,
                                    "_pores.png",  "CBAM pores")

    # Save volumes as .npy for later use 
    print("\n── Saving 3D volumes (.npy) ──────────────────────────────────────")
    for vol, name in [
        (cv_sample_vol,   "cv_sample"),
        (cv_pore_vol,     "cv_pores"),
        (cbam_sample_vol, "cbam_sample"),
        (cbam_pore_vol,   "cbam_pores"),
    ]:
        if vol is not None:
            out = cfg.RECON_DIR / f"{name}_volume.npy"
            np.save(str(out), vol.astype(np.uint8))
            print(f"  Saved: {out.name}  shape={vol.shape}")

    # Generate meshes 
    if HAS_PYVISTA:
        print("\n── Building meshes ───────────────────────────────────────────────")
        meshes = {}

        for vol, label, color, opacity in [
            (cv_sample_vol,   "CV — Sample",  cfg.COLOR_SAMPLE, 0.55),
            (cv_pore_vol,     "CV — Pores",   cfg.COLOR_PORE,   1.00),
            (cbam_sample_vol, "CBAM — Sample",cfg.COLOR_SAMPLE, 0.55),
            (cbam_pore_vol,   "CBAM — Pores", cfg.COLOR_PORE,   1.00),
        ]:
            if vol is None:
                continue
            print(f"\n  {label}:")
            mesh = volume_to_mesh(vol, voxel_size=voxel)
            meshes[label] = (mesh, color, opacity)
            save_mesh(mesh, cfg.MESH_DIR, label.replace(" ","_").replace("—",""))

        # Launch interactive viewers 
        print("\n── Launching viewers ─────────────────────────────────────────────")
        print("  [1/2] CV approach")
        launch_viewer(
            {k: v for k, v in meshes.items() if "CV" in k},
            title = "CV Approach — Sample & Pores"
        )
        print("  [2/2] CBAM model")
        launch_viewer(
            {k: v for k, v in meshes.items() if "CBAM" in k},
            title = "CBAM Model — Sample & Pores"
        )
        print("  [3/3] Side-by-side comparison")
        launch_viewer(meshes, title="CV vs CBAM — Full Comparison")

    else:
        print("\n[INFO] Install PyVista for interactive 3D viewing:")
        print("       pip install pyvista vtk")

    print(f"\n[DONE]")
    print(f"  Volumes -> {cfg.RECON_DIR}")
    print(f"  Meshes  -> {cfg.MESH_DIR}")
    print(f"  (Open .stl files in MeshLab, Blender, or any 3D viewer)")


if __name__ == "__main__":
    main()
