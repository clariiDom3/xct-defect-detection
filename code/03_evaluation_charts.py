"""
03_evaluation_charts.py
========================
Generates evaluation charts for the CBAM segmentation model results.
Saves all charts to config_dl.CHARTS_DIR  (DL/CBAM/evaluation_charts/).

Charts produced
---------------
  01_summary_metrics.png      Grouped bar chart — sample vs pores, all metrics
  02_per_image_iou.png        IoU per image, both classes
  03_per_image_pores.png      All 5 metrics per image, pores class
  04_per_image_sample.png     All 5 metrics per image, sample class
  05_metric_distributions.png Violin + strip plots across images
  06_precision_recall.png     Precision-Recall scatter with F1 iso-lines
  07_confusion_pores.png      TP / FP / FN pixel counts per image (pores)
  08_vs_baseline.png          CBAM vs IP baseline comparison (if baseline
                               CSVs are available in config_v2.RESULTS_DIR)

Usage
-----
  python 03_evaluation_charts.py

Dependencies
------------
  pip install matplotlib seaborn pandas numpy
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

import config_dl as cfg

# ── Load CBAM results ─────────────────────────────────────────────────────────
EVAL_DIR   = cfg.EVAL_DIR
CHARTS_DIR = cfg.CHARTS_DIR
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def _load(name):
    p = EVAL_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"Missing: {p}\nRun  python evaluate.py  first."
        )
    return pd.read_csv(p)

df_sample  = _load("evaluation_sample.csv")
df_pores   = _load("evaluation_pores.csv")
df_summary = _load("evaluation_summary.csv")

df_sample["image"] = df_sample["image"].astype(str)
df_pores["image"]  = df_pores["image"].astype(str)

METRIC_COLS = ["IoU", "Dice", "Precision", "Recall", "F1"]

# ── Palette ───────────────────────────────────────────────────────────────────
SAMPLE_COL = "#4C9BE8"
PORE_COL   = "#E8614C"

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _save(fig, name):
    out = CHARTS_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 01 — Summary bar chart
# ══════════════════════════════════════════════════════════════════════════════
def chart_summary():
    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(METRIC_COLS)
    x = np.arange(n)
    w = 0.35

    s_row = df_summary[df_summary["class"] == "sample"].iloc[0]
    p_row = df_summary[df_summary["class"] == "pores"].iloc[0]
    s_vals = [s_row[f"{m}_mean"] for m in METRIC_COLS]
    p_vals = [p_row[f"{m}_mean"] for m in METRIC_COLS]
    s_stds = [s_row[f"{m}_std"]  for m in METRIC_COLS]
    p_stds = [p_row[f"{m}_std"]  for m in METRIC_COLS]

    bs = ax.bar(x - w/2, s_vals, w, yerr=s_stds, capsize=4,
                color=SAMPLE_COL, label="Sample", alpha=0.88,
                error_kw={"linewidth": 1.2})
    bp = ax.bar(x + w/2, p_vals, w, yerr=p_stds, capsize=4,
                color=PORE_COL,   label="Pores",  alpha=0.88,
                error_kw={"linewidth": 1.2})

    for bars in (bs, bp):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_COLS, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (mean ± std)")
    ax.set_title("CBAM Model — Summary Metrics\nSample vs Pore Detection",
                 fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.axhline(1.0, color="grey", lw=0.6, ls="--", alpha=0.5)
    _save(fig, "01_summary_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# 02 — Per-image IoU
# ══════════════════════════════════════════════════════════════════════════════
def chart_per_image_iou():
    images = df_sample["image"].tolist()
    x = np.arange(len(images))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, df_sample["IoU"], w, color=SAMPLE_COL, label="Sample", alpha=0.88)
    ax.bar(x + w/2, df_pores["IoU"],  w, color=PORE_COL,   label="Pores",  alpha=0.88)
    ax.axhline(df_sample["IoU"].mean(), color=SAMPLE_COL, lw=1.5, ls="--", alpha=0.7,
               label=f"Sample mean ({df_sample['IoU'].mean():.3f})")
    ax.axhline(df_pores["IoU"].mean(),  color=PORE_COL,   lw=1.5, ls="--", alpha=0.7,
               label=f"Pores mean  ({df_pores['IoU'].mean():.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(images, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("IoU")
    ax.set_xlabel("Image ID")
    ax.set_title("Per-Image IoU — Sample vs Pores", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    _save(fig, "02_per_image_iou.png")


# ══════════════════════════════════════════════════════════════════════════════
# 03 — Per-image all metrics, pores
# ══════════════════════════════════════════════════════════════════════════════
def chart_per_image_pores():
    images = df_pores["image"].tolist()
    x = np.arange(len(images))
    n = len(METRIC_COLS)
    w = 0.14
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * w
    colors  = sns.color_palette("Set2", n)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (metric, color) in enumerate(zip(METRIC_COLS, colors)):
        ax.bar(x + offsets[i], df_pores[metric], w,
               label=metric, color=color, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(images, rotation=45, ha="right")
    ax.set_ylim(0.60, 1.00)
    ax.set_ylabel("Score")
    ax.set_xlabel("Image ID")
    ax.set_title("Per-Image Metrics — Pores", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9, ncol=5, fontsize=9)
    _save(fig, "03_per_image_pores.png")


# ══════════════════════════════════════════════════════════════════════════════
# 04 — Per-image all metrics, sample
# ══════════════════════════════════════════════════════════════════════════════
def chart_per_image_sample():
    images = df_sample["image"].tolist()
    x = np.arange(len(images))
    n = len(METRIC_COLS)
    w = 0.14
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * w
    colors  = sns.color_palette("Set1", n)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (metric, color) in enumerate(zip(METRIC_COLS, colors)):
        ax.bar(x + offsets[i], df_sample[metric], w,
               label=metric, color=color, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(images, rotation=45, ha="right")
    ax.set_ylim(0.40, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Image ID")
    ax.set_title("Per-Image Metrics — Sample", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9, ncol=5, fontsize=9)
    _save(fig, "04_per_image_sample.png")


# ══════════════════════════════════════════════════════════════════════════════
# 05 — Metric distributions
# ══════════════════════════════════════════════════════════════════════════════
def chart_distributions():
    rows = []
    for _, r in df_sample.iterrows():
        for m in METRIC_COLS:
            rows.append({"Metric": m, "Score": r[m], "Class": "Sample"})
    for _, r in df_pores.iterrows():
        for m in METRIC_COLS:
            rows.append({"Metric": m, "Score": r[m], "Class": "Pores"})
    df_long = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cls, color in zip(axes, ["Sample", "Pores"], [SAMPLE_COL, PORE_COL]):
        sub = df_long[df_long["Class"] == cls]
        sns.violinplot(data=sub, x="Metric", y="Score", ax=ax,
                       color=color, alpha=0.55, inner=None, linewidth=1.2)
        sns.stripplot(data=sub, x="Metric", y="Score", ax=ax,
                      color="black", size=5, jitter=True, alpha=0.75)
        ax.set_title(f"Score Distribution — {cls}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_xlabel("")
    fig.suptitle("Metric Distributions Across All Images",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "05_metric_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# 06 — Precision-Recall scatter
# ══════════════════════════════════════════════════════════════════════════════
def chart_precision_recall():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, df, cls, color, ylim in [
        (axes[0], df_sample, "Sample", SAMPLE_COL, (0.40, 1.02)),
        (axes[1], df_pores,  "Pores",  PORE_COL,   (0.70, 1.02)),
    ]:
        ax.scatter(df["Precision"], df["Recall"],
                   color=color, s=70, alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.8)
        for _, row in df.iterrows():
            ax.annotate(str(row["image"]),
                        (row["Precision"], row["Recall"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color="dimgrey")
        # F1 iso-lines
        p_range = np.linspace(0.01, 1.0, 200)
        for f1_val in [0.7, 0.8, 0.9, 0.95]:
            r_iso  = f1_val * p_range / (2 * p_range - f1_val + 1e-9)
            valid  = (r_iso > 0) & (r_iso <= 1) & (p_range <= 1)
            ax.plot(p_range[valid], r_iso[valid],
                    color="grey", lw=0.7, ls="--", alpha=0.5)
            idx = np.argmin(np.abs(p_range[valid] - 0.80))
            ax.text(p_range[valid][idx], r_iso[valid][idx],
                    f"F1={f1_val}", fontsize=6.5, color="grey", alpha=0.8)
        ax.set_xlim(0.3, 1.05)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        ax.set_title(f"Precision–Recall — {cls}\n(each point = one image)",
                     fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_precision_recall.png")


# ══════════════════════════════════════════════════════════════════════════════
# 07 — Confusion pixel counts (pores)
# ══════════════════════════════════════════════════════════════════════════════
def chart_confusion_pores():
    images = df_pores["image"].tolist()
    x = np.arange(len(images))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, df_pores["TP"], w, label="TP (correct pore)",  color="#2ecc71", alpha=0.88)
    ax.bar(x,     df_pores["FP"], w, label="FP (false alarm)",   color="#e74c3c", alpha=0.88)
    ax.bar(x + w, df_pores["FN"], w, label="FN (missed pore)",   color="#f39c12", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(images, rotation=45, ha="right")
    ax.set_ylabel("Pixel count")
    ax.set_xlabel("Image ID")
    ax.set_title("Pore Detection — TP / FP / FN Pixel Counts per Image",
                 fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    _save(fig, "07_confusion_pores.png")


# ══════════════════════════════════════════════════════════════════════════════
# 08 — CBAM vs IP baseline comparison
# ══════════════════════════════════════════════════════════════════════════════
def chart_vs_baseline():
    # Try to load baseline results from V2 results folder
    base_dir = cfg.RESULTS_DIR   # V2/results/ from config_v2
    bp  = base_dir / "evaluation_pores.csv"
    bsp = base_dir / "evaluation_sample.csv"

    if not bp.exists() or not bsp.exists():
        print("  [SKIP] Baseline CSVs not found — skipping chart 08.")
        print(f"         Expected in: {base_dir}")
        return

    b_pores  = pd.read_csv(bp)
    b_sample = pd.read_csv(bsp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metrics_to_plot = ["IoU", "Dice", "Precision", "Recall"]

    for ax, cls, df_cbam, df_base, title in [
        (axes[0], "Sample", df_sample, b_sample, "Sample Detection"),
        (axes[1], "Pores",  df_pores,  b_pores,  "Pore Detection"),
    ]:
        n  = len(metrics_to_plot)
        x  = np.arange(n)
        w  = 0.35

        cbam_vals = [df_cbam[m].mean() for m in metrics_to_plot]
        base_vals = [df_base[m].mean() for m in metrics_to_plot]
        cbam_stds = [df_cbam[m].std()  for m in metrics_to_plot]
        base_stds = [df_base[m].std()  for m in metrics_to_plot]

        ax.bar(x - w/2, base_vals, w, yerr=base_stds, capsize=4,
               color="#95a5a6", label="IP Baseline", alpha=0.85,
               error_kw={"linewidth": 1.2})
        ax.bar(x + w/2, cbam_vals, w, yerr=cbam_stds, capsize=4,
               color="#2980b9", label="CBAM Model",  alpha=0.88,
               error_kw={"linewidth": 1.2})

        # Delta labels
        for i, (bv, cv) in enumerate(zip(base_vals, cbam_vals)):
            delta = cv - bv
            color = "#27ae60" if delta >= 0 else "#e74c3c"
            ax.text(i + w/2, cv + 0.02,
                    f"{'+' if delta>=0 else ''}{delta:.3f}",
                    ha="center", fontsize=8, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score (mean ± std)")
        ax.set_title(f"{title}\nCBAM vs IP Baseline",
                     fontsize=11, fontweight="bold")
        ax.legend(framealpha=0.9)
        ax.axhline(1.0, color="grey", lw=0.6, ls="--", alpha=0.4)

    fig.suptitle("CBAM Model vs IP Baseline — All Metrics",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "08_vs_baseline.png")


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"[INFO] Saving charts to: {CHARTS_DIR}\n")
    chart_summary()
    chart_per_image_iou()
    chart_per_image_pores()
    chart_per_image_sample()
    chart_distributions()
    chart_precision_recall()
    chart_confusion_pores()
    chart_vs_baseline()
    print(f"\n[DONE] Charts saved to {CHARTS_DIR}")
