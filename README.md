# XCT Defect Detection Using CBAM Attention

## Introduction

This repository contains a re-implementation and extension of the Convolutional Block Attention Module (CBAM) proposed by Woo et al. (ECCV 2018) for the task of **porosity defect detection in X-ray Computed Tomography (XCT) scans** of additively manufactured metal parts. The original CBAM paper introduces a lightweight attention mechanism (channel + spatial gates) that can be inserted into any CNN backbone to improve classification accuracy. Our contribution adapts this classification-only attention module into a **pixel-level segmentation pipeline** that simultaneously detects the sample boundary and internal pore defects from 2D polar-transformed XCT slices.

**Original Paper:**
> Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module.* In Proceedings of the European Conference on Computer Vision (ECCV).

## Chosen Result

We aimed to reproduce and validate CBAM's core claim that **dual channel-spatial attention consistently improves feature representation** (Tables 3 and 5 in the original paper), and then extend this finding to a novel domain: binary segmentation of XCT scan slices. Specifically, we evaluate whether inserting CBAM after every residual block of a ResNet-34 encoder improves segmentation IoU for both sample material detection and pore detection compared to a plain ResNet-34 baseline without attention.

## GitHub Contents

```
.
├── README.md                  # This file
├── code/
│   ├── cbam.py                # Official CBAM module (from Jongchan/attention-module)
│   ├── model.py               # ResNet-34 + CBAM encoder, FPN decoder, dual-head segmentation
│   ├── dataset.py             # Patch-based dataset with augmentation and pore oversampling
│   ├── train.py               # Training loop (AdamW, cosine LR, early stopping on pore IoU)
│   ├── predict.py             # Inference with Gaussian-weighted patch blending
│   ├── evaluate.py            # Pixel-level metrics (IoU, Dice, Precision, Recall)
│   ├── config_dl.py           # All hyperparameters and paths
│   ├── generate_masks_v2.py   # Annotation-to-mask conversion
│   ├── 01_preprocessing_dl.py # Polar coordinate transformation
│   ├── 03_evaluation_charts.py# Evaluation chart generation
│   └── SETUP.txt              # Detailed setup instructions
├── data/
│   ├── README.md              # Data acquisition and structure
│   └── masks/                 # Ground-truth binary masks (sample + pores)
├── results/
│   ├── evaluation/            # Per-image and summary CSV metrics
│   ├── evaluation_charts/     # Metric visualisation plots
│   └── 3d_output/             # 3D volume reconstructions and mesh exports
├── poster/
│   └── XCT_defect_poster.pptx.pdf
└── report/
```

## Re-implementation Details

**Model Architecture:**
- **Encoder:** ResNet-34 backbone with CBAM inserted after each of the four residual layers (`layer1`-`layer4`), following the paper's Figure 3. The first convolutional layer is adapted from 3-channel RGB to 1-channel grayscale input by averaging pretrained ImageNet weights across the channel dimension.
- **Decoder (added):** FPN-style top-down decoder with lateral projections and element-wise addition, followed by two transposed convolution layers to upsample back to input resolution. This is **not** part of the original CBAM paper, which addresses classification only.
- **Dual heads:** Two independent 1x1 convolutional heads predict sample material and pore masks simultaneously from shared features.

**Dataset:**
- 14 XCT scan slices of additively manufactured metal samples, each transformed to polar coordinates for radial unwrapping of the cylindrical sample geometry.
- Images are divided into overlapping 256x256 patches (stride 128, 50% overlap) for training. Patches with <1% sample content are discarded. Pore-containing patches are oversampled 4x to address class imbalance.

**Training:**
- Loss: BCE + Dice (equal weight) per head; pore BCE uses `pos_weight=10` due to extreme class imbalance (~5% of sample area).
- Optimiser: AdamW (lr=1e-4, weight_decay=1e-5) with cosine annealing.
- Early stopping on validation pore IoU with patience of 15 epochs (max 80).
- Data augmentation: horizontal/vertical flips, rotation (+-15 deg), brightness/contrast jitter.

**Challenges:**
- The original CBAM paper only addresses image classification; adapting it to pixel-level segmentation required designing the FPN decoder and dual-head architecture.
- Pore defects are extremely small and sparse, causing the pore head to collapse to all-zeros without significant positive-class upweighting and patch-level oversampling.
- Patch boundary artifacts were resolved using Gaussian-weighted blending during inference.

## Reproduction Steps

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (recommended) or CPU-only mode
- Conda (Anaconda or Miniconda)

### Setup

```bash
# 1. Create and activate conda environment
conda create -n XCT python=3.10 -y
conda activate XCT

# 2. Install PyTorch (adjust for your CUDA version)
# CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 3. Install remaining dependencies
pip install opencv-python scikit-image numpy pandas tqdm matplotlib

# 4. Clone the official CBAM repository (inside code/)
cd code/
git clone https://github.com/Jongchan/attention-module.git
```

### Running the Pipeline

```bash
cd code/

# Step 1: Generate ground-truth masks from annotations
python generate_masks_v2.py --vis

# Step 2: Convert images and masks to polar coordinates
python 01_preprocessing_dl.py

# Step 3: Train the model
python train.py

# Step 4: Run inference on all polar images
python predict.py

# Step 5: Evaluate predictions against ground truth
python evaluate.py

# Step 6: Generate evaluation charts
python 03_evaluation_charts.py
```

### Computational Resources
- Training was performed on a single NVIDIA GPU. If CUDA out-of-memory errors occur, reduce `BATCH_SIZE` in `config_dl.py` (try 4, then 2).
- Training takes approximately 30-60 minutes depending on GPU. Early stopping typically triggers before the full 80 epochs.

## Results/Insights

Our CBAM-augmented ResNet-34 segmentation model achieves the following mean metrics across the test set:

| Class  | IoU    | Dice   | Precision | Recall |
|--------|--------|--------|-----------|--------|
| Sample | 0.9566 | 0.9732 | 0.9610    | 0.9952 |
| Pores  | 0.8434 | 0.9141 | 0.9219    | 0.9074 |

- **Sample detection** reaches near-perfect performance (95.7% IoU), indicating that CBAM's spatial attention effectively learns the ring boundary.
- **Pore detection** achieves 84.3% IoU, which is strong given the extreme class imbalance and tiny defect sizes. The channel attention mechanism in CBAM helps the model focus on the relevant feature maps for detecting subtle intensity variations caused by porosity.
- Running the repository produces per-image CSV metrics in `results/evaluation/`, visualisation charts in `results/evaluation_charts/`, and QA overlay images in `code/vis/` comparing predictions against ground truth.

## Conclusion

This project demonstrates that CBAM's lightweight attention mechanism, originally designed for image classification, **transfers effectively to pixel-level segmentation tasks** when paired with an appropriate decoder. The dual channel-spatial attention gates help the model focus on relevant features at both the "what" (channel) and "where" (spatial) levels, which is particularly valuable for detecting small, rare defects like porosity in XCT scans. The key takeaway is that attention modules designed for classification can be successfully repurposed for dense prediction tasks in industrial inspection domains with minimal architectural overhead.

## References

1. Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*. https://github.com/Jongchan/attention-module
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
3. Lin, T.-Y., Dollar, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature Pyramid Networks for Object Detection. *CVPR 2017*.
4. PyTorch. https://pytorch.org/

## Acknowledgements

This project was completed as part of a Deep Learning course (Spring 2026). The CBAM implementation (`cbam.py`) is sourced from the [official repository](https://github.com/Jongchan/attention-module) by Jongchan Park.
