"""
model.py
========
Single-model segmentation using CBAM (Woo et al., ECCV 2018).

Architecture — ResNet-34 encoder + CBAM + FPN decoder + two heads
------------------------------------------------------------------
From the paper (unchanged):
  - ResNet-34 backbone
  - CBAM inserted after every residual layer (layer1 -> layer4)
    * ChannelGate: which feature maps matter
    * SpatialGate: where in the map to focus

Added for segmentation (not in the paper — paper is classification only):
  - FPN-style decoder: top-down feature fusion with element-wise addition
  - Two output heads (1x1 conv each):
      sample_head -> binary logit for ring material
      pore_head   -> binary logit for pores

Single-model design
-------------------
One model predicts BOTH sample and pores simultaneously from a single
1-channel grayscale polar patch.  The shared ResNet+CBAM encoder learns
features useful for both tasks at once.  The two heads specialise
independently in the final layer.

CBAM import
-----------
cbam.py is imported directly from the same folder (copied from the
official Jongchan/attention-module repository).
model_resnet.py and train_imagenet.py from that repo are NOT used —
they are for ImageNet classification.

Reference:
  @InProceedings{Woo_2018_ECCV,
    author = {Woo, Sanghyun and Park, Jongchan and
              Lee, Joon-Young and Kweon, In So},
    title  = {CBAM: Convolutional Block Attention Module},
    booktitle = {ECCV},
    year   = {2018},
  }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

import config_dl as cfg

# cbam.py is in the same folder — direct import, no sys.path needed.
# (copied from attention-module/MODELS/cbam.py)
from cbam import CBAM   # official Jongchan implementation


# ══════════════════════════════════════════════════════════════════════════════
# Decoder helper
# ══════════════════════════════════════════════════════════════════════════════

class ConvBnRelu(nn.Module):
    """Conv2d -> BN -> ReLU.  Used in the FPN decoder."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ══════════════════════════════════════════════════════════════════════════════
# Encoder: ResNet-34 + CBAM  (paper architecture)
# ══════════════════════════════════════════════════════════════════════════════

class ResNetCBAMEncoder(nn.Module):
    """
    ResNet-34 with CBAM after every residual layer.

    This is the architecture described in the CBAM paper (Figure 3):
      stem -> layer1 -> CBAM -> layer2 -> CBAM -> layer3 -> CBAM -> layer4 -> CBAM

    in_channels is always 1 (grayscale polar patch).
    The first conv is replaced with a 1-channel version; pretrained
    weights are averaged over the original 3 channels to initialise it.

    Output scales for a 256x256 input patch:
      s1: (B,  64,  64, 64)   stride 4
      s2: (B, 128,  32, 32)   stride 8
      s3: (B, 256,  16, 16)   stride 16
      s4: (B, 512,   8,  8)   stride 32
    """

    def __init__(self,
                 pretrained:     bool = True,
                 cbam_reduction: int  = cfg.CBAM_REDUCTION):
        super().__init__()

        backbone = tvm.resnet34(
            weights=tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Replace 3-channel first conv with 1-channel version.
        # ── ADDED (not in paper): adapter to accept grayscale input ──────────
        orig = backbone.conv1  # weight shape (64, 3, 7, 7)
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Average pretrained weights over 3 input channels
            new_conv.weight.copy_(orig.weight.mean(dim=1, keepdim=True))
        self.stem = nn.Sequential(
            new_conv, backbone.bn1, backbone.relu, backbone.maxpool
        )

        # ResNet layers + CBAM — exactly the paper's Figure 3
        self.layer1 = backbone.layer1  # out:  64 ch
        self.cbam1  = CBAM( 64, reduction_ratio=cbam_reduction)

        self.layer2 = backbone.layer2  # out: 128 ch
        self.cbam2  = CBAM(128, reduction_ratio=cbam_reduction)

        self.layer3 = backbone.layer3  # out: 256 ch
        self.cbam3  = CBAM(256, reduction_ratio=cbam_reduction)

        self.layer4 = backbone.layer4  # out: 512 ch
        self.cbam4  = CBAM(512, reduction_ratio=cbam_reduction)

    def forward(self, x: torch.Tensor):
        x  = self.stem(x)
        s1 = self.cbam1(self.layer1(x))   # (B,  64, 64, 64)
        s2 = self.cbam2(self.layer2(s1))  # (B, 128, 32, 32)
        s3 = self.cbam3(self.layer3(s2))  # (B, 256, 16, 16)
        s4 = self.cbam4(self.layer4(s3))  # (B, 512,  8,  8)
        return s1, s2, s3, s4


# ══════════════════════════════════════════════════════════════════════════════
# Decoder: FPN-style  (ADDED — not in paper)
# ══════════════════════════════════════════════════════════════════════════════

class FPNDecoder(nn.Module):
    """
    ADDED for segmentation — not in the CBAM paper.

    Feature Pyramid Network top-down pathway:
      p4 = project(s4)
      p3 = project(s3) + upsample(p4)
      p2 = project(s2) + upsample(p3)
      p1 = project(s1) + upsample(p2)
    Then upsample p1 back to the input patch resolution (256x256).

    Element-wise addition (not concatenation) keeps the decoder
    lightweight.  Two ConvTranspose2d layers do the final 4x upsample
    from stride-4 feature maps back to full resolution.
    """

    def __init__(self, fpn_channels: int = 128):
        super().__init__()
        # Lateral projections: each encoder level -> fpn_channels
        self.lat4 = nn.Conv2d(512, fpn_channels, 1)
        self.lat3 = nn.Conv2d(256, fpn_channels, 1)
        self.lat2 = nn.Conv2d(128, fpn_channels, 1)
        self.lat1 = nn.Conv2d( 64, fpn_channels, 1)

        # Smoothing convs after addition (removes aliasing artefacts)
        self.smooth4 = ConvBnRelu(fpn_channels, fpn_channels)
        self.smooth3 = ConvBnRelu(fpn_channels, fpn_channels)
        self.smooth2 = ConvBnRelu(fpn_channels, fpn_channels)
        self.smooth1 = ConvBnRelu(fpn_channels, fpn_channels)

        # 4x upsample from stride-4 (64x64) to full resolution (256x256)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(fpn_channels,     fpn_channels // 2,
                               4, stride=2, padding=1),
            nn.BatchNorm2d(fpn_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fpn_channels // 2, fpn_channels // 4,
                               4, stride=2, padding=1),
            nn.BatchNorm2d(fpn_channels // 4),
            nn.ReLU(inplace=True),
        )
        self.out_channels = fpn_channels // 4

    def forward(self, s1, s2, s3, s4):
        p4 = self.smooth4(self.lat4(s4))
        p3 = self.smooth3(self.lat3(s3) + F.interpolate(p4, size=s3.shape[2:],
                                                          mode="nearest"))
        p2 = self.smooth2(self.lat2(s2) + F.interpolate(p3, size=s2.shape[2:],
                                                          mode="nearest"))
        p1 = self.smooth1(self.lat1(s1) + F.interpolate(p2, size=s1.shape[2:],
                                                          mode="nearest"))
        return self.final_up(p1)


# ══════════════════════════════════════════════════════════════════════════════
# Full model — single model, two heads
# ══════════════════════════════════════════════════════════════════════════════

class CBAMSegNet(nn.Module):
    """
    ResNet-34 + CBAM + FPN decoder + two binary segmentation heads.

    Input  : (B, 1, H, W)  — grayscale polar patch
    Outputs: sample_logit  (B, 1, H, W)  — ring material
             pore_logit    (B, 1, H, W)  — pores inside ring

    The two heads share the entire encoder and decoder; they only
    diverge at the final 1x1 conv.  This lets the model learn
    complementary features: the sample head learns the ring boundary,
    the pore head learns locally dark spots within that boundary.

    No sigmoid is applied inside the model.
    Training uses BCEWithLogitsLoss (numerically stable).
    Inference applies sigmoid explicitly in predict.py.
    """

    def __init__(self,
                 pretrained:     bool = True,
                 cbam_reduction: int  = cfg.CBAM_REDUCTION,
                 fpn_channels:   int  = 128):
        super().__init__()
        # Paper architecture
        self.encoder = ResNetCBAMEncoder(pretrained, cbam_reduction)
        # Added for segmentation
        self.decoder = FPNDecoder(fpn_channels)
        out = self.decoder.out_channels
        # Two independent 1x1 heads — one per segmentation task
        self.sample_head = nn.Conv2d(out, 1, kernel_size=1)  # ADDED
        self.pore_head   = nn.Conv2d(out, 1, kernel_size=1)  # ADDED

    def forward(self, x: torch.Tensor):
        s1, s2, s3, s4   = self.encoder(x)
        feat             = self.decoder(s1, s2, s3, s4)
        sample_logit     = self.sample_head(feat)  # (B, 1, H, W)
        pore_logit       = self.pore_head(feat)    # (B, 1, H, W)
        return sample_logit, pore_logit


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(pretrained: bool = True) -> CBAMSegNet:
    """Build the single CBAMSegNet model."""
    return CBAMSegNet(
        pretrained     = pretrained,
        cbam_reduction = cfg.CBAM_REDUCTION,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# Sanity check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = build_model(pretrained=False).to(device)
    x     = torch.randn(2, 1, 256, 256, device=device)
    s, p  = model(x)
    print(f"Input          : {tuple(x.shape)}")
    print(f"Sample logit   : {tuple(s.shape)}")
    print(f"Pore logit     : {tuple(p.shape)}")
    print(f"Trainable params: {count_parameters(model):,}")
