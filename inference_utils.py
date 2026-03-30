"""
inference_utils.py
==================
Utility module for HAM10000 skin lesion classification inference.
Loads all five trained models (HAMCNN, EnhancedCNN, DoubleCNN,
ViT-DINOv2-Small, EfficientNetB3), runs stochastic TTA to produce
per-class probabilities with 95% confidence intervals, and computes
Grad-CAM activation maps on EfficientNetB3.

DISCLAIMER
----------
This software is intended exclusively for research and educational purposes.
It does not constitute a medical device and must not be used as a substitute
for professional clinical diagnosis. The authors accept no liability for any
clinical decision made on the basis of these outputs. Always consult a
qualified dermatologist for medical evaluation.
"""

from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================

import gc
import os
import random
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.cm as mpl_cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =============================================================================
# Configuration (inference-only — no training parameters)
# =============================================================================

CONFIG: dict = {
    # Reproducibility
    "seed": 42,
    # Image preprocessing
    "img_size": 224,
    "norm_mean": [0.485, 0.456, 0.406],
    "norm_std":  [0.229, 0.224, 0.225],
    # Classification
    "num_classes": 7,
    # TTA
    "tta_stochastic_n": 16,   # stochastic dropout passes for CI estimation
    # HAMCNN architecture
    "hamcnn_stages": [
        (32,  2, True),
        (64,  2, True),
        (128, 3, True),
        (256, 3, True),
        (384, 2, True),
    ],
    "hamcnn_head_hidden": 256,
    "hamcnn_dropout": 0.4,
    # EnhancedCNN architecture
    "cnn_channels": [
        (32,  2, True),
        (64,  2, True),
        (128, 3, True),
        (256, 3, True),
        (384, 2, True),
    ],
    "cnn_fpn_proj_dim":      128,
    "cnn_stoch_depth_prob":  0.15,
    "cnn_gem_p":             2.0,
    "cnn_dropout":           0.40,
    # DoubleCNN architecture
    "dcnn_stem_stages": [(32, 2, True), (64, 2, True)],
    "dcnn_cls_stages":  [(128, 3, True), (256, 3, True), (384, 2, True)],
    "dcnn_fpn_proj_dim":      128,
    "dcnn_stoch_depth_prob":  0.15,
    "dcnn_gem_p":             3.0,
    "dcnn_dropout":           0.40,
    "dcnn_recon_lambda":      0.15,
    # ViT architecture
    "vit_model":              "vit_small_patch14_dinov2",
    "vit_dropout":            0.3,
    "vit_head_hidden":        512,
    "vit_grad_checkpointing": False,
    # EfficientNetB3 architecture
    "effnet_model":       "efficientnet_b3",
    "effnet_dropout":     0.4,
    "effnet_head_hidden": 512,
    # Checkpoint paths — adjust if your checkpoints live elsewhere
    "hamcnn_checkpoint":  "processed/best_hamcnn.pth",
    "cnn_checkpoint":     "processed/best_enhanced_cnn.pth",
    "dcnn_checkpoint":    "processed/best_double_cnn.pth",
    "vit_checkpoint":     "processed/best_vit_dinov2.pth",
    "effnet_checkpoint":  "processed/best_efficientnet_b3.pth",
    # Ensemble weights [HAMCNN, EnhancedCNN, DoubleCNN, ViT, EfficientNetB3]
    # Optimised via differential evolution on clinically-weighted recall.
    "ensemble_weights": [0.03, 0.03, 0.14, 0.60, 0.20],
}

# =============================================================================
# Label maps & clinical constants
# =============================================================================

CLASS_NAMES: dict[str, str] = {
    "akiec": "Actinic Keratosis",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesion",
}
LABEL_TO_IDX: dict[str, int] = {
    lbl: idx for idx, lbl in enumerate(sorted(CLASS_NAMES))
}
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}

# Clinical severity weights (index order = alphabetical = LABEL_TO_IDX order).
# akiec=2.0  bcc=2.5  bkl=1.0  df=1.0  mel=3.0  nv=1.0  vasc=1.0
CLINICAL_SEVERITY: np.ndarray = np.array(
    [2.0, 2.5, 1.0, 1.0, 3.0, 1.0, 1.0], dtype=np.float64
)

MODEL_NAMES: list[str] = [
    "HAMCNN", "EnhancedCNN", "DoubleCNN", "ViT-DINOv2", "EfficientNetB3"
]

DISCLAIMER: str = (
    "\u26a0\ufe0f  DISCLAIMER \u2014 Research use only.\n"
    "This tool is intended exclusively for research and educational purposes. "
    "It does not constitute a medical device and must not replace professional "
    "clinical diagnosis. Always consult a qualified dermatologist."
)

GRADCAM_EXPLANATION: str = (
    "The activation map highlights the spatial regions of the image that most "
    "influenced the model\u2019s prediction. Areas shown in red carry the highest "
    "weight in the classification decision; areas in blue had negligible influence. "
    "In a reliable dermatological analysis the highlighted regions should concentrate "
    "on the lesion itself. If strong activation is observed on surrounding skin, "
    "the measurement scale bar, or gel bubbles, the model may be relying on "
    "diagnostically irrelevant artefacts."
)

# =============================================================================
# Device & reproducibility seed
# =============================================================================

def _seed(seed: int = CONFIG["seed"]) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Inference transform (identical to val_transform used during training)
# =============================================================================

_S   = CONFIG["img_size"]
_M   = CONFIG["norm_mean"]
_STD = CONFIG["norm_std"]

inference_transform = A.Compose([
    A.Resize(_S, _S),
    A.Normalize(mean=_M, std=_STD),
    ToTensorV2(),
])

# =============================================================================
# Architecture definitions
# Exact copies of the training notebook classes — do not modify.
# =============================================================================

# ── helpers ───────────────────────────────────────────────────────────────────

def _silu() -> nn.SiLU:
    return nn.SiLU(inplace=True)


# ── StochasticDepth ───────────────────────────────────────────────────────────

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival = 1.0 - self.drop_prob
        noise = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype)
        noise.bernoulli_(survival)
        return x * noise / survival


# ── CBAM ──────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid      = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.act = _silu()
        self.fc2 = nn.Linear(mid, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg   = x.mean(dim=[2, 3])
        mx    = x.amax(dim=[2, 3])
        h_avg = self.act(self.fc1(avg))
        h_mx  = self.act(self.fc1(mx))
        gate  = torch.sigmoid(self.fc2(h_avg) + self.fc2(h_mx))
        return x * gate.view(b, c, 1, 1)

    def init_gate_near_one(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 3.0)


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg  = x.mean(dim=1, keepdim=True)
        mx   = x.amax(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ── ResidualBlock ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            _silu(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act         = _silu()
        self.stoch_depth = StochasticDepth(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.stoch_depth(self.branch(x)))


# ── ConvBlock ─────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int = 2,
        pool: bool = False,
        drop_probs: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if drop_probs is None:
            drop_probs = [0.0] * num_blocks
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            _silu(),
        ]
        for i in range(num_blocks - 1):
            layers.append(ResidualBlock(out_ch, drop_prob=drop_probs[i]))
        layers.append(CBAM(out_ch))
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── GeM Pooling ───────────────────────────────────────────────────────────────

class GeMPool2d(nn.Module):
    def __init__(self, p: float = 2.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0, max=6.0)
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(p), 1).pow(1.0 / p)


# ── FPN-lite Aggregator ───────────────────────────────────────────────────────

class FPNAggregator(nn.Module):
    def __init__(
        self, in_channels: List[int], proj_dim: int, gem_p: float = 2.0
    ) -> None:
        super().__init__()
        self.pools = nn.ModuleList([GeMPool2d(p=gem_p) for _ in in_channels])
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, proj_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(proj_dim),
                _silu(),
            )
            for c in in_channels
        ])

    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for feat, pool, proj in zip(feature_maps, self.pools, self.projs):
            p = pool(feat)
            p = proj(p)
            pooled.append(p.flatten(1))
        return torch.cat(pooled, dim=1)


# ── HAMCNN ────────────────────────────────────────────────────────────────────

class HAMCNN(nn.Module):
    def __init__(
        self,
        stages: list[tuple[int, int, bool]],
        num_classes: int,
        head_hidden: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        feature_stages, in_ch = [], 3
        for out_ch, num_blocks, pool in stages:
            feature_stages.append(self._make_layer(in_ch, out_ch, num_blocks, pool))
            in_ch = out_ch
        self.features   = nn.Sequential(*feature_stages)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, num_classes),
        )

    @staticmethod
    def _make_layer(
        in_ch: int, out_ch: int, num_blocks: int, pool: bool
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks - 1):
            layers.append(HAMCNN._residual_block(out_ch))
        if pool:
            layers.extend([nn.MaxPool2d(2, 2), nn.Dropout2d(0.1)])
        return nn.Sequential(*layers)

    @staticmethod
    def _residual_block(channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── EnhancedCNN ───────────────────────────────────────────────────────────────

class EnhancedCNN(nn.Module):
    _FPN_STAGE_INDICES = [2, 3, 4]

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.4,
        stages: Optional[List[Tuple[int, int, bool]]] = None,
        stoch_depth_prob: float = 0.15,
        gem_p: float = 2.0,
        fpn_proj_dim: int = 128,
    ) -> None:
        super().__init__()
        if stages is None:
            stages = CONFIG["cnn_channels"]

        total_blocks    = sum(n for _, n, _ in stages)
        block_idx, all_probs = 0, []
        for _, n, _ in stages:
            for _ in range(n):
                all_probs.append(
                    stoch_depth_prob * block_idx / max(total_blocks - 1, 1)
                )
                block_idx += 1

        self.stages_list = nn.ModuleList()
        in_ch, prob_cursor = 3, 0
        self._stage_out_ch: List[int] = []

        for out_ch, n_blocks, pool in stages:
            self.stages_list.append(
                ConvBlock(
                    in_ch, out_ch, num_blocks=n_blocks, pool=pool,
                    drop_probs=all_probs[prob_cursor: prob_cursor + n_blocks],
                )
            )
            self._stage_out_ch.append(out_ch)
            in_ch = out_ch
            prob_cursor += n_blocks

        tap_ch   = [self._stage_out_ch[i] for i in self._FPN_STAGE_INDICES]
        self.fpn = FPNAggregator(tap_ch, proj_dim=fpn_proj_dim, gem_p=gem_p)
        head_in  = fpn_proj_dim * len(tap_ch)
        head_mid = max(head_in, 512)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(head_in),
            nn.Dropout(p=dropout),
            nn.Linear(head_in, head_mid, bias=False),
            nn.BatchNorm1d(head_mid),
            _silu(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(head_mid, num_classes),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        taps: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages_list):
            x = stage(x)
            if i in self._FPN_STAGE_INDICES:
                taps.append(x)
        return self.classifier(self.fpn(taps))

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Re-apply CBAM gate init after the generic linear reset (FIX-5)
        for m in self.modules():
            if isinstance(m, ChannelAttention):
                m.init_gate_near_one()


# ── DoubleCNN ─────────────────────────────────────────────────────────────────

class ReconstructionDecoder(nn.Module):
    def __init__(self, in_channels: int = 64) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            _silu(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            _silu(),
            nn.Conv2d(16, 3, kernel_size=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class DoubleCNN(nn.Module):
    _CLS_FPN_INDICES = [0, 1, 2]

    def __init__(
        self,
        num_classes: int,
        stem_stages: Optional[List[Tuple[int, int, bool]]] = None,
        cls_stages:  Optional[List[Tuple[int, int, bool]]] = None,
        fpn_proj_dim: int = 128,
        stoch_depth_prob: float = 0.15,
        gem_p: float = 3.0,
        dropout: float = 0.4,
        recon_lambda: float = 0.15,
    ) -> None:
        super().__init__()
        self.recon_lambda = recon_lambda

        if stem_stages is None:
            stem_stages = CONFIG.get("dcnn_stem_stages", [(32, 2, True), (64, 2, True)])
        if cls_stages is None:
            cls_stages = CONFIG.get("dcnn_cls_stages", [(128, 3, True), (256, 3, True), (384, 2, True)])

        all_stages   = stem_stages + cls_stages
        total_blocks = sum(n for _, n, _ in all_stages)
        block_idx, all_probs = 0, []
        for _, n, _ in all_stages:
            for _ in range(n):
                all_probs.append(
                    stoch_depth_prob * block_idx / max(total_blocks - 1, 1)
                )
                block_idx += 1

        self.stem = nn.ModuleList()
        in_ch, prob_cursor = 3, 0
        self._stem_out_channels: List[int] = []

        for out_ch, n_blocks, pool in stem_stages:
            self.stem.append(
                ConvBlock(in_ch, out_ch, num_blocks=n_blocks, pool=pool,
                          drop_probs=all_probs[prob_cursor: prob_cursor + n_blocks])
            )
            self._stem_out_channels.append(out_ch)
            in_ch = out_ch
            prob_cursor += n_blocks

        bottleneck_ch       = in_ch
        self.cls_stages     = nn.ModuleList()
        self._cls_out_channels: List[int] = []

        for out_ch, n_blocks, pool in cls_stages:
            self.cls_stages.append(
                ConvBlock(in_ch, out_ch, num_blocks=n_blocks, pool=pool,
                          drop_probs=all_probs[prob_cursor: prob_cursor + n_blocks])
            )
            self._cls_out_channels.append(out_ch)
            in_ch = out_ch
            prob_cursor += n_blocks

        tap_channels = [self._cls_out_channels[i] for i in self._CLS_FPN_INDICES]
        self.fpn     = FPNAggregator(tap_channels, proj_dim=fpn_proj_dim, gem_p=gem_p)
        head_in      = fpn_proj_dim * len(tap_channels)
        head_mid     = max(head_in, 512)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(head_in),
            nn.Dropout(p=dropout),
            nn.Linear(head_in, head_mid, bias=False),
            nn.BatchNorm1d(head_mid),
            _silu(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(head_mid, num_classes),
        )
        self.decoder = ReconstructionDecoder(in_channels=bottleneck_ch)

    def forward(self, x: torch.Tensor):
        for stage in self.stem:
            x = stage(x)
        if self.training:
            recon = self.decoder(x)
        fpn_taps: List[torch.Tensor] = []
        for i, stage in enumerate(self.cls_stages):
            x = stage(x)
            if i in self._CLS_FPN_INDICES:
                fpn_taps.append(x)
        logits = self.classifier(self.fpn(fpn_taps))
        if self.training:
            return logits, recon  # type: ignore[return-value]
        return logits


# ── ViTClassifier ─────────────────────────────────────────────────────────────

def _resolve_dinov2_model(preferred: str) -> str:
    candidates = [
        preferred,
        "vit_small_patch14_reg4_dinov2",
        "vit_small_patch14_reg4_dinov2.lvd142m",
    ]
    available = set(timm.list_models("*small*dinov2*"))
    for c in candidates:
        if c in available:
            return c
    if available:
        return next(iter(available))
    raise RuntimeError("No DINOv2-Small model found in timm.")


_VIT_MODEL_NAME: str = _resolve_dinov2_model(CONFIG["vit_model"])


class ViTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float,
        head_hidden: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            _VIT_MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        in_feat = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_feat),
            nn.Linear(in_feat, head_hidden, bias=False),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


# ── EfficientNetB3Classifier ──────────────────────────────────────────────────

class EfficientNetB3Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float,
        head_hidden: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            CONFIG["effnet_model"],
            pretrained=pretrained,
            num_classes=0,
            drop_rate=0.0,
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, head_hidden, bias=False),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


# =============================================================================
# Checkpoint loading with validation
# =============================================================================

def _load_checkpoint(path: str, model: nn.Module, model_name: str) -> None:
    """
    Load weights from a training-notebook checkpoint into model.
    Raises FileNotFoundError immediately if the .pth file is missing.
    Emits a UserWarning if the config hash does not match (architecture
    mismatch is still possible — weights are loaded regardless).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {model_name}: '{path}'\n"
            f"  Resolved path : {p.resolve()}\n"
            f"  Update CONFIG[...checkpoint] if the file lives elsewhere."
        )

    ckpt = torch.load(path, map_location=device, weights_only=False)

    saved_hash   = ckpt.get("config_hash")
    runtime_hash = ckpt.get("CONFIG", {}).get("_config_hash")
    if saved_hash and runtime_hash and saved_hash != runtime_hash:
        warnings.warn(
            f"[{model_name}] Config hash mismatch — "
            f"checkpoint={saved_hash}, runtime={runtime_hash}. "
            f"Weights loaded; verify architecture compatibility.",
            UserWarning,
            stacklevel=3,
        )

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()


def load_all_models(verbose: bool = True) -> dict[str, nn.Module]:
    """
    Instantiate and load all five models from their checkpoints.

    Returns a dict keyed by model name:
        {"HAMCNN": ..., "EnhancedCNN": ..., "DoubleCNN": ...,
         "ViT-DINOv2": ..., "EfficientNetB3": ...}

    Raises FileNotFoundError immediately if any checkpoint is missing so
    the Streamlit UI can surface a clear error message.
    """
    specs = [
        (
            "HAMCNN",
            HAMCNN(
                stages=CONFIG["hamcnn_stages"],
                num_classes=CONFIG["num_classes"],
                head_hidden=CONFIG["hamcnn_head_hidden"],
                dropout=CONFIG["hamcnn_dropout"],
            ),
            CONFIG["hamcnn_checkpoint"],
        ),
        (
            "EnhancedCNN",
            EnhancedCNN(
                num_classes=CONFIG["num_classes"],
                dropout=CONFIG["cnn_dropout"],
                stages=CONFIG["cnn_channels"],
                stoch_depth_prob=CONFIG["cnn_stoch_depth_prob"],
                gem_p=CONFIG["cnn_gem_p"],
                fpn_proj_dim=CONFIG["cnn_fpn_proj_dim"],
            ),
            CONFIG["cnn_checkpoint"],
        ),
        (
            "DoubleCNN",
            DoubleCNN(
                num_classes=CONFIG["num_classes"],
                stem_stages=CONFIG["dcnn_stem_stages"],
                cls_stages=CONFIG["dcnn_cls_stages"],
                fpn_proj_dim=CONFIG["dcnn_fpn_proj_dim"],
                stoch_depth_prob=CONFIG["dcnn_stoch_depth_prob"],
                gem_p=CONFIG["dcnn_gem_p"],
                dropout=CONFIG["dcnn_dropout"],
                recon_lambda=CONFIG["dcnn_recon_lambda"],
            ),
            CONFIG["dcnn_checkpoint"],
        ),
        (
            "ViT-DINOv2",
            ViTClassifier(
                num_classes=CONFIG["num_classes"],
                dropout=CONFIG["vit_dropout"],
                head_hidden=CONFIG["vit_head_hidden"],
                pretrained=False,
            ),
            CONFIG["vit_checkpoint"],
        ),
        (
            "EfficientNetB3",
            EfficientNetB3Classifier(
                num_classes=CONFIG["num_classes"],
                dropout=CONFIG["effnet_dropout"],
                head_hidden=CONFIG["effnet_head_hidden"],
                pretrained=False,
            ),
            CONFIG["effnet_checkpoint"],
        ),
    ]

    models: dict[str, nn.Module] = {}
    for name, model, ckpt_path in specs:
        model = model.to(device)
        _load_checkpoint(ckpt_path, model, name)
        models[name] = model
        if verbose:
            n = sum(p.numel() for p in model.parameters())
            print(f"  \u2713 {name:<18} loaded  ({n:,} params)  \u2190 {ckpt_path}")

    return models


# =============================================================================
# Image preprocessing
# =============================================================================

def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a normalised [1, 3, H, W] tensor ready for inference.
    Handles EXIF rotation metadata and forces RGB colour space.
    """
    img    = ImageOps.exif_transpose(pil_image).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    tensor = inference_transform(image=img_np)["image"]
    return tensor.unsqueeze(0).to(device)


# =============================================================================
# Prediction with stochastic TTA and 95% confidence intervals
# =============================================================================

def _stochastic_passes(
    model: nn.Module,
    tensor: torch.Tensor,
    n: int,
) -> np.ndarray:
    """
    Run n stochastic forward passes with Dropout layers in train() mode
    while BatchNorm layers stay in eval() mode (BN-safe MC Dropout).
    Returns shape (n, num_classes).
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    passes: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n):
            probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
            passes.append(probs)

    model.eval()
    return np.stack(passes)   # (n, C)


def predict_single(
    pil_image: Image.Image,
    models: dict[str, nn.Module],
    n_tta: int = CONFIG["tta_stochastic_n"],
) -> dict:
    """
    Run stochastic TTA inference on a single PIL image with all five models
    and the weighted ensemble.

    Each entry in the returned dict contains:
        probs      : np.ndarray [num_classes]  mean softmax probabilities
        pred_class : int                        argmax class index
        pred_label : str                        short label  (e.g. "mel")
        pred_name  : str                        full name    (e.g. "Melanoma")
        confidence : float                      P(predicted class)
        ci_low     : np.ndarray [num_classes]  2.5th  percentile
        ci_high    : np.ndarray [num_classes]  97.5th percentile
        std        : np.ndarray [num_classes]  standard deviation

    Keys: "HAMCNN", "EnhancedCNN", "DoubleCNN", "ViT-DINOv2",
          "EfficientNetB3", "Ensemble"
    """
    tensor = preprocess(pil_image)
    result: dict = {}
    all_mean_probs: list[np.ndarray] = []

    for name in MODEL_NAMES:
        passes    = _stochastic_passes(models[name], tensor, n_tta)  # (n, C)
        mean_p    = passes.mean(axis=0)
        ci_low    = np.percentile(passes, 2.5,  axis=0)
        ci_high   = np.percentile(passes, 97.5, axis=0)
        std       = passes.std(axis=0)
        pred_cls  = int(mean_p.argmax())
        pred_lbl  = IDX_TO_LABEL[pred_cls]

        result[name] = {
            "probs":      mean_p,
            "pred_class": pred_cls,
            "pred_label": pred_lbl,
            "pred_name":  CLASS_NAMES[pred_lbl],
            "confidence": float(mean_p[pred_cls]),
            "ci_low":     ci_low,
            "ci_high":    ci_high,
            "std":        std,
        }
        all_mean_probs.append(mean_p)

    # Ensemble: weighted average of mean probability vectors
    weights   = CONFIG["ensemble_weights"]
    ens_probs = sum(w * p for w, p in zip(weights, all_mean_probs))
    ens_cls   = int(ens_probs.argmax())
    ens_lbl   = IDX_TO_LABEL[ens_cls]

    # Propagate variance analytically: Var(sum w_i X_i) = sum w_i^2 Var(X_i)
    ens_var   = sum(w**2 * r["std"]**2
                    for w, r in zip(weights, result.values()))
    ens_std   = np.sqrt(ens_var)
    ens_ci_lo = np.clip(ens_probs - 1.96 * ens_std, 0.0, 1.0)
    ens_ci_hi = np.clip(ens_probs + 1.96 * ens_std, 0.0, 1.0)

    result["Ensemble"] = {
        "probs":      ens_probs,
        "pred_class": ens_cls,
        "pred_label": ens_lbl,
        "pred_name":  CLASS_NAMES[ens_lbl],
        "confidence": float(ens_probs[ens_cls]),
        "ci_low":     ens_ci_lo,
        "ci_high":    ens_ci_hi,
        "std":        ens_std,
    }

    return result


# =============================================================================
# Grad-CAM  (EfficientNetB3 — backbone.blocks[6])
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., CVPR 2017).

    Registers forward and gradient hooks on target_layer.
    Always call remove_hooks() after use to prevent memory leaks.

    Usage:
        gc = GradCAM(model, target_layer)
        cam_np, pred_class = gc(tensor)
        gc.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model         = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._fwd_hook     = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, output) -> None:
        output.requires_grad_(True)
        output.retain_grad()
        self._activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach()

    def __call__(self, x: torch.Tensor) -> Tuple[np.ndarray, int]:
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            output     = self.model(x)
            pred_class = int(output.argmax(dim=1).item())
            output[0, pred_class].backward()

        if self._gradients is None:
            raise RuntimeError(
                "GradCAM: gradient hook did not fire. "
                "Verify that target_layer lies on the forward path."
            )

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu(
            (weights * self._activations.detach()).sum(dim=1, keepdim=True)
        )
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        self._activations = None
        self._gradients   = None
        return cam.squeeze().cpu().numpy(), pred_class

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()


def get_gradcam_overlay(
    pil_image: Image.Image,
    effnet_model: nn.Module,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, int]:
    """
    Compute a Grad-CAM overlay for EfficientNetB3.

    The target layer is backbone.blocks[6] — the last inverted-residual
    stage before global average pooling, chosen for its large receptive
    field and high spatial resolution relative to deeper layers.

    Args:
        pil_image    : input PIL image (any resolution).
        effnet_model : loaded EfficientNetB3Classifier in eval() mode.
        alpha        : heatmap blending strength (0 = original, 1 = heatmap).

    Returns:
        overlay    : np.ndarray uint8 H x W x 3 — ready for st.image().
        pred_class : int — argmax class predicted by EfficientNetB3.
    """
    tensor       = preprocess(pil_image)
    target_layer = effnet_model.backbone.blocks[6]
    gradcam      = GradCAM(effnet_model, target_layer)

    cam_np, pred_class = gradcam(tensor)
    gradcam.remove_hooks()

    # Denormalise the preprocessed tensor for display
    mean   = torch.tensor(CONFIG["norm_mean"]).view(3, 1, 1)
    std    = torch.tensor(CONFIG["norm_std"]).view(3, 1, 1)
    img_np = (
        (tensor.squeeze(0).cpu() * std + mean)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )

    cam_resized = (
        np.array(
            Image.fromarray((cam_np * 255).astype(np.uint8))
            .resize((_S, _S), Image.BILINEAR)
        ) / 255.0
    )

    heatmap = mpl_cm.jet(cam_resized)[:, :, :3]
    blended = (1 - alpha) * img_np + alpha * heatmap
    return (blended.clip(0, 1) * 255).astype(np.uint8), pred_class


# =============================================================================
# Memory management
# =============================================================================

def free_models(models: dict[str, nn.Module]) -> None:
    """Delete all model references and release GPU memory."""
    for m in models.values():
        del m
    models.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
