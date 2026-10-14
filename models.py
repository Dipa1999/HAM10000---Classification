# models.py
# Assumptions: MODELS is keyed by the model name in config["models"]; spec["builder"] is not used for dispatch. Backbone freeze/unfreeze is driven by train.py via freeze_backbone()/unfreeze_backbone(). Channels_last is applied in build().
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _silu() -> nn.SiLU:
    return nn.SiLU(inplace=True)


def _init_weights(model: nn.Module, gate_near_one: bool = False) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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
    if gate_near_one:  # must run after the generic Linear reset
        for m in model.modules():
            if isinstance(m, ChannelAttention):
                nn.init.zeros_(m.fc2.weight)
                nn.init.constant_(m.fc2.bias, 3.0)


def _drop_probs(stages: list, max_prob: float) -> list[float]:
    total = sum(n for _, n, _ in stages)
    return [max_prob * i / max(total - 1, 1) for i in range(total)]


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


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.act = _silu()
        self.fc2 = nn.Linear(mid, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        h_avg = self.act(self.fc1(x.mean(dim=[2, 3])))
        h_mx = self.act(self.fc1(x.amax(dim=[2, 3])))
        gate = torch.sigmoid(self.fc2(h_avg) + self.fc2(h_mx))
        return x * gate.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


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
        self.act = _silu()
        self.stoch_depth = StochasticDepth(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.stoch_depth(self.branch(x)))


class ConvBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, num_blocks: int, pool: bool, drop_probs: list[float]
    ) -> None:
        super().__init__()
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


class GeMPool2d(nn.Module):
    def __init__(self, p: float = 2.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0, max=6.0)
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(p), 1).pow(1.0 / p)


class FPNAggregator(nn.Module):
    def __init__(self, in_channels: list[int], proj_dim: int, gem_p: float) -> None:
        super().__init__()
        self.pools = nn.ModuleList([GeMPool2d(p=gem_p) for _ in in_channels])
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, proj_dim, 1, bias=False),
                    nn.BatchNorm2d(proj_dim),
                    _silu(),
                )
                for c in in_channels
            ]
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        out = [proj(pool(f)).flatten(1) for f, pool, proj in zip(feats, self.pools, self.projs)]
        return torch.cat(out, dim=1)


def _fpn_head(head_in: int, num_classes: int, dropout: float) -> nn.Sequential:
    head_mid = max(head_in, 512)
    return nn.Sequential(
        nn.BatchNorm1d(head_in),
        nn.Dropout(p=dropout),
        nn.Linear(head_in, head_mid, bias=False),
        nn.BatchNorm1d(head_mid),
        _silu(),
        nn.Dropout(p=dropout * 0.5),
        nn.Linear(head_mid, num_classes),
    )


class HAMCNN(nn.Module):
    def __init__(
        self, stages: list, num_classes: int, head_hidden: int, dropout: float
    ) -> None:
        super().__init__()
        feats, in_ch = [], 3
        for out_ch, num_blocks, pool in stages:
            feats.append(self._make_layer(in_ch, out_ch, num_blocks, pool))
            in_ch = out_ch
        self.features = nn.Sequential(*feats)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, num_classes),
        )
        _init_weights(self)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, num_blocks: int, pool: bool) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks - 1):
            # no skip connection: baseline behaviour preserved
            layers.append(
                nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
            )
        if pool:
            layers.extend([nn.MaxPool2d(2, 2), nn.Dropout2d(0.1)])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @classmethod
    def from_cfg(cls, spec: dict, num_classes: int, pretrained: bool) -> "HAMCNN":
        a = spec["arch"]
        return cls(a["stages"], num_classes, a["head_hidden"], spec["dropout"])


class EnhancedCNN(nn.Module):
    _FPN_IDX = [2, 3, 4]

    def __init__(
        self,
        stages: list,
        num_classes: int,
        dropout: float,
        stoch_depth_prob: float,
        gem_p: float,
        fpn_proj_dim: int,
    ) -> None:
        super().__init__()
        probs = _drop_probs(stages, stoch_depth_prob)
        self.stages_list = nn.ModuleList()
        in_ch, cur, out_chs = 3, 0, []
        for out_ch, n, pool in stages:
            self.stages_list.append(ConvBlock(in_ch, out_ch, n, pool, probs[cur : cur + n]))
            out_chs.append(out_ch)
            in_ch, cur = out_ch, cur + n
        tap_ch = [out_chs[i] for i in self._FPN_IDX]
        self.fpn = FPNAggregator(tap_ch, fpn_proj_dim, gem_p)
        self.classifier = _fpn_head(fpn_proj_dim * len(tap_ch), num_classes, dropout)
        _init_weights(self, gate_near_one=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        taps = []
        for i, stage in enumerate(self.stages_list):
            x = stage(x)
            if i in self._FPN_IDX:
                taps.append(x)
        return self.classifier(self.fpn(taps))

    @classmethod
    def from_cfg(cls, spec: dict, num_classes: int, pretrained: bool) -> "EnhancedCNN":
        a = spec["arch"]
        return cls(
            a["stages"],
            num_classes,
            spec["dropout"],
            a["stoch_depth_prob"],
            a["gem_p"],
            a["fpn_proj_dim"],
        )


class ReconstructionDecoder(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            _silu(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            _silu(),
            nn.Conv2d(16, 3, 1, bias=True),  # linear output, no Tanh
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class DoubleCNN(nn.Module):
    _CLS_FPN_IDX = [0, 1, 2]

    def __init__(
        self,
        stem_stages: list,
        cls_stages: list,
        num_classes: int,
        dropout: float,
        stoch_depth_prob: float,
        gem_p: float,
        fpn_proj_dim: int,
        recon_lambda: float,
    ) -> None:
        super().__init__()
        self.recon_lambda = recon_lambda
        probs = _drop_probs(list(stem_stages) + list(cls_stages), stoch_depth_prob)
        in_ch, cur = 3, 0
        self.stem = nn.ModuleList()
        for out_ch, n, pool in stem_stages:
            self.stem.append(ConvBlock(in_ch, out_ch, n, pool, probs[cur : cur + n]))
            in_ch, cur = out_ch, cur + n
        bottleneck_ch = in_ch
        self.cls_stages = nn.ModuleList()
        out_chs = []
        for out_ch, n, pool in cls_stages:
            self.cls_stages.append(ConvBlock(in_ch, out_ch, n, pool, probs[cur : cur + n]))
            out_chs.append(out_ch)
            in_ch, cur = out_ch, cur + n
        tap_ch = [out_chs[i] for i in self._CLS_FPN_IDX]
        self.fpn = FPNAggregator(tap_ch, fpn_proj_dim, gem_p)
        self.classifier = _fpn_head(fpn_proj_dim * len(tap_ch), num_classes, dropout)
        self.decoder = ReconstructionDecoder(bottleneck_ch)
        _init_weights(self)

    def forward(self, x: torch.Tensor):
        for stage in self.stem:
            x = stage(x)
        recon = self.decoder(x) if self.training else None
        taps = []
        for i, stage in enumerate(self.cls_stages):
            x = stage(x)
            if i in self._CLS_FPN_IDX:
                taps.append(x)
        logits = self.classifier(self.fpn(taps))
        return (logits, recon) if self.training else logits

    @classmethod
    def from_cfg(cls, spec: dict, num_classes: int, pretrained: bool) -> "DoubleCNN":
        a = spec["arch"]
        return cls(
            a["stem_stages"],
            a["cls_stages"],
            num_classes,
            spec["dropout"],
            a["stoch_depth_prob"],
            a["gem_p"],
            a["fpn_proj_dim"],
            a["recon_lambda"],
        )


class TransferNet(nn.Module):
    def __init__(
        self, backbone: nn.Module, num_classes: int, dropout: float, head_hidden: int
    ) -> None:
        super().__init__()
        self.backbone = backbone
        in_feat = backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_feat),
            nn.Linear(in_feat, head_hidden, bias=False),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def param_groups(self, lr_backbone: float, lr_head: float) -> list[dict]:
        return [
            {"params": list(self.backbone.parameters()), "lr": lr_backbone},
            {"params": list(self.classifier.parameters()), "lr": lr_head},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def _resolve_dinov2_model(preferred: str) -> str:
    available = sorted(timm.list_models("*small*dinov2*"))
    for c in (preferred, "vit_small_patch14_reg4_dinov2"):
        if c in available:
            return c
    if available:
        return available[0]
    raise RuntimeError("No DINOv2-Small model found in timm")


class ViTClassifier(TransferNet):
    @classmethod
    def from_cfg(cls, spec: dict, num_classes: int, pretrained: bool) -> "ViTClassifier":
        a = spec["arch"]
        backbone = timm.create_model(
            _resolve_dinov2_model(a["timm_name"]),
            pretrained=pretrained and a["pretrained"],
            num_classes=0,
            dynamic_img_size=True,
        )
        return cls(backbone, num_classes, spec["dropout"], a["head_hidden"])


class EfficientNetB3Classifier(TransferNet):
    @classmethod
    def from_cfg(
        cls, spec: dict, num_classes: int, pretrained: bool
    ) -> "EfficientNetB3Classifier":
        a = spec["arch"]
        backbone = timm.create_model(
            a["timm_name"],
            pretrained=pretrained and a["pretrained"],
            num_classes=0,
            drop_rate=0.0,
        )
        return cls(backbone, num_classes, spec["dropout"], a["head_hidden"])


MODELS: dict[str, type[nn.Module]] = {
    "hamcnn": HAMCNN,
    "enhanced_cnn": EnhancedCNN,
    "double_cnn": DoubleCNN,
    "vit_dinov2": ViTClassifier,
    "efficientnet_b3": EfficientNetB3Classifier,
}


def build(name: str, cfg: dict, pretrained: bool = False) -> nn.Module:
    model = MODELS[name].from_cfg(cfg["models"][name], cfg["num_classes"], pretrained)
    return model.to(memory_format=torch.channels_last)
