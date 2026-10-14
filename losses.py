# losses.py
# Assumptions: hard and mix criteria had identical settings in the original, so one train criterion serves both. Eval criterion is plain CE (gamma=0, smoothing=0). Autocast is opened by train.py/evaluate.py; losses are computed in fp32 here.
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        with torch.no_grad():  # focal weight from the unsmoothed p_t
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        ce = F.cross_entropy(
            logits, targets, label_smoothing=self.label_smoothing, reduction="none"
        )
        return ((1.0 - pt) ** self.gamma * ce).mean()


def make_criteria(cfg: dict) -> tuple[FocalLoss, FocalLoss]:
    train = FocalLoss(cfg["focal_gamma"], cfg["label_smoothing"])
    return train, FocalLoss(0.0, 0.0)


def mixed_loss(
    criterion: nn.Module, logits: torch.Tensor, la: torch.Tensor, lb: torch.Tensor, lam: float
) -> torch.Tensor:
    return lam * criterion(logits, la) + (1.0 - lam) * criterion(logits, lb)


def recon_loss(recon: torch.Tensor, inp: torch.Tensor, recon_lambda: float) -> torch.Tensor:
    # inp must be the exact tensor fed to the encoder (post mixup/cutmix)
    return recon_lambda * F.mse_loss(recon.float(), inp.detach().float())


def step_loss(
    out,
    inp: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    mix: tuple[torch.Tensor, torch.Tensor, float] | None = None,
    recon_lambda: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """out: logits or (logits, recon). mix: (la, lb, lam) when the batch was mixed, else None."""
    logits, recon = out if isinstance(out, tuple) else (out, None)
    if mix is not None:
        la, lb, lam = mix
        loss = mixed_loss(criterion, logits, la, lb, lam)
    else:
        loss = criterion(logits, labels)
    if recon is not None:
        loss = loss + recon_loss(recon, inp, recon_lambda)
    return loss, logits
