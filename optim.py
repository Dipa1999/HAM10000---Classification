# optim.py
# Assumptions: SAM state_dict nests base_optimizer state under "base_optimizer"; scheduler is stepped per epoch by train.py. Grad clipping stays in train.py (spec["grad_clip"] / spec["grad_clip_unfreeze"]). Transfer models build backbone params even while frozen (requires_grad=False) so unfreeze needs no optimizer rebuild; AdamW skips params with grad None.
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls: type, rho: float = 0.05, **kwargs) -> None:
        assert rho >= 0.0, f"rho must be non-negative, got {rho}"
        super().__init__(params, dict(rho=rho, **kwargs))
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        scale_n = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (scale_n + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.detach().clone()
                p.add_(p.grad * scale)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                old = self.state[p].pop("old_p", None)
                if old is not None:
                    p.copy_(old)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        norms = [
            p.grad.norm(2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.stack(norms).norm(2) if norms else torch.zeros(())

    def state_dict(self) -> dict:
        # old_p is transient (only alive between first_step and second_step) and is excluded
        sd = super().state_dict()
        sd["state"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "old_p"} for k, v in sd["state"].items()
        }
        sd["base_optimizer"] = self.base_optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        sd = dict(state_dict)
        base = sd.pop("base_optimizer")
        super().load_state_dict(sd)
        self.base_optimizer.load_state_dict(base)
        self.param_groups = self.base_optimizer.param_groups

    def step(self, closure=None):  # type: ignore[override]
        raise NotImplementedError("SAM requires first_step() then second_step().")


def _split_decay(model: nn.Module) -> tuple[list, list]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "bias" in name else decay).append(p)
    return decay, no_decay


def build_optimizer(model: nn.Module, spec: dict, cfg: dict) -> torch.optim.Optimizer:
    wd = spec["weight_decay"]
    if spec["optimizer"] == "sam_adamw":
        decay, no_decay = _split_decay(model)
        return SAM(
            [
                {"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            base_optimizer_cls=torch.optim.AdamW,
            rho=cfg["sam_rho"],
            lr=spec["lr"]["backbone"],
        )
    if spec["optimizer"] == "adamw":
        return torch.optim.AdamW(
            model.param_groups(spec["lr"]["backbone"], spec["lr"]["head"]), weight_decay=wd
        )
    raise ValueError(f"unknown optimizer: {spec['optimizer']}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict) -> SequentialLR:
    w, n = cfg["warmup_epochs"], cfg["num_epochs"]
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=w)
    cosine = CosineAnnealingLR(optimizer, T_max=n - w, eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[w])
