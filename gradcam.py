# gradcam.py
# Assumptions: split defaults to val (test stays reserved for evaluate.py --final; --split test is an explicit opt-in); models = family "transfer"; target layer = last block's norm1 if present (ViT: patch tokens get zero grad at the last block output), else the last block (EfficientNet); selection and titles use the same non-TTA predictions of the displayed model.
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import load_config, load_splits, make_loader, run_dir
from evaluate import load_best


class GradCAM:
    def __init__(self, model: torch.nn.Module, size: int) -> None:
        self.model, self.size = model, size
        self.prefix = getattr(model.backbone, "num_prefix_tokens", 0)
        blk = model.backbone.blocks[-1]
        layer = getattr(blk, "norm1", blk)
        self.act = self.grad = None
        self.handle = layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out) -> None:
        self.act = out.detach()
        out.register_hook(lambda g: setattr(self, "grad", g.detach()))

    def _maps(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        if t.ndim == 4:
            return t
        tok = t[:, self.prefix :, :]
        s = int(round(math.sqrt(tok.size(1))))
        assert s * s == tok.size(1), "token grid is not square"
        return tok.reshape(t.size(0), s, s, -1).permute(0, 3, 1, 2)

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        x = x.detach().requires_grad_(True)  # params are frozen: the graph exists only through the input
        with torch.enable_grad(), torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            logits = self.model(x)
        pred = logits.argmax(1)
        logits.float().gather(1, pred[:, None]).sum().backward()  # samples are independent in eval mode
        a, g = self._maps(self.act), self._maps(self.grad)
        self.act = self.grad = None
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
        mn, mx = cam.amin(dim=(1, 2, 3), keepdim=True), cam.amax(dim=(1, 2, 3), keepdim=True)
        cam = (cam - mn) / (mx - mn).clamp_min(1e-8)
        cam = F.interpolate(cam, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return cam[:, 0].cpu().numpy(), pred.cpu().numpy()

    def remove_hooks(self) -> None:
        self.handle.remove()


def _denorm(img: torch.Tensor, cfg: dict) -> np.ndarray:
    mean = torch.tensor(cfg["norm_mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["norm_std"]).view(3, 1, 1)
    return (img.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay_heatmap(img: torch.Tensor, cam: np.ndarray, cfg: dict, alpha: float = 0.5) -> np.ndarray:
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return ((1.0 - alpha) * _denorm(img, cfg) + alpha * heat).clip(0.0, 1.0)


def show_gradcam_grid(
    imgs: torch.Tensor,
    cams: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    cfg: dict,
    title: str,
    path: str,
) -> None:
    n = len(labels)
    if n == 0:
        print(f"{title}: no samples")
        return
    ncols = min(n, 8)
    blocks = math.ceil(n / ncols)
    fig, axes = plt.subplots(2 * blocks, ncols, figsize=(2.6 * ncols, 5.4 * blocks), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    names = [cfg["class_names"][c][:12] for c in cfg["classes"]]
    for k in range(n):
        b, c = divmod(k, ncols)
        axes[2 * b, c].imshow(_denorm(imgs[k], cfg))
        axes[2 * b, c].set_title(f"True: {names[labels[k]]}", fontsize=7)
        axes[2 * b + 1, c].imshow(overlay_heatmap(imgs[k], cams[k], cfg))
        axes[2 * b + 1, c].set_title(
            f"Pred: {names[preds[k]]}", fontsize=7, color="green" if preds[k] == labels[k] else "red"
        )
    fig.tight_layout()
    tmp = path + ".tmp"
    fig.savefig(tmp, dpi=150, bbox_inches="tight", format="png")
    os.replace(tmp, path)
    plt.show()
    plt.close(fig)


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device) -> np.ndarray:
    out = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(imgs)
        out.append(logits.argmax(1).cpu())
    return torch.cat(out).numpy()


def _explain(
    gc_: GradCAM, ds, idx: np.ndarray, labels: np.ndarray, cfg: dict, device: torch.device, title: str, path: str
) -> None:
    imgs = torch.stack([ds[int(i)][0] for i in idx]) if len(idx) else torch.empty(0)
    cams, preds = [], []
    for s in range(0, len(idx), 16):
        x = imgs[s : s + 16].to(device, memory_format=torch.channels_last)
        c, p = gc_(x)
        cams.append(c)
        preds.append(p)
    if len(idx):
        show_gradcam_grid(imgs, np.concatenate(cams), labels[idx], np.concatenate(preds), cfg, title, path)
    else:
        print(f"{title}: no samples")


def run(
    name: str, cfg: dict, split: str = "val", per_class: int = 2, n_wrong: int = 8,
    device: torch.device | None = None,
) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    rd = run_dir(cfg)
    best_p = os.path.join(rd, name, "best.pt")
    assert os.path.exists(best_p), f"{name}: best.pt missing"
    out_dir = os.path.join(rd, "gradcam")
    os.makedirs(out_dir, exist_ok=True)

    df = load_splits(cfg)[split]
    labels = df["label_idx"].to_numpy(np.int64)
    loader = make_loader(cfg, split, df)
    ds = loader.dataset
    model = load_best(name, cfg, best_p, device)
    for p in model.parameters():
        p.requires_grad_(False)
    preds = predict(model, loader, device)
    del loader
    gc.collect()

    rng = np.random.default_rng(cfg["seed"])
    ok = []
    for c in range(cfg["num_classes"]):
        hits = np.where((labels == c) & (preds == c))[0]
        ok.extend(np.sort(rng.choice(hits, min(len(hits), per_class), replace=False)).tolist())
    bad = np.where(preds != labels)[0]
    bad = np.sort(rng.choice(bad, min(len(bad), n_wrong), replace=False))

    gc_ = GradCAM(model, cfg["img_size"])
    for kind, idx in (("correct", np.asarray(ok, np.int64)), ("wrong", bad)):
        _explain(
            gc_, ds, idx, labels, cfg, device,
            f"Grad-CAM {name} ({split}, {kind} predictions)",
            os.path.join(out_dir, f"{name}_{split}_{kind}.png"),
        )
    gc_.remove_hooks()
    del model, gc_
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--model", action="append")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    names = a.model or [k for k, s in cfg["models"].items() if s["family"] == "transfer"]
    for n in names:
        assert n in cfg["models"], f"unknown model: {n}"
        run(n, cfg, split=a.split)


if __name__ == "__main__":
    main()
