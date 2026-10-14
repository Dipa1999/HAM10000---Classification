# plots.py
# Assumptions: confusion matrices and per-class F1 are read from metrics.json / ensemble_metrics.json (calibrated variant), so no preds are recomputed; split defaults to val (test is an explicit opt-in, requires evaluate.py --final). Curves need only history.json.
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from data import load_config, run_dir


def _read(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_show(fig, path: str) -> None:
    tmp = path + ".tmp"
    fig.savefig(tmp, dpi=150, bbox_inches="tight", format="png")
    os.replace(tmp, path)
    plt.show()
    plt.close(fig)
    print(f"saved {path}")


def _colors(cfg: dict) -> dict[str, tuple]:
    cm = plt.get_cmap("tab10")
    return {n: cm(i % 10) for i, n in enumerate(cfg["models"])}


def _histories(cfg: dict, rd: str) -> dict[str, dict]:
    h = {n: _read(os.path.join(rd, n, "history.json")) for n in cfg["models"]}
    return {n: v for n, v in h.items() if v and v["val_f1"]}


def _marks(ax, cfg: dict, names: list[str], cols: dict) -> None:
    ax.axvline(cfg["warmup_epochs"], color="grey", ls=":", lw=1.2)
    for n in names:
        fe = cfg["models"][n]["freeze_epochs"]
        if fe > 0:
            ax.axvline(fe + 1, color=cols[n], ls="--", lw=1.0, alpha=0.6)


def plot_curves_per_model(cfg: dict, hist: dict, fdir: str) -> None:
    for n, h in hist.items():
        ep = np.arange(1, len(h["val_f1"]) + 1)
        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"Training curves: {n}", fontweight="bold")
        ax[0, 0].plot(ep, h["train_loss"], label="train")
        ax[0, 0].plot(ep, h["val_loss"], label="val")
        ax[0, 0].set_title("Loss")
        ax[0, 1].plot(ep, h["val_acc"], label="val acc")
        ax[0, 1].plot(ep, h["val_f1"], label="val macro-F1")
        ax[0, 1].set_title("Validation metrics")
        ax[1, 0].plot(ep, h["grad_norm_preclip"], label="grad norm (pre-clip)")
        ax[1, 0].set_title("Grad norm (pre-clip)")
        ax[1, 1].plot(ep, h["lr"], label="lr")
        ax[1, 1].set_yscale("log")
        ax[1, 1].set_title("Learning rate")
        for a in ax.flat:
            _marks(a, cfg, [n], {n: "crimson"})
            a.set_xlabel("Epoch")
            a.grid(True, alpha=0.35)
            a.legend(fontsize=8)
        fig.tight_layout()
        _save_show(fig, os.path.join(fdir, f"curves_{n}.png"))


def plot_curves_compared(cfg: dict, hist: dict, fdir: str) -> None:
    cols = _colors(cfg)
    panels = [
        ("train_loss", "Train loss"),
        ("val_loss", "Val loss"),
        ("val_acc", "Val accuracy"),
        ("val_f1", "Val macro-F1 (early-stop metric)"),
        ("grad_norm_preclip", "Grad norm (pre-clip)"),
        ("lr", "Learning rate"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    fig.suptitle("Training curves: all models", fontweight="bold")
    for ax, (k, t) in zip(axes.flat, panels):
        for n, h in hist.items():
            ax.plot(range(1, len(h[k]) + 1), h[k], label=n, lw=2, color=cols[n])
        _marks(ax, cfg, list(hist), cols)
        if k == "lr":
            ax.set_yscale("log")
        ax.set_title(t, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save_show(fig, os.path.join(fdir, "curves_compared.png"))


def _split_metrics(cfg: dict, rd: str, split: str) -> dict[str, dict]:
    out = {}
    for n in cfg["models"]:
        m = (_read(os.path.join(rd, n, "metrics.json")) or {}).get(split)
        if m:
            out[n] = m["calibrated"]
    e = (_read(os.path.join(rd, "ensemble_metrics.json")) or {}).get(split)
    if e:
        out["ensemble"] = e["calibrated"]
    return out


def plot_confusions(cfg: dict, mets: dict, split: str, fdir: str) -> None:
    cms = {n: np.asarray(m["confusion"], np.float64) for n, m in mets.items()}
    names = [cfg["class_names"][c][:12] for c in cfg["classes"]]
    k = len(cms)
    ncols = min(k, 3)
    nrows = math.ceil(k / ncols)
    for norm in (True, False):
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows), squeeze=False)
        fig.suptitle(
            f"Confusion matrices ({'row-normalised' if norm else 'counts'}, {split}, calibrated)",
            fontweight="bold",
        )
        for ax in axes.flat:
            ax.axis("off")
        for ax, (n, cm) in zip(axes.flat, cms.items()):
            ax.axis("on")
            v = cm / np.maximum(cm.sum(1, keepdims=True), 1.0) if norm else cm
            vmax = 1.0 if norm else max(v.max(), 1.0)
            ax.imshow(v, cmap="Blues", vmin=0.0, vmax=vmax)
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    ax.text(
                        j, i, f"{v[i, j]:.2f}" if norm else f"{int(v[i, j])}",
                        ha="center", va="center", fontsize=8,
                        color="white" if v[i, j] > 0.6 * vmax else "black",
                    )
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_yticklabels(names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(n, fontweight="bold")
        fig.tight_layout()
        _save_show(fig, os.path.join(fdir, f"confusion_{'normalised' if norm else 'absolute'}_{split}.png"))


def plot_per_class_f1(cfg: dict, mets: dict, split: str, fdir: str) -> None:
    f1 = {n: [m["per_class"][c]["f1"] for c in cfg["classes"]] for n, m in mets.items()}
    x = np.arange(len(cfg["classes"]))
    w = 0.8 / len(f1)
    fig, ax = plt.subplots(figsize=(18, 6))
    for i, (n, v) in enumerate(f1.items()):
        ax.bar(x + (i - len(f1) / 2 + 0.5) * w, v, w, label=n, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([cfg["class_names"][c] for c in cfg["classes"]], rotation=15, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.set_title(f"Per-class F1 ({split}, calibrated)", fontweight="bold")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    _save_show(fig, os.path.join(fdir, f"per_class_f1_{split}.png"))


def plot_all(cfg: dict, split: str = "val") -> None:
    rd = run_dir(cfg)
    fdir = os.path.join(rd, "figures")
    os.makedirs(fdir, exist_ok=True)
    hist = _histories(cfg, rd)
    if hist:
        plot_curves_per_model(cfg, hist, fdir)
        plot_curves_compared(cfg, hist, fdir)
    mets = _split_metrics(cfg, rd, split)
    assert mets, f"no {split} metrics found: run evaluate.py{' --final' if split == 'test' else ''} first"
    plot_confusions(cfg, mets, split, fdir)
    plot_per_class_f1(cfg, mets, split, fdir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    a = ap.parse_args()
    plot_all(load_config(a.config), a.split)


if __name__ == "__main__":
    main()
