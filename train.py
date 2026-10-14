# train.py
# Assumptions: metrics.json is written by evaluate.py, not here. history.json uses key "grad_norm_preclip" (norm before clipping; plots.py must label it so).
# The tighter grad_clip_unfreeze applies only to the unfreeze epoch, as in the original. SAM models require grad_accum=1.
# A finished run (best.pt + history.json, no last.ckpt) is skipped even without --resume. Main-process RNG is restored on resume; augmentation RNG inside persistent DataLoader workers cannot be, so resume is not bit-exact.
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score

from data import config_hash, load_config, load_splits, make_loader, make_mixer, run_dir, splits_hash
from losses import make_criteria, step_loss
from models import build
from optim import build_optimizer, build_scheduler

HIST_KEYS = ("train_loss", "val_loss", "val_acc", "val_f1", "grad_norm_preclip", "lr")


@dataclass
class EarlyStopping:
    patience: int
    best: float = float("-inf")
    counter: int = 0
    triggered: bool = False

    def update(self, value: float) -> bool:
        if value > self.best:
            self.best, self.counter = value, 0
            return True
        self.counter += 1
        self.triggered = self.counter >= self.patience
        return False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _atomic_torch_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _ac(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _rng_states() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_states(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    if s["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def train_epoch(model, opt, loader, spec: dict, crit, mixer, clip: float, device) -> tuple[float, float]:
    model.train()
    sam = spec["optimizer"] == "sam_adamw"
    ga = spec["grad_accum"]
    assert not sam or ga == 1, "SAM requires grad_accum=1"
    rl = getattr(model, "recon_lambda", 0.0)
    params = list(model.parameters())
    n = len(loader)
    tot = torch.zeros((), device=device)
    cnt = 0
    gns = []
    opt.zero_grad(set_to_none=True)
    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        inp, la, lb, lam = mixer(imgs, labels)
        mix = (la, lb, lam)

        def fwd() -> torch.Tensor:
            with _ac(device):
                out = model(inp)
            return step_loss(out, inp, labels, crit, mix, rl)[0]

        if sam:
            loss = fwd()
            loss.backward()
            gns.append(torch.nn.utils.clip_grad_norm_(params, clip).detach())
            opt.first_step(zero_grad=True)
            fwd().backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.second_step(zero_grad=True)
        else:
            loss = fwd()
            (loss / ga).backward()
            if (i + 1) % ga == 0 or i == n - 1:
                gns.append(torch.nn.utils.clip_grad_norm_(params, clip).detach())
                opt.step()
                opt.zero_grad(set_to_none=True)
        tot += loss.detach().float() * labels.size(0)
        cnt += labels.size(0)
    gn = torch.stack(gns).mean().item() if gns else 0.0
    return tot.item() / cnt, gn


@torch.no_grad()
def validate(model, loader, crit, device) -> tuple[float, float, float]:
    model.eval()
    tot = torch.zeros((), device=device)
    preds, labs = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with _ac(device):
            logits = model(imgs)
        logits = logits.float()
        tot += crit(logits, labels) * labels.size(0)
        preds.append(logits.argmax(1))
        labs.append(labels)
    p = torch.cat(preds).cpu().numpy()
    y = torch.cat(labs).cpu().numpy()
    f1 = float(f1_score(y, p, average="macro", zero_division=0))
    return tot.item() / len(y), float((p == y).mean()), f1


def _save_best(model, name: str, cfg: dict, path: str) -> None:
    _atomic_torch_save(
        {
            "model": name,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "label_to_idx": {c: i for i, c in enumerate(cfg["classes"])},
            "config_hash": config_hash(cfg),
            "splits_hash": splits_hash(cfg),
        },
        path,
    )


def train(name: str, cfg: dict, resume: bool = False) -> dict[str, list[float]] | None:
    spec = cfg["models"][name]
    mdir = os.path.join(run_dir(cfg), name)
    os.makedirs(mdir, exist_ok=True)
    best_p, last_p, hist_p = (os.path.join(mdir, f) for f in ("best.pt", "last.ckpt", "history.json"))

    sh = splits_hash(cfg)
    if os.path.exists(best_p) and os.path.exists(hist_p) and not os.path.exists(last_p):
        done = torch.load(best_p, map_location="cpu", weights_only=True)
        assert done.get("splits_hash") == sh, f"{name}: finished run trained on other splits; delete {mdir} to retrain"
        del done
        print(f"{name}: already finished, skipping")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg["seed"])
    ck = torch.load(last_p, map_location="cpu", weights_only=False) if resume and os.path.exists(last_p) else None
    assert ck is None or ck.get("splits_hash") == sh, f"{name}: last.ckpt from other splits; delete {last_p}"

    dfs = load_splits(cfg)
    train_loader = make_loader(cfg, "train", dfs["train"], train=True)
    val_loader = make_loader(cfg, "val", dfs["val"])
    mixer = make_mixer(cfg, device)
    crit, crit_eval = make_criteria(cfg)

    model = build(name, cfg, pretrained=ck is None).to(device)
    fe = spec["freeze_epochs"]
    start = ck["epoch"] + 1 if ck else 1
    if fe > 0 and start <= fe:
        model.freeze_backbone()
    opt = build_optimizer(model, spec, cfg)
    sched = build_scheduler(opt, cfg)
    es = EarlyStopping(cfg["patience"])
    history: dict[str, list[float]] = {k: [] for k in HIST_KEYS}
    if ck:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        es = EarlyStopping(**ck["es"])
        history = ck["history"]
        _set_rng_states(ck["rng"])
        del ck

    for epoch in range(start, cfg["num_epochs"] + 1):
        if es.triggered:
            break
        seed_torch = cfg["seed"] + epoch
        torch.manual_seed(seed_torch)
        torch.cuda.manual_seed_all(seed_torch)
        clip = spec["grad_clip"]
        if fe and epoch == fe + 1:
            model.unfreeze_backbone()
            clip = spec["grad_clip_unfreeze"]

        tr_loss, gn = train_epoch(model, opt, train_loader, spec, crit, mixer, clip, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        vl_loss, vl_acc, vl_f1 = validate(model, val_loader, crit_eval, device)

        lr = opt.param_groups[-1]["lr"]
        sched.step()
        improved = es.update(vl_f1)
        for k, v in zip(HIST_KEYS, (tr_loss, vl_loss, vl_acc, vl_f1, gn, lr)):
            history[k].append(v)
        print(
            f"{name} ep{epoch:02d} tr={tr_loss:.4f} vl={vl_loss:.4f} acc={vl_acc:.3f} "
            f"f1={vl_f1:.3f} gn_preclip={gn:.3f} lr={lr:.2e}{' *' if improved else ''}"
        )

        if improved:
            _save_best(model, name, cfg, best_p)
        _atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "epoch": epoch,
                "es": asdict(es),
                "best_f1": es.best,
                "splits_hash": sh,
                "history": history,
                "rng": _rng_states(),
            },
            last_p,
        )
        _atomic_json(history, hist_p)

    _atomic_json(history, hist_p)
    if os.path.exists(last_p):
        os.remove(last_p)

    del model, opt, sched, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config", default="config.json")
    a = ap.parse_args()
    cfg = load_config(a.config)
    assert a.model in cfg["models"], f"unknown model: {a.model}"
    train(a.model, cfg, resume=a.resume)


if __name__ == "__main__":
    main()
