# evaluate.py
# Assumptions: calibration logits = log(mean softmax over flips). Cached *_probs.npz are reused only if newer than best.pt.
# Weight search: DE instead of SLSQP (the argmax objective is piecewise constant, so SLSQP stays at x0); simplex projection inside the objective, then shrink toward uniform.
# tau: "+tau*log(pi)" is the right sign for balanced-sampler training (restores the natural prior) but lowers minority recall; tau=0 is in the grid, so the severity objective can reject it.
# metrics.json = {split: {raw, calibrated}} per model; ensemble test/val metrics go to ensemble_metrics.json. raw = T=1, tau=0 (ensemble keeps fitted weights).
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from data import class_prior, config_hash, eval_hash, load_config, load_splits, make_loader, run_dir, splits_hash
from models import build

FLIP_DIMS = [(), (3,), (2,), (2, 3)]


def _atomic_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _atomic_npz(path: str, **arrays) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _update_json(path: str, key: str, value) -> None:
    d = {}
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
    d[key] = value
    _atomic_json(d, path)


def to_logits(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-12, 1.0))


def load_best(name: str, cfg: dict, path: str, device: torch.device) -> torch.nn.Module:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck["config_hash"] == config_hash(cfg), f"{name}: best.pt from another config"
    assert ck.get("splits_hash") == splits_hash(cfg), f"{name}: best.pt trained on other splits or before the manifest existed; retrain"
    model = build(name, cfg, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    del ck
    return model.to(device).eval()


@torch.no_grad()
def predict_probs(model: torch.nn.Module, loader, device: torch.device, n_tta: int) -> np.ndarray:
    assert 1 <= n_tta <= len(FLIP_DIMS)
    dims = FLIP_DIMS[:n_tta]
    out = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        x = torch.cat([torch.flip(imgs, d) if d else imgs for d in dims])
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x)
        p = F.softmax(logits.float(), dim=1).view(len(dims), imgs.size(0), -1).mean(0)
        out.append(p.cpu())
    return torch.cat(out).numpy()


def run_split(
    split: str, cfg: dict, names: list[str], dfs: dict, rd: str, device: torch.device
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    labels = dfs[split]["label_idx"].to_numpy(np.int64)
    sh = splits_hash(cfg)
    loader = None
    probs: dict[str, np.ndarray] = {}
    for name in names:
        mdir = os.path.join(rd, name)
        best_p, out_p = os.path.join(mdir, "best.pt"), os.path.join(mdir, f"{split}_probs.npz")
        if os.path.exists(out_p) and os.path.getmtime(out_p) >= os.path.getmtime(best_p):
            z = np.load(out_p)
            if (
                "tta_n" in z.files and int(z["tta_n"]) == cfg["tta_n"]
                and "splits_hash" in z.files and str(z["splits_hash"]) == sh
            ):
                assert np.array_equal(z["labels"], labels), f"{name}: cached labels mismatch"
                probs[name] = z["probs"].astype(np.float64)
                continue
        if loader is None:
            loader = make_loader(cfg, split, dfs[split])
        model = load_best(name, cfg, best_p, device)
        p = predict_probs(model, loader, device, cfg["tta_n"])
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _atomic_npz(
            out_p, probs=p.astype(np.float32), labels=labels,
            tta_n=np.int64(cfg["tta_n"]), splits_hash=np.array(sh),
        )
        probs[name] = p.astype(np.float64)
    del loader
    gc.collect()
    return probs, labels


def safe_auc(probs: np.ndarray, labels: np.ndarray, n: int) -> float:
    present = np.unique(labels)
    if len(present) < 2:
        return float("nan")
    try:
        return float(
            roc_auc_score(
                labels,
                probs,
                multi_class="ovr",
                average="macro",
                labels=present,
            )
        )
    except ValueError:
        return float("nan")
    

def sev_recall(preds: np.ndarray, labels: np.ndarray, sev: np.ndarray, n: int) -> float:
    cnt = np.bincount(labels, minlength=n)
    hit = np.bincount(labels[preds == labels], minlength=n)
    return float(sev @ (hit / np.maximum(cnt, 1)))


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    idx = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)  # last bin includes conf==1
    cnt = np.bincount(idx, minlength=n_bins)
    sc = np.bincount(idx, weights=conf, minlength=n_bins)
    sa = np.bincount(idx, weights=correct, minlength=n_bins)
    m = cnt > 0
    return float(np.abs(sc[m] - sa[m]).sum() / len(labels))


def compute_metrics(probs: np.ndarray, labels: np.ndarray, cfg: dict) -> dict:
    n = cfg["num_classes"]
    preds = probs.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(
        labels, preds, labels=list(range(n)), zero_division=0
    )
    nll = -np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)))
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f.mean()),
        "balanced_acc": float(r[s > 0].mean()),
        "sev_recall": sev_recall(preds, labels, np.asarray(cfg["severities"], np.float64), n),
        "auc": safe_auc(probs, labels, n), 
        "ece": ece_score(probs, labels),
        "nll": float(nll),
        "per_class": {
            c: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, c in enumerate(cfg["classes"])
        },
        "confusion": confusion_matrix(labels, preds, labels=list(range(n))).tolist(),
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    rows = np.arange(len(labels))

    def nll(log_t: float) -> float:
        return float(-np.mean(log_softmax(logits / np.exp(log_t), axis=1)[rows, labels]))

    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(r.x))


def fit_calibration(
    probs: dict[str, np.ndarray], labels: np.ndarray, cfg: dict, prior: np.ndarray
) -> dict:
    names = list(probs)
    m, n = len(names), cfg["num_classes"]
    sev = np.asarray(cfg["severities"], np.float64)
    temps = {k: fit_temperature(to_logits(probs[k]), labels) for k in names}
    cal = np.stack([softmax(to_logits(probs[k]) / temps[k], axis=1) for k in names])

    def neg_obj(raw: np.ndarray) -> float:
        w = raw / raw.sum()
        return -sev_recall(np.einsum("m,mnc->nc", w, cal).argmax(axis=1), labels, sev, n)

    res = differential_evolution(
        neg_obj,
        bounds=[(1e-6, 1.0)] * m,
        seed=cfg["seed"],
        maxiter=300,
        popsize=20,
        tol=1e-9,
        polish=False,
    )
    a = cfg["ensemble_shrink"]
    w = (1.0 - a) * (res.x / res.x.sum()) + a / m
    log_ens = to_logits(np.einsum("m,mnc->nc", w, cal))
    log_pi = to_logits(prior)
    lo, hi = cfg["tau_grid"]
    best_tau, best_obj = 0.0, -np.inf
    for t in np.linspace(lo, hi, 31):
        o = sev_recall((log_ens + t * log_pi).argmax(axis=1), labels, sev, n)
        if o > best_obj:  # strict: ties keep the smallest tau
            best_tau, best_obj = float(t), o
    t_ens = fit_temperature(log_ens + best_tau * log_pi, labels)
    print(
        f"val weights={ {k: round(float(v), 3) for k, v in zip(names, w)} } "
        f"T={ {k: round(v, 3) for k, v in temps.items()} } tau={best_tau:.2f} T_ens={t_ens:.3f} obj={best_obj:.4f}"
    )
    return {
        "config_hash": config_hash(cfg),
        "eval_hash": eval_hash(cfg),
        "splits_hash": splits_hash(cfg),
        "models": names,
        "weights": {k: float(v) for k, v in zip(names, w)},
        "temperatures": temps,
        "tau": best_tau,
        "ens_temperature": t_ens,
        "class_prior": prior.tolist(),
        "label_to_idx": {c: i for i, c in enumerate(cfg["classes"])},
        "classes": cfg["classes"],
        "norm_mean": cfg["norm_mean"],
        "norm_std": cfg["norm_std"],
        "severities": cfg["severities"],
        "ensemble_shrink": a,
        "tta_n": cfg["tta_n"],
        "objective_val": float(best_obj),
    }


def apply_pipeline(probs: dict[str, np.ndarray], ens: dict, calibrate: bool = True) -> np.ndarray:
    p = 0.0
    for k in ens["models"]:
        q = softmax(to_logits(probs[k]) / ens["temperatures"][k], axis=1) if calibrate else probs[k]
        p = p + ens["weights"][k] * q
    if not calibrate:
        return p
    z = to_logits(p) + ens["tau"] * to_logits(np.asarray(ens["class_prior"]))
    return softmax(z / ens["ens_temperature"], axis=1)


def report(
    split: str, probs: dict[str, np.ndarray], labels: np.ndarray, ens: dict, cfg: dict, rd: str
) -> None:
    for k in ens["models"]:
        cal = softmax(to_logits(probs[k]) / ens["temperatures"][k], axis=1)
        m = {
            "raw": compute_metrics(probs[k], labels, cfg),
            "calibrated": compute_metrics(cal, labels, cfg),
        }
        _update_json(os.path.join(rd, k, "metrics.json"), split, m)
        print(
            f"{split} {k} f1={m['calibrated']['macro_f1']:.3f} bacc={m['calibrated']['balanced_acc']:.3f} "
            f"ece={m['raw']['ece']:.4f}->{m['calibrated']['ece']:.4f}"
        )
    e = {
        "raw": compute_metrics(apply_pipeline(probs, ens, calibrate=False), labels, cfg),
        "calibrated": compute_metrics(apply_pipeline(probs, ens), labels, cfg),
    }
    _update_json(os.path.join(rd, "ensemble_metrics.json"), split, e)
    print(
        f"{split} ensemble f1={e['calibrated']['macro_f1']:.3f} bacc={e['calibrated']['balanced_acc']:.3f} "
        f"sev_recall={e['calibrated']['sev_recall']:.4f} ece={e['raw']['ece']:.4f}->{e['calibrated']['ece']:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rd = run_dir(cfg)
    ens_p = os.path.join(rd, "ensemble.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    dfs = load_splits(cfg)

    if a.final:
        if not os.path.exists(ens_p):
            raise SystemExit("ensemble.json missing: run evaluate.py (val) first")
        with open(ens_p) as f:
            ens = json.load(f)
        assert ens["config_hash"] == config_hash(cfg), "ensemble.json from another config"
        assert ens.get("splits_hash") == splits_hash(cfg), "ensemble.json fitted on other splits: rerun evaluate.py (val)"
        assert ens.get("eval_hash") == eval_hash(cfg), "ensemble.json fitted with other eval settings: rerun evaluate.py (val)"
        probs, labels = run_split("test", cfg, ens["models"], dfs, rd, device)
        report("test", probs, labels, ens, cfg, rd)
        return

    names = [k for k in cfg["models"] if os.path.exists(os.path.join(rd, k, "best.pt"))]
    assert names, "no trained models found"
    probs, labels = run_split("val", cfg, names, dfs, rd, device)
    prior = class_prior(dfs["train"], cfg["num_classes"])
    ens = fit_calibration(probs, labels, cfg, prior)
    _atomic_json(ens, ens_p)
    report("val", probs, labels, ens, cfg, rd)


if __name__ == "__main__":
    main()
