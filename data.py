# data.py
# Assumptions: CSV columns = filepath,label_idx,lesion_id. config hash excludes "paths" and keys starting with "_".
# OpticalDistortion: shift_limit omitted (not verified in albumentations 2.0.8); library default applies.
# Mixers use the global np.random for lam/box (train.py saves/restores RNG states). mixed_loss lives in losses.py.
from functools import partial
import hashlib
import json
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)

EVAL_ONLY_KEYS = ("tta_n", "tau_grid", "ensemble_shrink", "severities", "class_names")
UNHASHED_KEYS = ("paths", "num_workers") + EVAL_ONLY_KEYS


def _md5(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def config_hash(cfg: dict) -> str:
    return _md5({k: v for k, v in cfg.items() if k not in UNHASHED_KEYS and not k.startswith("_")})


def eval_hash(cfg: dict) -> str:
    return _md5({k: cfg[k] for k in EVAL_ONLY_KEYS})


def splits_hash(cfg: dict) -> str:
    p = cfg["paths"]
    mp = os.path.join(os.path.dirname(p["train_csv"]) or ".", "splits_manifest.json")
    assert os.path.exists(mp), f"{mp} missing: run prepare.py first"
    with open(mp) as f:
        m = json.load(f)
    for s in ("train", "val", "test"):  # catches CSVs edited or replaced after prepare.py
        with open(p[f"{s}_csv"], "rb") as f:
            assert hashlib.md5(f.read()).hexdigest() == m["csv_md5"][s], f"{s} CSV does not match {mp}: rerun prepare.py"
    return m["splits_hash"]

def run_dir(cfg: dict) -> str:
    d = os.path.join(cfg["paths"]["runs_dir"], config_hash(cfg))
    os.makedirs(d, exist_ok=True)
    return d


def load_splits(cfg: dict) -> dict[str, pd.DataFrame]:
    p = cfg["paths"]
    dfs = {s: pd.read_csv(p[f"{s}_csv"]) for s in ("train", "val", "test")}
    for df in dfs.values():
        assert {"filepath", "label_idx", "lesion_id"} <= set(df.columns)
    ids = {s: set(df["lesion_id"]) for s, df in dfs.items()}
    assert not (ids["train"] & ids["val"]), "lesion_id overlap train/val"
    assert not (ids["train"] & ids["test"]), "lesion_id overlap train/test"
    assert not (ids["val"] & ids["test"]), "lesion_id overlap val/test"
    return dfs


def class_prior(train_df: pd.DataFrame, num_classes: int) -> np.ndarray:
    counts = np.bincount(train_df["label_idx"].to_numpy(np.int64), minlength=num_classes)
    return counts / counts.sum()


def build_cache(cfg: dict, split: str, df: pd.DataFrame) -> str:
    s = cfg["img_size"]
    cache_dir = cfg["paths"]["cache_dir"]
    path = os.path.join(cache_dir, f"cache_{s}_{split}.npy")
    shape = (len(df), s, s, 3)
    if os.path.exists(path):
        mm = np.load(path, mmap_mode="r")
        ok = mm.shape == shape
        del mm
        if ok:
            return path
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=shape)
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    for i, fp in enumerate(df["filepath"].tolist()):
        img = cv2.imread(fp, flags)
        if img is None:
            print(f"[WARNING] unreadable image, using black: {fp}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mm[i] = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
    mm.flush()
    del mm
    os.replace(tmp, path)
    return path


def build_transforms(cfg: dict, train: bool) -> A.Compose:
    s = cfg["img_size"]
    norm = [A.Normalize(mean=cfg["norm_mean"], std=cfg["norm_std"]), ToTensorV2()]
    if not train:
        return A.Compose(norm)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.OpticalDistortion(distort_limit=0.2, p=0.2),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.2,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(s // 16, s // 8),
                hole_width_range=(s // 16, s // 8),
                fill=0,
                p=0.3,
            ),
            *norm,
        ], 
        seed=cfg['seed'], 
    )


class CachedDataset(Dataset):
    def __init__(self, path: str, labels: np.ndarray, transform: A.Compose) -> None:
        self.path = path
        self.labels = labels
        self.transform = transform
        self.mm = None  # opened lazily per worker

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        if self.mm is None:
            self.mm = np.load(self.path, mmap_mode="r")
        img = np.array(self.mm[i])
        return self.transform(image=img)["image"], int(self.labels[i])


def make_weighted_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = torch.from_numpy((1.0 / counts)[labels]).float()
    return WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)


def worker_init_fn(worker_id: int, seed: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.transform.set_random_seed(seed + info.id)


def make_loader(cfg: dict, split: str, df: pd.DataFrame, train: bool = False) -> DataLoader:
    labels = df["label_idx"].to_numpy(np.int64)
    ds = CachedDataset(build_cache(cfg, split, df), labels, build_transforms(cfg, train))
    nw = min(cfg["num_workers"], 4)
    kw = dict(
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )
    if train:
        return DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            sampler=make_weighted_sampler(labels, cfg["num_classes"]),
            drop_last=True,
            worker_init_fn=partial(worker_init_fn, seed=cfg["seed"]),
            **kw,
        )
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, **kw)


def mixup(imgs: torch.Tensor, labels: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1.0 - lam) * imgs[perm], labels, labels[perm], lam


def cutmix(imgs: torch.Tensor, labels: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    b, _, h, w = imgs.shape
    perm = torch.randperm(b, device=imgs.device)
    r = np.sqrt(1.0 - lam)
    ch, cw = int(h * r), int(w * r)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(0, cy - ch // 2), min(h, cy + ch // 2)
    x1, x2 = max(0, cx - cw // 2), min(w, cx + cw // 2)
    mixed = imgs.clone()
    mixed[:, :, y1:y2, x1:x2] = imgs[perm, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    return mixed, labels, labels[perm], lam


def build_similarity(cfg: dict, device: torch.device) -> torch.Tensor:
    idx = {c: i for i, c in enumerate(cfg["classes"])}
    n = cfg["num_classes"]
    sim = torch.zeros(n, n, dtype=torch.bool)
    for c, others in cfg["clinical_similarity"].items():
        for o in others:
            sim[idx[c], idx[o]] = True
    return sim.to(device)


def label_aware_mixup(imgs: torch.Tensor, labels: torch.Tensor, alpha: float, sim: torch.Tensor):
    lam = float(np.random.beta(alpha, alpha))
    b = imgs.size(0)
    dev = imgs.device
    elig = sim[labels][:, labels] & ~torch.eye(b, dtype=torch.bool, device=dev)
    scores = torch.rand(b, b, device=dev).masked_fill(~elig, -1.0)
    partner = scores.argmax(dim=1)
    perm = torch.where(elig.any(dim=1), partner, torch.randperm(b, device=dev))
    return lam * imgs + (1.0 - lam) * imgs[perm], labels, labels[perm], lam


def make_mixer(cfg: dict, device: torch.device):
    sim = build_similarity(cfg, device)
    alpha, p_cutmix = cfg["mixup_alpha"], cfg["p_cutmix"]

    def mix(imgs: torch.Tensor, labels: torch.Tensor):
        if np.random.rand() < p_cutmix:
            return cutmix(imgs, labels, alpha)
        return label_aware_mixup(imgs, labels, alpha, sim)

    return mix
