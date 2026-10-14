# prepare.py
# Assumptions: manifest sits next to train_csv (processed/splits_manifest.json, same rule data.splits_hash must use); CSVs carry front columns + eda_columns
# Unverified: not executed; class presence in every split and StratifiedGroupKFold behaviour on the real data (asserts fail loudly); images that PIL decodes but cv2.imread (data.build_cache) rejects would become black in the cache.
# Divergences from legacy (unlisted): per-image brightness/contrast are computed before the split (no cross-image statistic; thresholds are train-only); undecodable images are removed in the clean stage with reason undecodable_image; MD5 chunk is 1 MiB; tqdm without .auto; stratification uses image-level labels instead of lesion-majority labels.
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

SPLITS = ("train", "val", "test")
PREP_KEYS = (
    "metadata_csv", "img_dir_part1", "img_dir_part2", "val_ratio", "test_ratio",
    "split_seed", "drop_train_outliers", "iqr_factor", "eda_columns", 
    "n_folds", "split_tries",
)
DROP_FLAGS = ("bright_outlier", "contrast_outlier")

FLAGS = {"age": "age_outlier", "brightness": "bright_outlier", "contrast": "contrast_outlier"}


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _atomic_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _atomic_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def md5_file(fp: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(fp, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _md5_or_empty(fp: str) -> str:
    try:
        return md5_file(fp)
    except OSError:
        return ""


def manifest_path(cfg: dict) -> str:
    return os.path.join(os.path.dirname(cfg["paths"]["train_csv"]) or ".", "splits_manifest.json")


def build_metadata(cfg: dict, prep: dict) -> pd.DataFrame:
    df = pd.read_csv(prep["metadata_csv"])
    need = {"image_id", "lesion_id", "dx", *prep["eda_columns"]}
    assert need <= set(df.columns), f"metadata missing columns: {sorted(need - set(df.columns))}"
    assert df[["image_id", "lesion_id", "dx"]].notna().all().all(), "null image_id/lesion_id/dx in metadata"
    idx = {c: i for i, c in enumerate(cfg["classes"])}
    unknown = sorted(set(df["dx"]) - set(idx))
    assert not unknown, f"dx values not in config classes: {unknown}"
    df["label_idx"] = df["dx"].map(idx).astype(np.int64)
    print(f"metadata: {len(df)} rows, {df['lesion_id'].nunique()} lesions")
    return df


def _list_images(p: Path) -> list[Path]:
    f = sorted(p.glob("*.jpg"))
    return f or sorted(list(p.glob("*.JPG")) + list(p.glob("*.jpeg")))


def resolve_paths(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    p1, p2 = Path(prep["img_dir_part1"]), Path(prep["img_dir_part2"])
    bad = [str(p.resolve()) for p in (p1, p2) if not p.is_dir()]
    if bad:
        raise FileNotFoundError(
            f"image folders not found: {bad}; fix img_dir_part1/img_dir_part2 in prep_config.json (cwd={Path.cwd()})"
        )
    f1, f2 = _list_images(p1), _list_images(p2)
    if not f1 and not f2:
        raise FileNotFoundError(f"no jpg files in {p1.resolve()} or {p2.resolve()}")
    lookup = {p.stem: p for p in f1}
    both = sum(p.stem in lookup for p in f2)
    lookup.update({p.stem: p for p in f2})  # part2 has priority
    lookup = {k: Path(os.path.relpath(v)).as_posix() for k, v in lookup.items()}
    df = df.copy()
    df["filepath"] = df["image_id"].map(lookup)
    print(
        f"resolve: part1={len(f1)} part2={len(f2)} in_both={both} unique={len(lookup)} "
        f"unresolved={int(df['filepath'].isna().sum())}"
    )
    return df


def _drop(df: pd.DataFrame, mask, reason: str, log: dict, classes: list[str]) -> pd.DataFrame:
    mask = np.asarray(mask, dtype=bool)
    rem = df.loc[mask, "dx"].value_counts()
    log[reason] = {"n": int(mask.sum()), "by_class": {c: int(rem.get(c, 0)) for c in classes}}
    return df.loc[~mask].reset_index(drop=True)


def image_stats(fp: str) -> tuple[float, float]:
    try:
        with Image.open(fp) as im:
            a = np.asarray(im.convert("L"), dtype=np.float32)
        return round(float(a.mean()), 4), round(float(a.std()), 4)
    except Exception:
        return float("nan"), float("nan")


def clean(df: pd.DataFrame, classes: list[str]) -> tuple[pd.DataFrame, dict]:
    log: dict = {}
    n0 = len(df)
    df = _drop(df, df["filepath"].isna(), "missing_filepath", log, classes)
    df = _drop(df, df.duplicated("image_id", keep="first"), "duplicate_image_id", log, classes)
    df["file_hash"] = [_md5_or_empty(fp) for fp in tqdm(df["filepath"], desc="md5")]
    df = _drop(df, df["file_hash"] == "", "unreadable_file", log, classes)
    df = _drop(df, df.duplicated("file_hash", keep="first"), "duplicate_file_md5", log, classes)
    st = [image_stats(fp) for fp in tqdm(df["filepath"], desc="stats")]
    df["brightness"] = [s[0] for s in st]
    df["contrast"] = [s[1] for s in st]
    df = _drop(df, df["brightness"].isna(), "undecodable_image", log, classes)
    df = df.drop(columns="file_hash")
    assert len(df) > 0, "no rows left after cleaning"
    print(f"clean: {n0} -> {len(df)} | " + " ".join(f"{k}={v['n']}" for k, v in log.items()))
    return df, log


def class_counts(parts: dict[str, pd.DataFrame], classes: list[str]) -> dict[str, dict[str, int]]:
    n = len(classes)
    return {
        s: {c: int(v) for c, v in zip(classes, np.bincount(p["label_idx"].to_numpy(np.int64), minlength=n))}
        for s, p in parts.items()
    }


def validate(parts: dict[str, pd.DataFrame], classes: list[str]) -> None:
    ids = {s: set(p["lesion_id"]) for s, p in parts.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = ids[a] & ids[b]
        assert not shared, f"lesion-level leakage: {a}/{b} share {len(shared)} lesion_id"
    cc = class_counts(parts, classes)
    for s in SPLITS:
        absent = [c for c in classes if cc[s][c] == 0]
        assert not absent, f"{s}: classes absent {absent}"
    print("class counts per split:")
    print(f"  {'split':<6}" + "".join(f"{c:>7}" for c in classes) + f"{'total':>8}")
    for s in SPLITS:
        print(f"  {s:<6}" + "".join(f"{cc[s][c]:>7}" for c in classes) + f"{len(parts[s]):>8}")


def _assign(df: pd.DataFrame, k: int, n_val: int, n_test: int, seed: int) -> dict[str, pd.DataFrame]:
    groups = pd.factorize(df["lesion_id"])[0]
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=np.int64)
    for i, (_, te) in enumerate(skf.split(np.zeros(len(df)), df["label_idx"].to_numpy(), groups)):
        fold[te] = i
    m = {"test": fold < n_test, "val": (fold >= n_test) & (fold < n_test + n_val)}
    m["train"] = ~(m["test"] | m["val"])
    return {s: df.loc[m[s]].reset_index(drop=True) for s in SPLITS}


def _split_score(parts: dict[str, pd.DataFrame], df: pd.DataFrame, classes: list[str], ratios: dict[str, float]) -> float:
    n, nc = len(df), len(classes)
    g = np.maximum(np.bincount(df["label_idx"].to_numpy(np.int64), minlength=nc) / n, 1e-9)
    out = 0.0
    for s in SPLITS:
        p = parts[s]
        q = np.bincount(p["label_idx"].to_numpy(np.int64), minlength=nc) / max(len(p), 1)
        out += float(np.mean(np.abs(q / g - 1.0))) + 10.0 * abs(len(p) / n - ratios[s])
    return out


def split(df: pd.DataFrame, prep: dict, classes: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    k = prep["n_folds"]
    n_test, n_val = int(round(prep["test_ratio"] * k)), int(round(prep["val_ratio"] * k))
    assert n_test >= 1 and n_val >= 1 and n_test + n_val < k, "n_folds too small for the ratios"
    ratios = {"test": n_test / k, "val": n_val / k, "train": 1.0 - (n_test + n_val) / k}
    best = None
    for seed in range(prep["split_seed"], prep["split_seed"] + prep["split_tries"]):
        parts = _assign(df, k, n_val, n_test, seed)
        sc = _split_score(parts, df, classes, ratios)
        if best is None or sc < best[0]:
            best = (sc, seed, parts)
    _, seed, parts = best
    total = sum(len(p) for p in parts.values())
    print(
        f"split: folds={k} test={n_test} val={n_val} seed={seed} | "
        + " ".join(f"{s}={len(p)} ({len(p) / total:.1%})" for s, p in parts.items())
    )
    validate(parts, classes)
    return parts, {"folds": k, "test": n_test, "val": n_val, "seed": seed}


def iqr_bounds(s: pd.Series, k: float) -> tuple[float, float]:
    s = s.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return float(q1 - k * (q3 - q1)), float(q3 + k * (q3 - q1))


def add_flags(
    parts: dict[str, pd.DataFrame], prep: dict, classes: list[str], log: dict
) -> tuple[dict[str, pd.DataFrame], dict[str, list[float]], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    th = {c: iqr_bounds(parts["train"][c], prep["iqr_factor"]) for c in FLAGS}  # train only
    out = {
        s: p.assign(**{FLAGS[c]: p[c].notna() & ((p[c] < lo) | (p[c] > hi)) for c, (lo, hi) in th.items()})
        for s, p in parts.items()
    }
    counts = {s: {n: int(p[n].sum()) for n in FLAGS.values()} for s, p in out.items()}
    tr = out["train"]
    lab = tr["label_idx"].to_numpy(np.int64)

    def per_class(m: np.ndarray) -> dict[str, int]:
        return {c: int(v) for c, v in zip(classes, np.bincount(lab[m], minlength=len(classes)))}

    flagged = tr[list(DROP_FLAGS)].any(axis=1).to_numpy()
    by_class = {
        "bright_outlier": per_class(tr["bright_outlier"].to_numpy(bool)),
        "contrast_outlier": per_class(tr["contrast_outlier"].to_numpy(bool)),
        "any_drop_flag": per_class(flagged),
    }
    mask = flagged if prep["drop_train_outliers"] else np.zeros(len(tr), dtype=bool)
    out["train"] = _drop(tr, mask, "train_outlier_flag", log, classes)
    log["train_outlier_flag"]["rows"] = tr.loc[mask, ["image_id", "dx", "filepath", "brightness", "contrast"]].to_dict("records")
    
    print(
        "flags: "
        + " ".join(f"{c}=[{lo:.1f},{hi:.1f}]" for c, (lo, hi) in th.items())
        + " | flagged train/val/test "
        + " ".join(f"{n}={'/'.join(str(counts[s][n]) for s in SPLITS)}" for n in FLAGS.values())
        + f" | train_dropped={log['train_outlier_flag']['n']}"
    )
    return out, {c: [lo, hi] for c, (lo, hi) in th.items()}, counts, by_class


def drop_stale_cache(cfg: dict) -> int:
    d = Path(cfg["paths"]["cache_dir"])
    n = 0
    for s in SPLITS:
        for pat in (f"cache_*_{s}.npy", f"cache_*_{s}.npy.tmp"):
            for f in d.glob(pat):
                f.unlink()
                n += 1
    return n


def write_outputs(parts: dict[str, pd.DataFrame], cfg: dict, prep: dict) -> tuple[dict[str, str], str, int]:
    # manifest removed first and rewritten last: a crash in between leaves no manifest, so train/evaluate fail loudly
    Path(manifest_path(cfg)).unlink(missing_ok=True)
    n_stale = drop_stale_cache(cfg)
    front = ["filepath", "label_idx", "lesion_id", "image_id", "dx"]
    cols = front + list(prep["eda_columns"]) + ["brightness", "contrast"] + list(FLAGS.values())
    md5s = {}
    for s in SPLITS:
        path = cfg["paths"][f"{s}_csv"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        p = parts[s]
        _atomic_csv(p[cols], path)
        md5s[s] = md5_file(path)
    sh = hashlib.md5("|".join(f"{s}={md5s[s]}" for s in SPLITS).encode()).hexdigest()[:8]
    return md5s, sh, n_stale


def run(config_path: str = "config.json", prep_path: str = "prep_config.json") -> dict:
    cfg, prep = load_json(config_path), load_json(prep_path)
    miss = [k for k in PREP_KEYS if k not in prep]
    assert not miss, f"prep_config missing keys: {miss}"
    assert prep["val_ratio"] > 0 and prep["test_ratio"] > 0 and prep["val_ratio"] + prep["test_ratio"] < 1
    classes = cfg["classes"]
    assert "age" in prep["eda_columns"], "eda_columns must contain 'age' (age_outlier flag)"

    df = build_metadata(cfg, prep)
    n_meta = len(df)
    df = resolve_paths(df, prep)
    df, log = clean(df, classes)
    n_clean = len(df)
    parts, n_splits = split(df, prep, classes)
    parts, thresholds, flag_counts, flag_by_class = add_flags(parts, prep, classes, log)
    if prep["drop_train_outliers"]:
        validate(parts, classes)

    md5s, sh, n_stale = write_outputs(parts, cfg, prep)
    total = sum(len(p) for p in parts.values())
    manifest = {
        "splits_hash": sh,
        "csv_md5": md5s,
        "rows": {**{s: len(p) for s, p in parts.items()}, "total": total, "metadata": n_meta, "after_cleaning": n_clean},
        "ratios": {s: round(len(p) / total, 4) for s, p in parts.items()},
        "n_splits": n_splits,
        "class_counts": class_counts(parts, classes),
        "removal_log": log,
        "flag_thresholds": thresholds,
        "flag_counts": flag_counts,
        "train_flag_by_class": flag_by_class,
        "prep_config": {k: v for k, v in prep.items() if not k.startswith("_")},
        "sklearn_version": sklearn.__version__,
    }
    _atomic_json(manifest, manifest_path(cfg))
    print(f"write: 3 csv + manifest | stale cache files removed={n_stale} | splits_hash={sh}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--prep", default="prep_config.json")
    a = ap.parse_args()
    run(a.config, a.prep)


if __name__ == "__main__":
    main()