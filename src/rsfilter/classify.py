"""Land-cover classification used in experiment E2.

Steps
-----
1. Regroup ESA WorldCover codes into six classes (``config.yaml``,
   ``experiments.e2_classes``) and drop classes covering less than 0.5 % of the
   AOI.
2. Split the AOI into 128 x 128 pixel blocks and assign 70 % of blocks to
   training and 30 % to testing (seed 42). Whole blocks, not single pixels, are
   split because neighbouring pixels are strongly correlated: with a random
   pixel split, a test pixel usually has a near-identical training pixel a few
   metres away, and accuracy is overstated.
3. Draw up to 2,000 training pixels per class from the training blocks.
4. Train a Random Forest (200 trees) on 4 bands + NDVI, computed after filtering,
   and predict the whole AOI.
5. Score on every labelled pixel of the test blocks.

Data structures: ``labels`` int16 (H, W) with class index 0..K-1 and -1 for
pixels without a usable class; ``features`` float32 (H*W, F).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, f1_score

from .metrics import ndvi, salt_pepper_index


def regroup(wc: np.ndarray, classes: dict[str, list[int]], min_fraction: float
            ) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Map WorldCover codes to class indices; drop rare classes.

    Returns ``(labels, kept_class_names, dropped)`` where ``dropped`` maps each
    removed class to its share of the AOI (as a fraction).
    """
    labels = np.full(wc.shape, -1, np.int16)
    kept, dropped = [], {}
    for name, codes in classes.items():
        mask = np.isin(wc, codes)
        frac = float(mask.mean())
        if frac < min_fraction:
            dropped[name] = frac
            continue
        labels[mask] = len(kept)
        kept.append(name)
    return labels, kept, dropped


def block_split(shape: tuple[int, int], block: int, train_fraction: float, seed: int) -> np.ndarray:
    """Boolean (H, W) mask, True for pixels in training blocks."""
    nby, nbx = -(-shape[0] // block), -(-shape[1] // block)  # ceiling division
    rng = np.random.default_rng(seed)
    order = rng.permutation(nby * nbx)
    train_blocks = np.zeros(nby * nbx, bool)
    train_blocks[order[:int(round(train_fraction * nby * nbx))]] = True
    grid = train_blocks.reshape(nby, nbx)
    return np.kron(grid, np.ones((block, block), bool))[:shape[0], :shape[1]]


def features(img: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel features: the 4 bands, NDVI, and optional extra layers. Shape (H*W, F)."""
    layers = [*img, ndvi(img).astype(np.float32)]
    if extra is not None:
        layers += list(extra)
    return np.stack([l.ravel() for l in layers], axis=1).astype(np.float32)


def sample_training(labels: np.ndarray, train_mask: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Flat indices of up to ``per_class`` random training pixels for each class."""
    rng = np.random.default_rng(seed)
    flat_lab, flat_train = labels.ravel(), train_mask.ravel()
    idx = []
    for c in range(labels.max() + 1):
        pool = np.flatnonzero((flat_lab == c) & flat_train)
        idx.append(rng.choice(pool, min(per_class, pool.size), replace=False))
    return np.concatenate(idx)


def run_classification(feat: np.ndarray, labels: np.ndarray, train_mask: np.ndarray,
                       class_names: list[str], cfg: dict) -> tuple[dict, np.ndarray]:
    """Train, predict the full grid, and score on the test blocks.

    Returns ``(scores, prediction)`` where ``prediction`` is int16 (H, W).
    ``scores`` holds overall accuracy, Cohen's kappa, macro F1, per-class F1 and
    the salt-and-pepper index of the predicted map.
    """
    e = cfg["experiments"]
    tr = sample_training(labels, train_mask, e["e2_train_per_class"], cfg["seed"])
    rf = RandomForestClassifier(n_estimators=e["e2_trees"], random_state=cfg["seed"], n_jobs=-1)
    rf.fit(feat[tr], labels.ravel()[tr])
    pred = np.concatenate([rf.predict(feat[i:i + 1_000_000]) for i in range(0, len(feat), 1_000_000)])
    pred = pred.reshape(labels.shape).astype(np.int16)

    test = (~train_mask) & (labels >= 0)
    y, p = labels[test], pred[test]
    k = len(class_names)
    per_class = f1_score(y, p, labels=list(range(k)), average=None, zero_division=0)
    scores = {
        "overall_accuracy": float((y == p).mean()),
        "kappa": float(cohen_kappa_score(y, p)),
        "macro_f1": float(per_class.mean()),
        **{f"f1_{n}": float(v) for n, v in zip(class_names, per_class)},
        "salt_pepper_index": salt_pepper_index(pred, k),
        "n_train": int(tr.size), "n_test": int(test.sum()),
    }
    return scores, pred
