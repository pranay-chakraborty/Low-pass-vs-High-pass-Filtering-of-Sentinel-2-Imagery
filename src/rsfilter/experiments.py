"""Experiments E1 to E5 and the decision matrix.

Each experiment returns a ``pandas.DataFrame`` that :func:`run_all` writes to
``<out>/tables/<name>.csv``. Nothing in the report or slides is typed by hand:
every number there is read back from these CSVs.

E1  noise suppression        PSNR / SSIM of every low-pass filter on a noisy crop
E2  land-cover classification Random Forest accuracy on raw / filtered inputs
E3  edge and detail          gradient, sharpness, entropy, Canny edges vs WorldCover boundaries
E4  spectral fidelity        correlation, RMSE, spectral angle, NDVI change against the original
E5  computational cost       run time of spatial vs frequency implementations
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import classify, metrics
from . import filters_freq as ff
from . import filters_spatial as fs
from .bank import FilterSpec, filter_bank
from .preprocess import write_stack

log = logging.getLogger(__name__)
NIR = 3


def _is_lowpass(spec: FilterSpec) -> bool:
    return spec.family.startswith("LPF")


# ----------------------------------------------------------------- E1
def add_noise(clean: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Return the two noisy versions of ``clean`` used in E1 (seed from config).

    * gaussian: ``clean + N(0, sigma^2)`` with sigma = 0.02 reflectance.
    * salt_pepper: 2 % of the pixels of each band set to 0 or 1 (half each),
      imitating dead and saturated detector elements.
    """
    e = cfg["experiments"]
    rng = np.random.default_rng(cfg["seed"])
    gauss = clean + rng.normal(0, e["e1_gauss_sigma"], clean.shape).astype(np.float32)
    sp = clean.copy()
    hit = rng.random(clean.shape) < e["e1_salt_pepper_fraction"]
    sp[hit] = rng.integers(0, 2, hit.sum()).astype(np.float32)
    return {"gaussian": gauss, "salt_pepper": sp}


def e1_noise(clean: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Apply every low-pass filter to both noisy crops and score against the clean crop."""
    rows = []
    for noise, noisy in add_noise(clean, cfg).items():
        rows.append({"noise": noise, "filter": "none (noisy input)", "family": "-",
                     "psnr_db": metrics.psnr(clean, noisy), "ssim": metrics.ssim(clean, noisy)})
        for spec in filter(_is_lowpass, filter_bank(cfg)):
            out = spec.fn(noisy)
            rows.append({"noise": noise, "filter": spec.name, "family": spec.family,
                         "psnr_db": metrics.psnr(clean, out), "ssim": metrics.ssim(clean, out)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- E2
def e2_inputs(cfg: dict, quick: bool = False) -> list[tuple[str, str, object]]:
    """List of (name, family, function returning (image, extra_layers)) compared in E2."""
    bank = filter_bank(cfg)
    chosen = [s for s in bank if _is_lowpass(s) or s.family == "HPF-enhanced"]
    if quick:
        chosen = [s for s in chosen if s.name in {"gauss_s1", "median_3", "boost_n3_k1.0"}]
    inputs = [("raw", "none", lambda img: (img, None))]
    inputs += [(s.name, s.family, lambda img, s=s: (s.fn(img), None)) for s in chosen]
    for n in cfg["filters"]["hpf_sizes"][:1] if quick else cfg["filters"]["hpf_sizes"]:
        inputs.append((f"raw+hpf_{n}", "raw + HPF layers",
                       lambda img, n=n: (img, fs.high_pass(img, n))))
    return inputs


def e2_classification(img: np.ndarray, wc: np.ndarray, cfg: dict, quick: bool = False
                      ) -> tuple[pd.DataFrame, dict, dict]:
    """Train and score one Random Forest per input. Returns (table, predictions, class info)."""
    e = cfg["experiments"]
    labels, names, dropped = classify.regroup(wc, e["e2_classes"], e["e2_min_class_fraction"])
    train = classify.block_split(labels.shape, e["e2_block"], e["e2_train_fraction"], cfg["seed"])
    rows, preds = [], {}
    for name, family, make in e2_inputs(cfg, quick):
        t0 = time.perf_counter()
        image, extra = make(img)
        scores, pred = classify.run_classification(classify.features(image, extra), labels, train, names, cfg)
        rows.append({"input": name, "family": family, **scores, "seconds": time.perf_counter() - t0})
        preds[name] = pred
        log.info("E2 %-16s kappa %.4f", name, scores["kappa"])
    info = {"classes": names, "dropped_classes_fraction": dropped,
            "class_fraction": {n: float((labels == i).mean()) for i, n in enumerate(names)},
            "train_pixels_fraction": float(train.mean()), "labels": labels}
    return pd.DataFrame(rows), preds, info


# ----------------------------------------------------------------- E3 + E4
def _window(img2d: np.ndarray, box: dict, margin: int = 0) -> np.ndarray:
    h = box["size"] // 2 + margin
    return img2d[box["row"] - h:box["row"] + h, box["col"] - h:box["col"] + h]


def e3_row(name: str, family: str, output: str, band: np.ndarray, boundaries: dict, cfg: dict,
           insets: dict | None) -> dict:
    """E3 metrics for one NIR band."""
    e = cfg["experiments"]
    row = {"filter": name, "family": family, "output": output,
           "avg_gradient": metrics.average_gradient(band),
           "laplacian_var": metrics.laplacian_variance(band),
           "entropy_bits": metrics.shannon_entropy(band)}
    if output == "image":  # Canny thresholds are in reflectance units; meaningless for edge layers
        edges = metrics.canny_edges(band, e["canny_sigma"], e["canny_low"], e["canny_high"])
        row["edge_density"] = float(edges.mean())
        for key, ref in boundaries.items():
            p, r, f = metrics.boundary_prf(edges, ref, e["boundary_tolerance"])
            row |= {f"{key}_precision": p, f"{key}_recall": r, f"{key}_f1": f}
    if insets:  # orientation coherence of the filtered band in each sub-area
        rho = e["coherence_rho"]
        for k, v in insets.items():
            m = int(3 * rho) + 1  # margin so the Gaussian window sees real pixels at the window edge
            row[f"coherence_{k}"] = metrics.coherence(_window(band, v, m), rho, m)
    return row


def e4_row(name: str, family: str, output: str, orig: np.ndarray, out: np.ndarray,
           ndvi_orig: np.ndarray) -> dict:
    """E4 metrics: how far a filtered stack moved from the original radiometry."""
    r = metrics.pearson_per_band(orig, out)
    return {"filter": name, "family": family, "output": output,
            **{f"pearson_{b}": v for b, v in zip(["blue", "green", "red", "nir"], r)},
            "pearson_mean": float(np.mean(r)), "rmse": metrics.rmse(orig, out),
            "sam_deg": metrics.spectral_angle(orig, out),
            "ndvi_abs_change": float(np.mean(np.abs(metrics.ndvi(out) - ndvi_orig)))}


def e3_e4(img: np.ndarray, wc: np.ndarray, cfg: dict, insets: dict | None
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run E3 and E4 together so each filter is applied to the full stack only once."""
    labels, names, _ = classify.regroup(wc, cfg["experiments"]["e2_classes"], 0.0)
    valid = labels >= 0
    boundaries = {"boundary": metrics.class_boundaries(labels, valid)}
    if "Built-up" in names:
        built = labels == names.index("Built-up")
        boundaries["builtup_boundary"] = metrics.class_boundaries(built.astype(np.int8), valid)
    nd0 = metrics.ndvi(img)
    rows3 = [e3_row("original", "none", "image", img[NIR], boundaries, cfg, insets)]
    rows4 = []
    for spec in filter_bank(cfg):
        t0 = time.perf_counter()
        out = spec.fn(img)
        rows3.append(e3_row(spec.name, spec.family, spec.output, out[NIR], boundaries, cfg, insets))
        rows4.append(e4_row(spec.name, spec.family, spec.output, img, out, nd0))
        log.info("E3/E4 %-22s %.1f s", spec.name, time.perf_counter() - t0)
    return pd.DataFrame(rows3), pd.DataFrame(rows4)


# ----------------------------------------------------------------- E5
def _median_time(fn, runs: int) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def e5_runtime(band: np.ndarray, cfg: dict, quick: bool = False) -> pd.DataFrame:
    """Median wall-clock time of spatial and frequency implementations on one band.

    Spatial cost grows with the kernel area (n^2 shifted additions); the FFT
    route costs about the same for every cutoff because the work is two FFTs of
    the padded image plus one multiplication. The Gaussian is run in both
    domains with matched parameters (D0 = 1 / (pi sigma)) for a like-for-like
    comparison.
    """
    e, f = cfg["experiments"], cfg["filters"]
    runs = 1 if quick else e["e5_runs"]
    box_sizes = e["e5_box_sizes"][:3] if quick else e["e5_box_sizes"]
    sigmas = e["e5_gauss_sigmas"][:2] if quick else e["e5_gauss_sigmas"]
    pad, order = f["fft_pad"], f["butterworth_order"]
    rows = []

    def add(domain, method, param, size, fn):
        rows.append({"domain": domain, "method": method, "parameter": param, "kernel_side": size,
                     "seconds": _median_time(fn, runs)})

    for n in box_sizes:
        add("spatial", "box mean", f"n={n}", n, lambda n=n: fs.mean_filter(band, n))
    for n in f["median_sizes"]:
        add("spatial", "median", f"n={n}", n, lambda n=n: fs.median_filter(band, n))
    for s in sigmas:
        add("spatial", "gaussian", f"sigma={s}", 2 * int(np.ceil(3 * s)) + 1,
            lambda s=s: fs.gaussian_filter(band, s))
        d0 = 1 / (np.pi * s)
        add("frequency", "gaussian", f"sigma={s} (D0={d0:.3f})", np.nan,
            lambda d0=d0: ff.freq_filter(band, "gaussian", d0, pad=pad))
    for kind in ["ideal", "butterworth"]:
        for d0 in f["freq_cutoffs"]:
            add("frequency", kind, f"D0={d0}", np.nan,
                lambda kind=kind, d0=d0: ff.freq_filter(band, kind, d0, order=order, pad=pad))
    df = pd.DataFrame(rows)
    df.insert(0, "image_shape", f"{band.shape[0]}x{band.shape[1]}")
    return df


# ----------------------------------------------------------------- ringing
def ringing_table(cfg: dict) -> pd.DataFrame:
    """Overshoot of each frequency-domain LPF on a synthetic 0.1 -> 0.3 step edge.

    Scene texture hides ringing, so it is measured on a clean step pushed
    through the same ``freq_filter`` code. The ideal filter should approach the
    Gibbs overshoot of about 9 % of the step height.
    """
    f = cfg["filters"]
    rows = []
    for kind in ["ideal", "butterworth", "gaussian"]:
        for d0 in f["freq_cutoffs"]:
            prof = ff.step_response(kind, d0, f["butterworth_order"], f["fft_pad"])
            rows.append({"mask": kind, "d0": d0, "overshoot_pct_of_step": ff.overshoot_pct(prof)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- decision matrix
def _tradeoff(name: str, e3: pd.DataFrame, e4: pd.DataFrame) -> str:
    """One-line cost summary of a filter, built from the E3 and E4 numbers."""
    if name.startswith("raw"):
        return "no filtering of the image bands"
    r4 = e4.set_index("filter").loc[name]
    g0 = e3.set_index("filter").loc["original", "avg_gradient"]
    g = e3.set_index("filter").loc[name, "avg_gradient"]
    text = (f"NIR average gradient {g / g0:.2f}x original; spectral angle {r4['sam_deg']:.2f} deg; "
            f"mean NDVI change {r4['ndvi_abs_change']:.3f}")
    return ("edge layer, not a reflectance image; " + text) if r4["output"] == "edge" else text


def decision_matrix(e1: pd.DataFrame, e2: pd.DataFrame, e3: pd.DataFrame, e4: pd.DataFrame,
                    cfg: dict) -> pd.DataFrame:
    """Pick the best and runner-up filter per application from E1 to E4.

    Rules (each is a single sort on a measured column):

    * noise removal: highest PSNR in E1, separately for Gaussian and salt-and-pepper noise;
    * land-cover classification: highest Cohen's kappa in E2;
    * lineament / structural mapping: among high-pass outputs, the highest structure-tensor
      coherence in the Aravalli ridge window (E3), i.e. the output whose detail is most
      strongly organised along one direction rather than isotropic speckle;
    * urban boundary delineation: highest F1 of Canny edges against WorldCover built-up
      boundaries (E3, image-type outputs);
    * visual interpretation: highest NIR average gradient among image-type outputs whose
      spectral angle and NDVI change stay within the limits in config (E3 + E4).
    """
    e = cfg["experiments"]
    rows = []

    def add(app, df, key, metric_name, baseline, ascending=False, label_col="filter"):
        df = df.dropna(subset=[key]).sort_values(key, ascending=ascending)
        best, second = df.iloc[0], df.iloc[1] if len(df) > 1 else None
        rows.append({
            "application": app, "best_filter": best[label_col],
            "key_metric": metric_name, "key_metric_value": round(float(best[key]), 4),
            "no_filter_value": round(float(baseline), 4),
            "runner_up": None if second is None else second[label_col],
            "runner_up_value": None if second is None else round(float(second[key]), 4),
            "trade_off": _tradeoff(best[label_col], e3, e4)})

    orig = e3.set_index("filter").loc["original"]
    for noise, title in [("gaussian", "Noise removal (Gaussian noise)"),
                         ("salt_pepper", "Noise removal (salt-and-pepper noise)")]:
        d = e1[e1["noise"] == noise]
        add(title, d[d["family"] != "-"], "psnr_db", "PSNR (dB)", d[d["family"] == "-"]["psnr_db"].iloc[0])
    add("Land-cover classification", e2[e2["input"] != "raw"], "kappa", "Cohen's kappa",
        e2.set_index("input").loc["raw", "kappa"], label_col="input")
    if "coherence_ridge" in e3:
        add("Lineament / structural mapping (Aravalli ridges)", e3[e3["family"].str.startswith("HPF")],
            "coherence_ridge", "orientation coherence on the ridge", orig["coherence_ridge"])
    if "builtup_boundary_f1" in e3:
        add("Urban boundary delineation", e3[e3["filter"] != "original"], "builtup_boundary_f1",
            "built-up boundary F1", orig["builtup_boundary_f1"])
    ok = e4[(e4["output"] == "image") & (e4["sam_deg"] <= e["visual_max_sam_deg"])
            & (e4["ndvi_abs_change"] <= e["visual_max_ndvi_change"])]["filter"]
    add("Visual interpretation", e3[e3["filter"].isin(ok)], "avg_gradient", "NIR average gradient",
        orig["avg_gradient"])
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- driver
def run_all(img: np.ndarray, wc: np.ndarray, profile: dict, cfg: dict, out_dir: Path,
            clean_crop: np.ndarray, insets: dict | None, quick: bool = False) -> dict[str, pd.DataFrame]:
    """Run E1 to E5 and the decision matrix; write CSVs and E2 class maps under ``out_dir``.

    ``img`` (4, H, W) and ``wc`` (H, W) are the scene and WorldCover on the same
    grid; ``clean_crop`` is the 512 x 512 crop used for E1; ``insets`` are the
    ridge / urban / cropland windows (None skips the per-region metrics).
    """
    tables, maps = out_dir / "tables", out_dir / "maps"
    tables.mkdir(parents=True, exist_ok=True)
    maps.mkdir(parents=True, exist_ok=True)

    def save(df, name):
        df.to_csv(tables / f"{name}.csv", index=False, float_format="%.6g")
        log.info("wrote %s", tables / f"{name}.csv")
        return df

    res = {"e1_noise": save(e1_noise(clean_crop, cfg), "e1_noise")}
    e2, preds, info = e2_classification(img, wc, cfg, quick)
    res["e2_classification"] = save(e2, "e2_classification")
    labels = info.pop("labels")
    (tables / "e2_classes.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    lpf = e2[e2["family"].str.startswith("LPF")].sort_values("kappa").iloc[-1]["input"]
    hpf = e2[e2["family"] == "HPF-enhanced"].sort_values("kappa").iloc[-1]["input"]
    write_stack(maps / "worldcover_regrouped.tif", labels, profile)
    for tag, name in [("raw", "raw"), ("best_lpf", lpf), ("best_hpf", hpf)]:
        write_stack(maps / f"e2_pred_{tag}__{name}.tif", preds[name], profile)
    e3, e4 = e3_e4(img, wc, cfg, insets)
    res["e3_edges"], res["e4_fidelity"] = save(e3, "e3_edges"), save(e4, "e4_fidelity")
    res["ringing_overshoot"] = save(ringing_table(cfg), "ringing_overshoot")
    res["e5_runtime"] = save(e5_runtime(img[NIR], cfg, quick), "e5_runtime")
    res["decision_matrix"] = save(decision_matrix(res["e1_noise"], e2, e3, e4, cfg), "decision_matrix")
    return res
