"""Figures for the report and slides. Each function saves one PNG at 200 dpi and returns its path.

Display rules
-------------
* The 2-98 % percentile stretch is used for display only, never before
  filtering or metrics.
* Panels that are compared share one stretch, taken from the original image.
* Map panels carry a scale bar; every panel is labelled with filter and parameters.
* Charts colour bars by filter family with a fixed colour per family.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from . import filters_freq as ff  # noqa: E402
from .preprocess import read_labels, read_stack, stretch  # noqa: E402

DPI = 200
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "figure.dpi": 100,
    "savefig.bbox": "tight", "savefig.dpi": DPI, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED,
    "axes.spines.top": False, "axes.spines.right": False})

FAMILY_COLOURS = {  # fixed order of the categorical palette; grey for "no filter"
    "LPF-spatial": "#2a78d6", "LPF-freq": "#eb6834", "HPF-spatial": "#1baf7a",
    "HPF-freq": "#eda100", "HPF-enhanced": "#e87ba4", "raw + HPF layers": "#6250d6",
    "none": "#8a8984", "-": "#8a8984"}
CLASS_COLOURS = {"Trees": "#006400", "Shrub/Grass": "#ffbb22", "Cropland": "#f096ff",
                 "Built-up": "#fa0000", "Bare/Mining": "#b4b4b4", "Water": "#0064c8"}
TRUE_COLOUR, FALSE_COLOUR = (2, 1, 0), (3, 2, 1)
NIR = 3


# ----------------------------------------------------------------- helpers
def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def rgb(img: np.ndarray, bands: tuple[int, int, int]) -> np.ndarray:
    """Stack three bands of a ``(4, H, W)`` array into ``(H, W, 3)`` for display."""
    return np.dstack([img[b] for b in bands])


def crop(arr: np.ndarray, box: dict) -> np.ndarray:
    """Cut a square window (centre row/col, size) from the last two axes."""
    h = box["size"] // 2
    return arr[..., box["row"] - h:box["row"] + h, box["col"] - h:box["col"] + h]


def scale_bar(ax, width_px: int, height_px: int, pixel_m: float = 10, length_m: float | None = None) -> None:
    """Black/white scale bar in the lower-left corner of an image axis."""
    if length_m is None:
        target = width_px * pixel_m / 4
        length_m = max([v for v in [100, 200, 250, 500, 1000, 2000, 5000, 10000] if v <= target] or [100])
    px = length_m / pixel_m
    x0, y0 = 0.05 * width_px, 0.93 * height_px
    ax.plot([x0, x0 + px], [y0, y0], color="black", lw=5, solid_capstyle="butt")
    ax.plot([x0, x0 + px], [y0, y0], color="white", lw=2.5, solid_capstyle="butt")
    label = f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m:g} m"
    ax.text(x0 + px / 2, y0 - 0.025 * height_px, label, ha="center", va="bottom", fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 1, "edgecolor": "none"})


def _image_axis(ax, title=None):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)
    if title:
        ax.set_title(title, fontsize=9)


def _stretch_like(img_rgb: np.ndarray, ref_rgb: np.ndarray) -> np.ndarray:
    return stretch(img_rgb, ref=ref_rgb)


def _load(cfg):
    img, profile = read_stack(cfg["paths"]["processed"] / "reflectance.tif")
    wc = read_labels(cfg["paths"]["processed"] / "worldcover.tif")
    meta = json.loads((cfg["paths"]["raw"] / "scene_metadata.json").read_text(encoding="utf-8"))
    return img, wc, meta


def _bank(cfg):
    from .bank import filter_bank
    return {s.name: s for s in filter_bank(cfg)}


# ----------------------------------------------------------------- figure 0: quick-looks
def quicklooks(img: np.ndarray, meta: dict, out_dir: Path) -> list[Path]:
    """True-colour (R-G-B) and false-colour (NIR-R-G) quick-looks of the whole AOI."""
    paths = []
    for name, bands, title in [("quicklook_true_colour.png", TRUE_COLOUR, "True colour (B4-B3-B2)"),
                               ("quicklook_false_colour.png", FALSE_COLOUR, "False colour (B8-B4-B3)")]:
        fig, ax = plt.subplots(figsize=(9, 8.3))
        ax.imshow(stretch(rgb(img, bands)))
        _image_axis(ax, f"{title}, Sentinel-2 L2A {meta['acquisition_date']}, tile {meta['mgrs_tile']}\n"
                        f"2-98 % stretch, display only")
        scale_bar(ax, img.shape[2], img.shape[1], meta["pixel_size_m"], 5000)
        paths.append(_save(fig, out_dir / name))
    return paths


def candidate_sheet(thumbs: list[np.ndarray], rows: list[dict], out: Path) -> Path:
    """Grid of candidate-scene thumbnails labelled with date, flagged AOI % and AOT."""
    cols = 4
    nrows = int(np.ceil(len(thumbs) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.2 * cols, 3.3 * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, th, r in zip(np.ravel(axes), thumbs, rows):
        ax.imshow(th)
        ax.set_title(f"#{r['rank']}  {r['date']}\nflagged {r['bad_pct']:.2f} %  AOT {r['aot_mean']:.3f}",
                     fontsize=9, fontweight="bold" if r["rank"] == 1 else "normal")
    fig.suptitle("Candidate scenes over the AOI (true colour, identical fixed stretch 0-0.3 reflectance)")
    return _save(fig, out)


# ----------------------------------------------------------------- figure 1: study area
def fig_study_area(cfg, img, meta, out: Path) -> Path:
    """India locator, true-colour and false-colour composites with the inset windows marked."""
    outline = json.loads((cfg["paths"]["sample"] / "india_outline.json").read_text(encoding="utf-8"))
    w, s, e, n = cfg["aoi"]["bbox"]
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1, 1], wspace=0.05)
    ax0 = fig.add_subplot(gs[0])
    for ring in outline["rings_lonlat"]:
        xy = np.array(ring)
        ax0.fill(xy[:, 0], xy[:, 1], color="#ecebe7", ec=MUTED, lw=0.6)
    ax0.plot((w + e) / 2, (s + n) / 2, marker="s", ms=7, color="#e34948", mec=INK)
    ax0.annotate("Study area\nGurugram-Faridabad", ((w + e) / 2, (s + n) / 2), xytext=(81, 32.5),
                 fontsize=8, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax0.set_aspect(1 / np.cos(np.radians(22)))
    ax0.set_xlim(67, 98), ax0.set_ylim(6, 37.5)
    ax0.set_xlabel("Longitude (deg E)"), ax0.set_ylabel("Latitude (deg N)")
    ax0.set_title("Location (India, Natural Earth 1:10m)", fontsize=9)
    ax0.grid(color=GRID, lw=0.5)
    H, W = img.shape[1:]
    for k, (bands, title) in enumerate([(TRUE_COLOUR, "True colour B4-B3-B2"),
                                         (FALSE_COLOUR, "False colour B8-B4-B3 (vegetation red)")]):
        ax = fig.add_subplot(gs[k + 1])
        ax.imshow(stretch(rgb(img, bands)))
        _image_axis(ax, f"{title}\nSentinel-2 L2A {meta['acquisition_date']}, "
                        f"{w}-{e} E, {s}-{n} N, UTM 43N")
        scale_bar(ax, W, H, 10, 5000)
        if k == 1:
            for name, box in cfg["insets"].items():
                h = box["size"] // 2
                ax.add_patch(Rectangle((box["col"] - h, box["row"] - h), box["size"], box["size"],
                                       fill=False, ec="yellow", lw=1.5))
                ax.text(box["col"] - h, box["row"] - h - 25, name, color="yellow", fontsize=8,
                        fontweight="bold")
    return _save(fig, out)


# ----------------------------------------------------------------- figure 2: result grids
def _grid(cfg, img, names, out, title, mode):
    """Rows = filters; columns = full AOI + the three insets.

    ``mode`` 'false' shows NIR-R-G with the stretch of the original; 'edge'
    shows the NIR band of an edge layer in grey with its own 2-98 % stretch.
    """
    bank = _bank(cfg)
    insets = cfg["insets"]
    cols = ["AOI"] + list(insets)
    ref_full = rgb(img, FALSE_COLOUR)
    fig, axes = plt.subplots(len(names), len(cols), figsize=(2.6 * len(cols), 2.6 * len(names)))
    for i, name in enumerate(names):
        out_img = img if name == "original" else bank[name].fn(img)
        label = "Original" if name == "original" else bank[name].label
        for j, col in enumerate(cols):
            ax = axes[i, j]
            if mode == "false" or name == "original":
                full = rgb(out_img, FALSE_COLOUR)
                if col == "AOI":
                    ax.imshow(_stretch_like(full, ref_full))
                else:
                    ax.imshow(_stretch_like(crop(full.transpose(2, 0, 1), insets[col]).transpose(1, 2, 0),
                                            crop(ref_full.transpose(2, 0, 1), insets[col]).transpose(1, 2, 0)),
                              interpolation="nearest")
            else:
                band = out_img[NIR]
                lo, hi = np.percentile(band, [2, 98])
                view = band if col == "AOI" else crop(band, insets[col])
                ax.imshow(np.clip((view - lo) / (hi - lo), 0, 1), cmap="gray", interpolation="nearest")
            _image_axis(ax)
            n_px = out_img.shape[2] if col == "AOI" else insets[col]["size"]
            scale_bar(ax, n_px, out_img.shape[1] if col == "AOI" else n_px, 10)
            if i == 0:
                ax.set_title(col if col == "AOI" else f"{col} ({insets[col]['size'] * 10 / 1000:g} km)",
                             fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return _save(fig, out)


def fig_lpf_grid(cfg, img, out):
    """Low-pass results (spatial and frequency domain) with the three zoomed insets."""
    names = ["original", "box_5", "gauss_s2", "median_5", "fft_ideal_lp_0.2", "fft_butter_lp_0.2",
             "fft_gauss_lp_0.2"]
    return _grid(cfg, img, names, out, "Low-pass filters (false colour NIR-R-G, stretch of the original)",
                 "false")


def fig_hpf_grid(cfg, img, out):
    """High-pass edge layers on the NIR band with the three zoomed insets."""
    names = ["original", "hpf_3", "hpf_7", "laplace_8", "sobel", "fft_gauss_hp_0.1", "fft_ideal_hp_0.2"]
    return _grid(cfg, img, names, out, "High-pass filters (NIR band; each edge layer has its own 2-98 % "
                                       "grey stretch)", "edge")


def fig_boost_grid(cfg, img, out):
    """High-boost (original + k x HPF) results with the three zoomed insets."""
    names = ["original", "boost_n3_k0.5", "boost_n3_k1.0", "boost_n5_k1.0", "boost_n7_k1.0", "boost_n7_k1.5"]
    return _grid(cfg, img, names, out, "High-boost edge enhancement, original + k x HPF "
                                       "(false colour, stretch of the original)", "false")


def fig_slide_grid(cfg, img, names, out, mode, regions=("ridge", "urban")):
    """Landscape grid for slides: rows = sub-areas, columns = filters (same display rules as figure 2)."""
    bank = _bank(cfg)
    insets = cfg["insets"]
    fig, axes = plt.subplots(len(regions), len(names), figsize=(2.6 * len(names), 2.75 * len(regions)))
    for j, name in enumerate(names):
        out_img = img if name == "original" else bank[name].fn(img)
        for i, region in enumerate(regions):
            ax = axes[i, j]
            box = insets[region]
            if mode == "false" or name == "original":
                view = rgb(crop(out_img, box), FALSE_COLOUR)
                ax.imshow(_stretch_like(view, rgb(crop(img, box), FALSE_COLOUR)), interpolation="nearest")
            else:
                band = out_img[NIR]
                lo, hi = np.percentile(band, [2, 98])
                ax.imshow(np.clip((crop(band, box) - lo) / (hi - lo), 0, 1), cmap="gray", interpolation="nearest")
            _image_axis(ax)
            scale_bar(ax, box["size"], box["size"])
            if i == 0:
                ax.set_title("Original" if name == "original" else bank[name].label, fontsize=10)
            if j == 0:
                ax.set_ylabel(region, fontsize=11, fontweight="bold")
    fig.tight_layout(pad=0.4)
    return _save(fig, out)


# ----------------------------------------------------------------- figure 3: spectra
def fig_spectra(band: np.ndarray, cfg, out: Path) -> Path:
    """Log-magnitude spectra of the NIR crop before and after low- and high-pass filtering.

    The "after" spectra are the original spectrum multiplied by the mask, i.e.
    exactly what the filter keeps. Re-transforming the filtered image instead
    would add a cross-shaped artefact from its non-periodic borders.
    """
    pad = cfg["filters"]["fft_pad"]
    spec = np.fft.fftshift(np.fft.fft2(band - band.mean()))
    lp = ff.make_mask("gaussian", band.shape, 0.1)
    cases = [("Original", band, spec),
             ("Gaussian LPF D0=0.1", ff.freq_filter(band, "gaussian", 0.1, pad=pad), spec * lp),
             ("Gaussian HPF D0=0.1", ff.freq_filter(band, "gaussian", 0.1, highpass=True, pad=pad), spec * (1 - lp))]
    ref = np.log1p(np.abs(spec))
    vmin, vmax = np.percentile(ref, [1, 99.9])
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
    lo, hi = np.percentile(band, [2, 98])
    for j, (title, img, s) in enumerate(cases):
        view = (img - lo) / (hi - lo) if j < 2 else img / (2 * np.percentile(np.abs(img), 98)) + 0.5
        axes[0, j].imshow(np.clip(view, 0, 1), cmap="gray")
        _image_axis(axes[0, j], f"{title}: NIR image")
        scale_bar(axes[0, j], band.shape[1], band.shape[0])
        axes[1, j].imshow(np.log1p(np.abs(s)), cmap="magma", vmin=vmin, vmax=vmax,
                          extent=[-0.5, 0.5, 0.5, -0.5])
        axes[1, j].set_title(f"{title}: log(1 + |F(u,v) H(u,v)|)" if j else "Original: log(1 + |F(u,v)|)",
                             fontsize=9)
        axes[1, j].set_xlabel("v (cycles / pixel)")
        if j == 0:
            axes[1, j].set_ylabel("u (cycles / pixel)")
    fig.suptitle("Fourier spectra, 512 x 512 NIR crop (centre = zero frequency, mean removed, same colour scale)")
    fig.tight_layout()
    return _save(fig, out)


# ----------------------------------------------------------------- figure 4: slide replication
def fig_square_masks(band: np.ndarray, cfg, out: Path) -> Path:
    """Square ideal low-pass masks of 61, 91, 121 pixels on the 512 x 512 crop (course slide layout)."""
    sides = cfg["filters"]["square_mask_sides"]
    fig, axes = plt.subplots(2, len(sides) + 1, figsize=(3.3 * (len(sides) + 1), 6.8))
    lo, hi = np.percentile(band, [2, 98])
    axes[0, 0].imshow(ff.log_spectrum(band), cmap="magma")
    _image_axis(axes[0, 0], "Spectrum of the original")
    axes[1, 0].imshow(np.clip((band - lo) / (hi - lo), 0, 1), cmap="gray")
    _image_axis(axes[1, 0], "Original NIR, 512 x 512")
    scale_bar(axes[1, 0], 512, 512)
    for j, side in enumerate(sides, start=1):
        axes[0, j].imshow(ff.square_mask(band.shape, side), cmap="gray", vmin=0, vmax=1)
        _image_axis(axes[0, j], f"Mask {side} x {side} (white = pass)")
        axes[1, j].imshow(np.clip((ff.square_lowpass(band, side) - lo) / (hi - lo), 0, 1), cmap="gray")
        _image_axis(axes[1, j], f"Result, {side} x {side} mask")
        scale_bar(axes[1, j], 512, 512)
    fig.suptitle("Frequency-domain low-pass filtering with square ideal masks (replicating the course slide)")
    fig.tight_layout()
    return _save(fig, out)


# ----------------------------------------------------------------- figure 5: ringing
def fig_ringing(cfg, img, out: Path) -> Path:
    """Ideal vs Butterworth vs Gaussian LPF (D0 = 0.1): real zoom (top) and synthetic step response (bottom)."""
    pad, order = cfg["filters"]["fft_pad"], cfg["filters"]["butterworth_order"]
    box = cfg["ringing_window"]
    margin = 3 * box["size"]  # filter a larger window so the zoom is unaffected by its own borders
    big = {"row": box["row"], "col": box["col"], "size": box["size"] + 2 * margin}
    band = crop(img[NIR], big)
    kinds = [("ideal", "Ideal"), ("butterworth", f"Butterworth n={order}"), ("gaussian", "Gaussian")]
    cases = [("Original", band)] + [(f"{lab} LPF D0=0.1", ff.freq_filter(band, kind, 0.1, order=order, pad=pad))
                                    for kind, lab in kinds]
    c = big["size"] // 2
    sl = slice(c - box["size"] // 2, c + box["size"] // 2)
    lo, hi = np.percentile(cases[0][1][sl, sl], [2, 98])
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), gridspec_kw={"height_ratios": [1, 0.75]})
    for j, (title, arr) in enumerate(cases):
        axes[0, j].imshow(np.clip((arr[sl, sl] - lo) / (hi - lo), 0, 1), cmap="gray", interpolation="nearest")
        _image_axis(axes[0, j], f"{title} (real NIR)")
        scale_bar(axes[0, j], box["size"], box["size"])
    x = np.arange(256) - 128
    step = np.where(x < 0, 0.1, 0.3)
    axes[1, 0].plot(x, step, color="#8a8984", lw=2)
    axes[1, 0].set_title("Synthetic step edge, 0.1 to 0.3", fontsize=9)
    colours = ["#2a78d6", "#eb6834", "#1baf7a"]
    for j, ((kind, lab), colour) in enumerate(zip(kinds, colours), start=1):
        prof = ff.step_response(kind, 0.1, order, pad)
        axes[1, j].plot(x, step, color="#8a8984", lw=1)
        axes[1, j].plot(x, prof, color=colour, lw=2)
        axes[1, j].set_title(f"{lab}: overshoot {ff.overshoot_pct(prof):.1f} % of step", fontsize=9)
    for ax in axes[1]:
        ax.set_xlim(-40, 40), ax.set_ylim(0.05, 0.35)
        ax.set_xlabel("pixels from the edge")
        ax.grid(color=GRID, lw=0.5)
    axes[1, 0].set_ylabel("value")
    fig.suptitle("Ringing test: the ideal mask's sharp cut makes ripples beside edges; "
                 "smooth masks overshoot less (bottom row: same filter code on a clean step)")
    fig.tight_layout()
    return _save(fig, out)


# ----------------------------------------------------------------- figure 6: line profile
def fig_line_profile(cfg, img, out: Path) -> Path:
    """NIR line profile across a sharp edge: original vs low-pass vs high-boost."""
    bank = _bank(cfg)
    p = cfg["profile"]
    r, c0, c1 = p["row"], p["col_start"], p["col_end"]
    pad = 64
    window = img[:, r - pad:r + pad, c0 - pad:c1 + pad]
    lines = {"original": window[NIR, pad, pad:-pad]}
    for name in ["box_5", "gauss_s2", "median_5", "boost_n3_k1.0", "boost_n7_k1.0"]:
        lines[name] = bank[name].fn(window[NIR])[pad, pad:-pad]
    styles = {"original": ("#0b0b0b", "-", 2.2), "box_5": ("#2a78d6", "-", 1.6),
              "gauss_s2": ("#2a78d6", "--", 1.6), "median_5": ("#2a78d6", ":", 1.8),
              "boost_n3_k1.0": ("#e87ba4", "-", 1.6), "boost_n7_k1.0": ("#e87ba4", "--", 1.6)}
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.6], wspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    view = img[:, r - 80:r + 80, c0:c1]
    ax0.imshow(stretch(rgb(view, FALSE_COLOUR)), interpolation="nearest")
    ax0.axhline(80, color="yellow", lw=1.2)
    _image_axis(ax0, f"Profile line (row {r}, cols {c0}-{c1})")
    scale_bar(ax0, c1 - c0, 160)
    ax = fig.add_subplot(gs[1])
    x = np.arange(c1 - c0) * 10
    for name, y in lines.items():
        col, ls, lw = styles[name]
        ax.plot(x, y, color=col, ls=ls, lw=lw, label="Original" if name == "original" else bank[name].label)
    ax.set_xlabel("distance along line (m)"), ax.set_ylabel("NIR reflectance")
    ax.grid(color=GRID, lw=0.5)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper right")
    ax.set_title("Low-pass filters flatten and widen the edge; high-boost steepens it and adds over/undershoot")
    return _save(fig, out)


# ----------------------------------------------------------------- figure 7: class maps
def fig_class_maps(cfg, out: Path) -> Path:
    """WorldCover (regrouped) next to the raw, best-LPF and best-HPF Random Forest maps."""
    maps = cfg["paths"]["maps"]
    info = json.loads((cfg["paths"]["tables"] / "e2_classes.json").read_text(encoding="utf-8"))
    e2 = pd.read_csv(cfg["paths"]["tables"] / "e2_classification.csv").set_index("input")
    names = info["classes"]
    cmap = ListedColormap(["white"] + [CLASS_COLOURS[n] for n in names])
    panels = [("WorldCover 2021 (regrouped reference)", maps / "worldcover_regrouped.tif", None)]
    for tag in ["raw", "best_lpf", "best_hpf"]:
        path = next(maps.glob(f"e2_pred_{tag}__*.tif"))
        name = path.stem.split("__", 1)[1]
        panels.append((name, path, e2.loc[name]))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))
    for ax, (title, path, row) in zip(axes, panels):
        lab = read_labels(path).astype(np.int16)
        lab = np.where(lab > 32000, -1, lab)  # -1 stored as int16; guard against unsigned reads
        ax.imshow(lab + 1, cmap=cmap, vmin=0, vmax=len(names), interpolation="nearest")
        if row is not None:
            title = (f"Random Forest on: {title}\nOA {row['overall_accuracy']:.3f}, kappa {row['kappa']:.3f}, "
                     f"salt-and-pepper {row['salt_pepper_index']:.3f}")
        _image_axis(ax, title)
        scale_bar(ax, lab.shape[1], lab.shape[0], 10, 5000)
    fig.legend(handles=[Patch(color=CLASS_COLOURS[n], label=n) for n in names] +
               [Patch(facecolor="white", edgecolor=MUTED, label="not used")],
               loc="lower center", ncol=len(names) + 1, frameon=False, bbox_to_anchor=(0.5, 0.1))
    fig.suptitle("Land-cover maps from the spatially blocked Random Forest experiment (E2)")
    return _save(fig, out)


# ----------------------------------------------------------------- figure 8: charts
def _barh(ax, labels, values, families, fmt="{:.3f}", annotate=True):
    y = np.arange(len(labels))
    ax.barh(y, values, color=[FAMILY_COLOURS.get(f, "#8a8984") for f in families], height=0.72,
            edgecolor="white", linewidth=1)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    if annotate:
        best = int(np.nanargmax(values))
        ax.text(values[best], best, " " + fmt.format(values[best]), va="center", fontsize=8, color=INK)


def _family_legend(fig, families):
    seen = [f for f in FAMILY_COLOURS if f in set(families) and f != "-"]
    fig.legend(handles=[Patch(color=FAMILY_COLOURS[f], label="no filter" if f in ("none", "-") else f)
                        for f in seen], loc="lower center", ncol=len(seen), frameon=False,
               bbox_to_anchor=(0.5, -0.02))


def fig_e1(e1: pd.DataFrame, out: Path) -> Path:
    """E1 chart: PSNR of every low-pass filter for both noise types."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    for ax, (noise, title) in zip(axes, [("gaussian", "Gaussian noise, sigma = 0.02"),
                                         ("salt_pepper", "Salt-and-pepper noise, 2 %")]):
        d = e1[e1["noise"] == noise]
        _barh(ax, d["filter"].tolist(), d["psnr_db"].to_numpy(), d["family"].tolist(), "{:.2f} dB")
        ax.set_title(title)
        ax.set_xlabel("PSNR against the clean crop (dB, higher is better)")
        ax.set_xlim(d["psnr_db"].min() - 3, d["psnr_db"].max() + 3)
    _family_legend(fig, e1["family"])
    fig.suptitle("E1 Noise suppression on the 512 x 512 crop (grey bar = noisy input, no filter)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out)


def fig_e2(e2: pd.DataFrame, out: Path) -> Path:
    """E2 chart: Cohen's kappa and salt-and-pepper index per classification input."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), sharey=True)
    labels = e2["input"].tolist()
    _barh(axes[0], labels, e2["kappa"].to_numpy(), e2["family"].tolist())
    axes[0].set_xlabel("Cohen's kappa on test blocks (higher is better)")
    axes[0].axvline(e2.loc[e2["input"] == "raw", "kappa"].iloc[0], color=INK, lw=1, ls="--")
    axes[0].set_xlim(e2["kappa"].min() - 0.05, e2["kappa"].max() + 0.04)
    sp = e2["salt_pepper_index"].to_numpy()
    _barh(axes[1], labels, sp, e2["family"].tolist(), annotate=False)
    axes[1].set_xlabel("salt-and-pepper index of the map (lower is better)")
    axes[1].axvline(sp[labels.index("raw")], color=INK, lw=1, ls="--")
    _family_legend(fig, e2["family"])
    fig.suptitle("E2 Land-cover classification, spatially blocked 70/30 split (dashed line = raw bands)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out)


def fig_e3(e3: pd.DataFrame, out: Path) -> Path:
    """E3 chart: boundary F1 scores and orientation coherence on the ridge window."""
    img = e3[e3["output"] == "image"]
    has_coh = "coherence_ridge" in e3
    fig, axes = plt.subplots(1, 3 if has_coh else 2, figsize=(17 if has_coh else 12, 8))
    for ax, col, title in [(axes[0], "boundary_f1", "F1, Canny edges vs all class boundaries"),
                           (axes[1], "builtup_boundary_f1", "F1, Canny edges vs built-up boundaries")]:
        _barh(ax, img["filter"].tolist(), img[col].to_numpy(), img["family"].tolist())
        ax.set_xlabel(f"{title}\n(2-pixel tolerance, higher is better)")
        ax.axvline(img.loc[img["filter"] == "original", col].iloc[0], color=INK, lw=1, ls="--")
    if has_coh:
        h = e3[e3["family"].str.startswith("HPF") | (e3["filter"] == "original")]
        _barh(axes[2], h["filter"].tolist(), h["coherence_ridge"].to_numpy(), h["family"].tolist())
        axes[2].set_xlabel("orientation coherence on the ridge window\n(1 = parallel linear features, "
                           "0 = speckle; high-pass outputs)")
        axes[2].axvline(h.loc[h["filter"] == "original", "coherence_ridge"].iloc[0], color=INK, lw=1, ls="--")
        axes[2].set_xlim(h["coherence_ridge"].min() - 0.05, h["coherence_ridge"].max() + 0.04)
    _family_legend(fig, e3["family"])
    fig.suptitle("E3 Edge and detail enhancement on the NIR band (dashed line = original image)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out)


def fig_e4(e4: pd.DataFrame, out: Path) -> Path:
    """E4 chart: spectral angle and NDVI change per filter (log scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.5), sharey=True)
    _barh(axes[0], e4["filter"].tolist(), e4["sam_deg"].to_numpy(), e4["family"].tolist(), annotate=False)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("mean spectral angle to the original (degrees, log scale, lower is better)")
    _barh(axes[1], e4["filter"].tolist(), e4["ndvi_abs_change"].to_numpy(), e4["family"].tolist(),
          annotate=False)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("mean |NDVI change| (log scale, lower is better)")
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
    _family_legend(fig, e4["family"])
    fig.suptitle("E4 Spectral fidelity: how far each filter moves the radiometry from the original")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out)


def fig_e5(e5: pd.DataFrame, out: Path) -> Path:
    """E5 chart: run time against kernel size, with the FFT range as a band."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    spatial = e5[e5["domain"] == "spatial"]
    colours = {"box mean": "#2a78d6", "gaussian": "#1baf7a", "median": "#6250d6"}
    for method, d in spatial.groupby("method"):
        ax.plot(d["kernel_side"], d["seconds"], marker="o", ms=6, lw=2, color=colours[method],
                label=f"spatial {method}")
    freq = e5[e5["domain"] == "frequency"]
    ax.axhspan(freq["seconds"].min(), freq["seconds"].max(), color="#eb6834", alpha=0.25,
               label=f"frequency domain, any mask / cutoff ({len(freq)} runs)")
    ax.set_xlabel("spatial kernel side (pixels)"), ax.set_ylabel("median wall-clock time (s)")
    ax.set_yscale("log")
    ax.grid(color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"E5 Run time on one {e5['image_shape'].iloc[0]} band: spatial cost grows with the "
                 "kernel, FFT cost does not")
    return _save(fig, out)


# ----------------------------------------------------------------- drivers
def make_all(cfg) -> list[Path]:
    """Regenerate every figure of section 7 from processed data and the CSV tables."""
    figs, tables = cfg["paths"]["figures"], cfg["paths"]["tables"]
    img, wc, meta = _load(cfg)
    sample, _ = read_stack(cfg["paths"]["sample"] / "reflectance.tif")
    paths = quicklooks(img, meta, figs)
    paths.append(fig_study_area(cfg, img, meta, figs / "fig01_study_area.png"))
    paths.append(fig_lpf_grid(cfg, img, figs / "fig02a_lpf_grid.png"))
    paths.append(fig_hpf_grid(cfg, img, figs / "fig02b_hpf_grid.png"))
    paths.append(fig_boost_grid(cfg, img, figs / "fig02c_highboost_grid.png"))
    paths.append(fig_spectra(sample[NIR], cfg, figs / "fig03_fft_spectra.png"))
    paths.append(fig_square_masks(sample[NIR], cfg, figs / "fig04_slide_square_masks.png"))
    paths.append(fig_ringing(cfg, img, figs / "fig05_ringing.png"))
    paths.append(fig_line_profile(cfg, img, figs / "fig06_line_profile.png"))
    paths.append(fig_class_maps(cfg, figs / "fig07_classification_maps.png"))
    for name, fn in [("e1_noise", fig_e1), ("e2_classification", fig_e2), ("e3_edges", fig_e3),
                     ("e4_fidelity", fig_e4), ("e5_runtime", fig_e5)]:
        paths.append(fn(pd.read_csv(tables / f"{name}.csv"), figs / f"fig08_{name}.png"))
    slides = figs / "slides"
    paths.append(fig_slide_grid(cfg, img, ["original", "box_5", "gauss_s2", "median_5", "fft_ideal_lp_0.2",
                                           "fft_gauss_lp_0.2"], slides / "slide_lpf.png", "false"))
    paths.append(fig_slide_grid(cfg, img, ["original", "hpf_7", "laplace_8", "sobel", "fft_gauss_hp_0.1"],
                                slides / "slide_hpf_edges.png", "edge"))
    paths.append(fig_slide_grid(cfg, img, ["original", "boost_n3_k0.5", "boost_n3_k1.0", "boost_n7_k1.5"],
                                slides / "slide_highboost.png", "false"))
    return paths


def make_demo(cfg, out_dir: Path) -> list[Path]:
    """Figures for the offline demo, drawn from the sample crop and the demo tables only."""
    figs, tables = out_dir / "figures", out_dir / "tables"
    sample, _ = read_stack(cfg["paths"]["sample"] / "reflectance.tif")
    paths = [fig_spectra(sample[NIR], cfg, figs / "demo_fft_spectra.png"),
             fig_square_masks(sample[NIR], cfg, figs / "demo_slide_square_masks.png")]
    for name, fn in [("e1_noise", fig_e1), ("e2_classification", fig_e2), ("e3_edges", fig_e3),
                     ("e4_fidelity", fig_e4), ("e5_runtime", fig_e5)]:
        paths.append(fn(pd.read_csv(tables / f"{name}.csv"), figs / f"demo_{name}.png"))
    return paths
