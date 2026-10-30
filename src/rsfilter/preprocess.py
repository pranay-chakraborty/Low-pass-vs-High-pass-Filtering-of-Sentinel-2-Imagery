"""Pre-processing: reflectance scaling, WorldCover regridding, sample crop, display stretch.

Data structures used throughout the project:

* scene array ``img``: ``numpy.ndarray`` of shape ``(4, H, W)``, dtype ``float32``,
  surface reflectance in [0, 1], band order blue, green, red, NIR.
* label array ``wc``: shape ``(H, W)``, dtype ``uint8``, ESA WorldCover codes.
* ``profile``: the rasterio profile dict (CRS, affine transform, size) of the grid.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

log = logging.getLogger(__name__)


def warp_to_grid(src_path: Path, grid: dict, out_path: Path) -> Path:
    """Reproject a categorical raster onto ``grid`` with nearest-neighbour resampling.

    Nearest neighbour is required for class labels: any interpolating method
    would invent codes that do not exist (e.g. the average of 10 and 50).
    """
    with rasterio.open(src_path) as src:
        dst = np.zeros((grid["height"], grid["width"]), src.dtypes[0])
        reproject(rasterio.band(src, 1), dst, dst_transform=grid["transform"], dst_crs=grid["crs"],
                  resampling=Resampling.nearest)
        profile = {"driver": "GTiff", "dtype": src.dtypes[0], "count": 1, "width": grid["width"],
                   "height": grid["height"], "crs": grid["crs"], "transform": grid["transform"],
                   "compress": "deflate"}
    with rasterio.open(out_path, "w", **profile) as out:
        out.write(dst, 1)
    return out_path


def to_reflectance(dn: np.ndarray, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Convert L2A digital numbers to surface reflectance.

    ``rho = (DN - offset) / 10000``, clipped to [0, 1]. No data is DN 0 in every
    band. A single band at 0 is not treated as no data: the harmonised GEE
    collection clamps very dark values to 0 while the other bands stay valid.
    Returns ``(reflectance float32, nodata mask bool)``.
    """
    nodata = (dn == 0).all(axis=0)
    rho = np.clip((dn.astype(np.float32) - offset) / 10000.0, 0.0, 1.0)
    return rho, nodata


def build_processed(cfg: dict) -> dict:
    """Turn ``data/raw`` downloads into analysis-ready files in ``data/processed``.

    Writes ``reflectance.tif`` (4 x float32), ``scl.tif`` and ``worldcover.tif``.
    No-data pixels (none are expected for the selected scene) are filled with
    the band median so that convolution never sees NaN; their count is logged
    and returned.
    """
    raw, proc = cfg["paths"]["raw"], cfg["paths"]["processed"]
    proc.mkdir(parents=True, exist_ok=True)
    meta = json.loads((raw / "scene_metadata.json").read_text(encoding="utf-8"))
    with rasterio.open(raw / "s2_aoi_dn.tif") as src:
        dn = src.read()
        profile = src.profile
    rho, nodata = to_reflectance(dn[:4], meta["dn_offset"])
    n_nodata = int(nodata.sum())
    if n_nodata:
        log.warning("%d no-data pixels filled with band medians", n_nodata)
        for b in range(rho.shape[0]):
            rho[b][nodata] = np.median(rho[b][~nodata])
    profile.update(count=4, dtype="float32", compress="deflate", predictor=3)
    with rasterio.open(proc / "reflectance.tif", "w", **profile) as dst:
        dst.write(rho)
        dst.descriptions = tuple(cfg["band_names"])
    profile.update(count=1, dtype="uint8", predictor=1)
    with rasterio.open(proc / "scl.tif", "w", **profile) as dst:
        dst.write(dn[4].astype("uint8"), 1)
    shutil.copy(raw / "worldcover_aoi.tif", proc / "worldcover.tif")
    return {"nodata_pixels": n_nodata}


def read_stack(path: Path) -> tuple[np.ndarray, dict]:
    """Read a multi-band GeoTIFF as float32 ``(bands, H, W)`` plus its profile."""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32), src.profile


def read_labels(path: Path) -> np.ndarray:
    """Read a single-band label raster (WorldCover codes) as ``uint8 (H, W)``."""
    with rasterio.open(path) as src:
        return src.read(1)


def write_stack(path: Path, arr: np.ndarray, profile: dict) -> None:
    """Write ``(bands, H, W)`` or ``(H, W)`` array with the georeferencing in ``profile``."""
    arr = arr[None] if arr.ndim == 2 else arr
    prof = dict(profile, count=arr.shape[0], dtype=str(arr.dtype), compress="deflate")
    for key in ("predictor", "blockxsize", "blockysize", "tiled", "photometric"):
        prof.pop(key, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr)


def class_entropy(labels: np.ndarray) -> float:
    """Shannon entropy (bits) of the class frequencies in ``labels``."""
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def choose_sample_window(wc: np.ndarray, size: int, step: int = 64) -> tuple[int, int]:
    """Pick the ``size`` x ``size`` window with the most varied land cover.

    Windows are scanned with stride ``step``; the one with the highest class
    entropy wins, so the demo crop contains ridge, built-up, cropland and
    water rather than a single cover type. Returns the (row, col) of its corner.
    """
    best, best_rc = -1.0, (0, 0)
    for r in range(0, wc.shape[0] - size + 1, step):
        for c in range(0, wc.shape[1] - size + 1, step):
            e = class_entropy(wc[r:r + size, c:c + size])
            if e > best:
                best, best_rc = e, (r, c)
    return best_rc


def make_sample(cfg: dict) -> dict:
    """Export the 512 x 512 demo crop (reflectance + WorldCover + metadata) to ``data/sample``."""
    proc, sample = cfg["paths"]["processed"], cfg["paths"]["sample"]
    sample.mkdir(parents=True, exist_ok=True)
    size = cfg["sample_crop_size"]
    wc = read_labels(proc / "worldcover.tif")
    r, c = choose_sample_window(wc, size)
    window = rasterio.windows.Window(c, r, size, size)
    for name in ["reflectance.tif", "worldcover.tif"]:
        with rasterio.open(proc / name) as src:
            arr = src.read(window=window)
            profile = dict(src.profile, width=size, height=size,
                           transform=rasterio.windows.transform(window, src.transform))
        write_stack(sample / name, arr, profile)
    meta = json.loads((cfg["paths"]["raw"] / "scene_metadata.json").read_text(encoding="utf-8"))
    meta["sample_window"] = {"row_off": int(r), "col_off": int(c), "size": size,
                             "class_entropy_bits": round(class_entropy(wc[r:r + size, c:c + size]), 3),
                             "selection": "512 x 512 window with the highest WorldCover class entropy"}
    (sample / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta["sample_window"]


def stretch(img: np.ndarray, lo: float = 2, hi: float = 98, ref: np.ndarray | None = None) -> np.ndarray:
    """Percentile stretch to [0, 1] for display only (never used before filtering or metrics).

    Works on ``(H, W)`` or ``(H, W, 3)``. If ``ref`` is given its percentiles are
    used, so several panels can share one stretch and stay comparable.
    """
    ref = img if ref is None else ref
    if img.ndim == 3:
        out = np.empty_like(img, dtype=np.float32)
        for i in range(img.shape[2]):
            a, b = np.percentile(ref[..., i], [lo, hi])
            out[..., i] = np.clip((img[..., i] - a) / (b - a + 1e-12), 0, 1)
        return out
    a, b = np.percentile(ref, [lo, hi])
    return np.clip((img - a) / (b - a + 1e-12), 0, 1)
