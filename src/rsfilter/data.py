"""Scene search, selection and download for the study area.

Three backends are supported, tried in this order by :func:`fetch_scene`:

1. ``gee``          Google Earth Engine, collection ``COPERNICUS/S2_SR_HARMONIZED``.
2. ``pc``           Microsoft Planetary Computer STAC, collection ``sentinel-2-l2a``.
3. ``earthsearch``  Element84 Earth Search STAC on AWS, collection ``sentinel-2-l2a``.

Scene selection works on the AOI, not the whole 110 x 110 km tile. For every
candidate scene we measure, inside the AOI only:

* ``nodata_pct``  pixels with no data (outside the swath, or SCL class 0),
* ``shadow_pct``  SCL class 3 (cloud shadow),
* ``cloud_pct``   SCL classes 8 and 9 (cloud, medium and high probability),
* ``cirrus_pct``  SCL class 10 (thin cirrus),
* ``prob_pct``    s2cloudless probability above the threshold (GEE only),
* ``bad_pct``     union of all of the above,
* ``aot_mean``    mean aerosol optical thickness at 550 nm (haze indicator).

Scenes whose ``bad_pct`` lies within ``aoi_bad_tolerance`` percentage points of
the cleanest scene are treated as equally clean, and among those the scene with
the lowest AOT is chosen. Winter over Delhi NCR is the season of smog, so a
scene can be free of cloud yet still hazy; AOT separates those cases.

Everything is written on one fixed grid: EPSG:32643 (UTM 43N), 10 m pixels,
covering the AOI bounding box snapped outwards to whole pixels.
"""
from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

log = logging.getLogger(__name__)

SCL_NODATA, SCL_SHADOW, SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS = 0, 3, 8, 9, 10


# --------------------------------------------------------------------------- grid
def aoi_grid(cfg: dict) -> dict:
    """Return the output grid for the AOI: CRS, affine transform, width, height.

    The WGS84 bounding box is projected to UTM and snapped outwards to multiples
    of the pixel size. Sentinel-2 tiles use the same 10 m alignment, so pixels in
    our grid coincide with native Sentinel-2 pixels and no resampling occurs.
    """
    res = cfg["aoi"]["resolution"]
    left, bottom, right, top = transform_bounds("EPSG:4326", cfg["aoi"]["crs"], *cfg["aoi"]["bbox"])
    left, bottom = np.floor(left / res) * res, np.floor(bottom / res) * res
    right, top = np.ceil(right / res) * res, np.ceil(top / res) * res
    return {
        "crs": cfg["aoi"]["crs"],
        "transform": from_origin(left, top, res, res),
        "width": int(round((right - left) / res)),
        "height": int(round((top - bottom) / res)),
        "bounds": (left, bottom, right, top),
    }


# --------------------------------------------------------------------------- ranking
def rank_candidates(df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Order candidate scenes from best to worst.

    Scenes with ``bad_pct <= min(bad_pct) + tolerance`` form the "clean" group
    and are ordered by AOT (least hazy first); the remaining scenes follow,
    ordered by ``bad_pct``. Adds the columns ``clean_group`` and ``rank``.
    """
    df = df.copy()
    df["clean_group"] = df["bad_pct"] <= df["bad_pct"].min() + tolerance
    clean = df[df["clean_group"]].sort_values(["aot_mean", "bad_pct"])
    rest = df[~df["clean_group"]].sort_values(["bad_pct", "aot_mean"])
    df = pd.concat([clean, rest]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def selection_rationale(ranked: pd.DataFrame, tolerance: float, window: str) -> str:
    """Plain-language explanation of why the top-ranked scene was chosen."""
    best = ranked.iloc[0]
    n_clean = int(ranked["clean_group"].sum())
    text = (
        f"{len(ranked)} Sentinel-2 L2A scenes covering the AOI were found for {window} "
        f"with scene-wide cloud cover below the limit. Each was scored inside the AOI only. "
        f"The cleanest scene had {ranked['bad_pct'].min():.2f} % of AOI pixels flagged as cloud, "
        f"cirrus, cloud shadow or no data; {n_clean} scene(s) were within {tolerance} percentage points "
        f"of that and were treated as equally clean. Among those, the scene with the lowest mean "
        f"aerosol optical thickness (AOT {best['aot_mean']:.3f}) was chosen to limit winter haze. "
        f"Selected: {best['date']} with {best['bad_pct']:.2f} % flagged AOI pixels "
        f"(SCL cloud {best['cloud_pct']:.2f} %, cirrus {best['cirrus_pct']:.2f} %, "
        f"shadow {best['shadow_pct']:.2f} %, no data {best['nodata_pct']:.2f} %)."
    )
    if "prob_pct" in ranked and pd.notna(best.get("prob_pct")):
        text += f" s2cloudless flagged {best['prob_pct']:.2f} % of AOI pixels."
    return text


# --------------------------------------------------------------------------- GEE
def _ee_init():
    """Initialise Earth Engine with the project ID from ``.env`` (``EE_PROJECT``)."""
    import ee
    from dotenv import load_dotenv

    load_dotenv()
    project = os.environ.get("EE_PROJECT")
    if not project:
        raise RuntimeError("EE_PROJECT is not set in .env")
    try:
        ee.Initialize(project=project)
    except Exception as exc:  # credentials missing or expired
        raise RuntimeError("Earth Engine is not authenticated. Run: earthengine authenticate") from exc
    return ee


def gee_candidates(cfg: dict, start: str, end: str) -> pd.DataFrame:
    """Score every GEE Sentinel-2 L2A scene whose footprint contains the AOI.

    The SCL band and the matching s2cloudless probability image are reduced
    over the AOI at 20 m (the SCL native resolution). Returns one row per scene.
    """
    ee = _ee_init()
    aoi = ee.Geometry.Rectangle(cfg["aoi"]["bbox"])
    sc = cfg["scene"]
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", sc["max_scene_cloud"]))
          .filter(ee.Filter.contains(leftField=".geo", rightValue=aoi)))
    prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(aoi).filterDate(start, end)
    joined = ee.Join.saveFirst("cloud_prob").apply(
        s2, prob, ee.Filter.equals(leftField="system:index", rightField="system:index"))

    def score(img):
        img = ee.Image(img)
        scl = img.select("SCL").unmask(0)
        nodata = scl.eq(SCL_NODATA).Or(img.select("B4").unmask(0).eq(0))
        shadow = scl.eq(SCL_SHADOW)
        cloud = scl.eq(SCL_CLOUD_MED).Or(scl.eq(SCL_CLOUD_HIGH))
        cirrus = scl.eq(SCL_CIRRUS)
        cp = ee.Image(img.get("cloud_prob")).select("probability").unmask(0).gt(sc["cloud_prob_threshold"])
        bad = nodata.Or(shadow).Or(cloud).Or(cirrus).Or(cp)
        stack = ee.Image.cat([nodata, shadow, cloud, cirrus, cp, bad,
                              img.select("AOT").multiply(0.001)]).rename(
            ["nodata", "shadow", "cloud", "cirrus", "prob", "bad", "aot"])
        stats = stack.reduceRegion(ee.Reducer.mean(), aoi, scale=20, crs=cfg["aoi"]["crs"], maxPixels=1e9)
        return ee.Feature(None, stats).set({
            "image_id": img.get("system:id"),
            "product_id": img.get("PRODUCT_ID"),
            "tile": img.get("MGRS_TILE"),
            "time": img.get("system:time_start"),
            "scene_cloud_pct": img.get("CLOUDY_PIXEL_PERCENTAGE"),
            "baseline": img.get("PROCESSING_BASELINE"),
        })

    feats = ee.FeatureCollection(ee.List(joined.toList(500)).map(score)).getInfo()["features"]
    rows = []
    for f in feats:
        p = f["properties"]
        rows.append({
            "date": datetime.fromtimestamp(p["time"] / 1000, timezone.utc).strftime("%Y-%m-%d"),
            "datetime_utc": datetime.fromtimestamp(p["time"] / 1000, timezone.utc).isoformat(),
            "image_id": p["image_id"], "product_id": p["product_id"], "tile": p["tile"],
            "scene_cloud_pct": p["scene_cloud_pct"], "processing_baseline": p["baseline"],
            "nodata_pct": 100 * p["nodata"], "shadow_pct": 100 * p["shadow"],
            "cloud_pct": 100 * p["cloud"], "cirrus_pct": 100 * p["cirrus"],
            "prob_pct": 100 * p["prob"], "bad_pct": 100 * p["bad"], "aot_mean": p["aot"],
        })
    return pd.DataFrame(rows)


def gee_thumbnail(cfg: dict, image_id: str, bands: list[str], size: int = 420) -> np.ndarray:
    """Download a small RGB PNG of the AOI for visual checking of a candidate scene."""
    import requests
    from PIL import Image

    ee = _ee_init()
    aoi = ee.Geometry.Rectangle(cfg["aoi"]["bbox"])
    url = ee.Image(image_id).select(bands).getThumbURL(
        {"region": aoi, "dimensions": size, "min": 0, "max": 3000, "format": "png", "crs": cfg["aoi"]["crs"]})
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))


def _gee_download_grid(image, grid: dict, out: Path, tiles: int = 2) -> Path:
    """Download an ``ee.Image`` on ``grid`` as GeoTIFF, in ``tiles`` x ``tiles`` pieces.

    ``getDownloadURL`` refuses requests above 32 MB, so the grid is split into
    tiles that each stay below that, then mosaicked with ``rasterio.merge``.
    """
    import requests

    t, w, h = grid["transform"], grid["width"], grid["height"]
    xs = np.linspace(0, w, tiles + 1).astype(int)
    ys = np.linspace(0, h, tiles + 1).astype(int)
    parts = []
    for i in range(tiles):
        for j in range(tiles):
            x0, y0 = t * (xs[j], ys[i])
            url = image.getDownloadURL({
                "crs": grid["crs"], "crs_transform": [t.a, 0, x0, 0, t.e, y0],
                "dimensions": [int(xs[j + 1] - xs[j]), int(ys[i + 1] - ys[i])], "format": "GEO_TIFF"})
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            part = out.with_suffix(f".part{i}{j}.tif")
            part.write_bytes(r.content)
            parts.append(part)
    srcs = [rasterio.open(p) for p in parts]
    mosaic, transform = merge(srcs)
    profile = srcs[0].profile | {"width": mosaic.shape[2], "height": mosaic.shape[1],
                                 "transform": transform, "compress": "deflate"}
    for s in srcs:
        s.close()
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(mosaic)
    for p in parts:
        p.unlink()
    return out


def download_gee(cfg: dict, row: pd.Series, raw_dir: Path) -> dict:
    """Download the selected scene (B2, B3, B4, B8, SCL) and WorldCover via GEE.

    WorldCover is downloaded in its native EPSG:4326 grid and reprojected
    locally in :func:`rsfilter.preprocess.warp_to_grid`, the same as for STAC.
    """
    import requests
    from .preprocess import warp_to_grid

    ee = _ee_init()
    grid = aoi_grid(cfg)
    img = ee.Image(row["image_id"]).select(cfg["bands"] + ["SCL"]).toUint16()
    scene_path = _gee_download_grid(img, grid, raw_dir / "s2_aoi_dn.tif")

    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    w, s, e, n = cfg["aoi"]["bbox"]
    region = ee.Geometry.Rectangle([w - 0.01, s - 0.01, e + 0.01, n + 0.01])
    url = wc.getDownloadURL({"region": region, "scale": 10, "crs": "EPSG:4326", "format": "GEO_TIFF"})
    native = raw_dir / "worldcover_native.tif"
    native.write_bytes(requests.get(url, timeout=600).content)
    wc_path = warp_to_grid(native, grid, raw_dir / "worldcover_aoi.tif")
    native.unlink()
    return {"scene": scene_path, "worldcover": wc_path, "offset": 0,
            "worldcover_source": "ESA/WorldCover/v200 (GEE)"}


# --------------------------------------------------------------------------- STAC
STAC = {
    "pc": {"url": "https://planetarycomputer.microsoft.com/api/stac/v1",
           "assets": {"B2": "B02", "B3": "B03", "B4": "B04", "B8": "B08", "SCL": "SCL", "AOT": "AOT"}},
    "earthsearch": {"url": "https://earth-search.aws.element84.com/v1",
                    "assets": {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir", "SCL": "scl", "AOT": "aot"}},
}


def _stac_items(backend: str, cfg: dict, start: str, end: str):
    """Search a STAC catalogue for L2A items over the AOI below the cloud limit."""
    from pystac_client import Client

    kwargs = {}
    if backend == "pc":
        import planetary_computer
        kwargs["modifier"] = planetary_computer.sign_inplace
    client = Client.open(STAC[backend]["url"], **kwargs)
    search = client.search(collections=["sentinel-2-l2a"], bbox=cfg["aoi"]["bbox"],
                           datetime=f"{start}/{end}",
                           query={"eo:cloud_cover": {"lt": cfg["scene"]["max_scene_cloud"]}})
    return list(search.items())


def _read_on_grid(href: str, grid: dict, resampling="nearest") -> np.ndarray:
    """Read one COG asset onto the output grid, fetching only the needed window."""
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    with rasterio.open(href) as src, WarpedVRT(
            src, crs=grid["crs"], transform=grid["transform"], width=grid["width"],
            height=grid["height"], resampling=Resampling[resampling], nodata=0) as vrt:
        return vrt.read(1)


def stac_candidates(backend: str, cfg: dict, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Score STAC items over the AOI using their SCL and AOT assets (no s2cloudless)."""
    grid = aoi_grid(cfg)
    a = STAC[backend]["assets"]
    rows, items = [], {}
    for it in _stac_items(backend, cfg, start, end):
        scl = _read_on_grid(it.assets[a["SCL"]].href, grid)
        aot = _read_on_grid(it.assets[a["AOT"]].href, grid).astype(float)
        nodata = scl == SCL_NODATA
        flags = {"nodata": nodata, "shadow": scl == SCL_SHADOW,
                 "cloud": np.isin(scl, [SCL_CLOUD_MED, SCL_CLOUD_HIGH]), "cirrus": scl == SCL_CIRRUS}
        bad = np.logical_or.reduce(list(flags.values()))
        rows.append({
            "date": it.datetime.strftime("%Y-%m-%d"), "datetime_utc": it.datetime.isoformat(),
            "image_id": it.id, "product_id": it.properties.get("s2:product_uri", it.id),
            "tile": it.properties.get("s2:mgrs_tile", it.properties.get("grid:code")),
            "scene_cloud_pct": it.properties.get("eo:cloud_cover"),
            "processing_baseline": it.properties.get("s2:processing_baseline"),
            **{f"{k}_pct": 100 * v.mean() for k, v in flags.items()},
            "prob_pct": np.nan, "bad_pct": 100 * bad.mean(),
            "aot_mean": float(aot[~nodata].mean() / 1000) if (~nodata).any() else np.nan,
        })
        items[it.id] = it
    return pd.DataFrame(rows), items


def download_stac(backend: str, cfg: dict, item, raw_dir: Path) -> dict:
    """Write the selected STAC scene and WorldCover onto the output grid.

    The BOA offset is decided from the item's ``s2:processing_baseline``:
    baseline 04.00 or later stores DN with +1000 added, so reflectance is
    ``(DN - 1000) / 10000``. Earth Search 'v1' items may already be harmonised;
    their ``earthsearch:boa_offset_applied`` flag is checked when present.
    """
    grid = aoi_grid(cfg)
    a = STAC[backend]["assets"]
    stack = np.stack([_read_on_grid(item.assets[a[b]].href, grid) for b in cfg["bands"] + ["SCL"]])
    baseline = float(item.properties.get("s2:processing_baseline", "0") or 0)
    offset = 1000 if baseline >= 4.0 else 0
    if item.properties.get("earthsearch:boa_offset_applied"):
        offset = 0
    scene_path = raw_dir / "s2_aoi_dn.tif"
    profile = {"driver": "GTiff", "dtype": "uint16", "count": stack.shape[0], "width": grid["width"],
               "height": grid["height"], "crs": grid["crs"], "transform": grid["transform"],
               "compress": "deflate"}
    with rasterio.open(scene_path, "w", **profile) as dst:
        dst.write(stack.astype("uint16"))
    wc_path = download_worldcover_pc(cfg, raw_dir)
    return {"scene": scene_path, "worldcover": wc_path, "offset": offset,
            "worldcover_source": "esa-worldcover v200 (Planetary Computer)"}


def download_worldcover_pc(cfg: dict, raw_dir: Path) -> Path:
    """Read ESA WorldCover 2021 (v200) from Planetary Computer onto the output grid."""
    import planetary_computer
    from pystac_client import Client

    grid = aoi_grid(cfg)
    client = Client.open(STAC["pc"]["url"], modifier=planetary_computer.sign_inplace)
    items = [i for i in client.search(collections=["esa-worldcover"], bbox=cfg["aoi"]["bbox"]).items()
             if i.properties.get("esa_worldcover:product_version", "").startswith("2")]
    wc = np.zeros((grid["height"], grid["width"]), np.uint8)
    for it in items:  # the AOI may straddle two 3 x 3 degree WorldCover tiles
        part = _read_on_grid(it.assets["map"].href, grid)
        wc = np.where(wc == 0, part, wc)
    out = raw_dir / "worldcover_aoi.tif"
    with rasterio.open(out, "w", driver="GTiff", dtype="uint8", count=1, width=grid["width"],
                       height=grid["height"], crs=grid["crs"], transform=grid["transform"],
                       compress="deflate") as dst:
        dst.write(wc, 1)
    return out


def save_candidate_sheet(cfg: dict, ranked: pd.DataFrame) -> Path:
    """Save thumbnails of the top-ranked GEE candidates side by side for visual checking."""
    from .viz import candidate_sheet

    top = ranked.drop_duplicates("date").head(cfg["scene"]["n_quicklook_candidates"])
    thumbs = [gee_thumbnail(cfg, r["image_id"], ["B4", "B3", "B2"]) for _, r in top.iterrows()]
    return candidate_sheet(thumbs, top.to_dict("records"), cfg["paths"]["figures"] / "scene_candidates.png")


# --------------------------------------------------------------------------- orchestration
def _search(backend: str, cfg: dict, start: str, end: str):
    if backend == "gee":
        return gee_candidates(cfg, start, end), None
    return stac_candidates(backend, cfg, start, end)


def fetch_scene(cfg: dict, backend: str = "auto") -> dict:
    """Search, rank, download and document one Sentinel-2 L2A scene for the AOI.

    With ``backend="auto"`` the backends are tried in the order gee, pc,
    earthsearch; the first that returns a usable scene wins and is logged. The
    primary date window is tried first, then the wider fallback window. Writes
    ``data/raw/s2_aoi_dn.tif``, ``data/raw/worldcover_aoi.tif``,
    ``data/raw/scene_metadata.json`` and ``results/tables/scene_candidates.csv``.
    """
    raw_dir = cfg["paths"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["tables"].mkdir(parents=True, exist_ok=True)
    sc = cfg["scene"]
    windows = [(sc["date_start"], sc["date_end"]), (sc["fallback_start"], sc["fallback_end"])]
    order = ["gee", "pc", "earthsearch"] if backend == "auto" else [backend]
    errors = {}
    for be in order:
        for start, end in windows:
            try:
                df, items = _search(be, cfg, start, end)
            except Exception as exc:
                log.warning("backend %s failed: %s", be, exc)
                errors[be] = str(exc)
                break
            if df.empty:
                log.warning("backend %s: no scene in %s to %s", be, start, end)
                continue
            ranked = rank_candidates(df, sc["aoi_bad_tolerance"])
            window = f"{start} to {end} (end exclusive)"
            ranked.to_csv(cfg["paths"]["tables"] / "scene_candidates.csv", index=False, float_format="%.4f")
            best = ranked.iloc[0]
            log.info("backend %s: selected %s (%s)", be, best["image_id"], best["date"])
            if be == "gee":
                save_candidate_sheet(cfg, ranked)
                files = download_gee(cfg, best, raw_dir)
            else:
                files = download_stac(be, cfg, items[best["image_id"]], raw_dir)
            grid = aoi_grid(cfg)
            meta = {
                "backend": be,
                "collection": "COPERNICUS/S2_SR_HARMONIZED" if be == "gee" else "sentinel-2-l2a",
                "image_id": best["image_id"],
                "product_id": best["product_id"],
                "mgrs_tile": best["tile"],
                "acquisition_date": best["date"],
                "acquisition_datetime_utc": best["datetime_utc"],
                "processing_baseline": best["processing_baseline"],
                "scene_cloud_pct": round(float(best["scene_cloud_pct"]), 4),
                "aoi_cloud_metrics_pct": {k: round(float(best[k]), 4) for k in
                                          ["cloud_pct", "cirrus_pct", "shadow_pct", "nodata_pct",
                                           "prob_pct", "bad_pct"] if pd.notna(best[k])},
                "aoi_aot_mean": round(float(best["aot_mean"]), 4),
                "search_window": window,
                "n_candidates": int(len(ranked)),
                "selection_rationale": selection_rationale(ranked, sc["aoi_bad_tolerance"], window),
                "dn_offset": files["offset"],
                "reflectance_formula": f"(DN - {files['offset']}) / 10000, clipped to [0, 1]; DN 0 in all bands = no data",
                "bands": dict(zip(cfg["bands"], cfg["band_names"])),
                "crs": grid["crs"],
                "pixel_size_m": cfg["aoi"]["resolution"],
                "width": grid["width"], "height": grid["height"],
                "bounds_utm": [float(v) for v in grid["bounds"]],
                "aoi_bbox_wgs84": cfg["aoi"]["bbox"],
                "worldcover": files["worldcover_source"],
                "download_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "backend_errors": errors,
            }
            (raw_dir / "scene_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return meta
    raise RuntimeError(f"No backend produced a scene. Errors: {errors}")
