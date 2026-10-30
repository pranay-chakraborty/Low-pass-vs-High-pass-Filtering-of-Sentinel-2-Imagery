"""Command-line interface: ``python -m rsfilter <command>`` (the executable exposes the same commands).

Commands
--------
fetch     search, rank and download the scene + WorldCover, build processed data,
          sample crop and quick-looks
filter    apply one filter to a GeoTIFF and write a GeoTIFF with the same georeferencing
evaluate  run experiments E1-E5 and the decision matrix
figures   regenerate every figure from the processed data and the CSV tables
demo      the whole pipeline on data/sample only (offline, about a minute)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from . import filters_freq as ff
from . import filters_spatial as fs
from .config import load_config

log = logging.getLogger("rsfilter")

# Base filters for `filter --filter NAME --params key=value ...`
FUNCTIONS = {
    "mean": (fs.mean_filter, {"n": int}),
    "gaussian": (fs.gaussian_filter, {"sigma": float}),
    "median": (fs.median_filter, {"n": int}),
    "highpass": (fs.high_pass, {"n": int}),
    "laplacian": (fs.laplacian, {"neighbours": int}),
    "sobel": (fs.sobel, {}),
    "prewitt": (fs.prewitt, {}),
    "highboost": (fs.high_boost, {"n": int, "k": float}),
    "fft": (ff.freq_filter, {"kind": str, "d0": float, "highpass": lambda v: v.lower() in ("1", "true", "yes"),
                             "order": int, "pad": int}),
}


def _parse_params(func_name: str, items: list[str]) -> dict:
    types = FUNCTIONS[func_name][1]
    params = {}
    for item in items:
        key, _, value = item.partition("=")
        if key not in types:
            raise SystemExit(f"{func_name} takes: {', '.join(types) or 'no parameters'}")
        params[key] = types[key](value)
    return params


def cmd_fetch(args, cfg):
    """Download the scene and WorldCover, build processed data, sample crop and quick-looks."""
    from . import data, preprocess, viz

    meta = data.fetch_scene(cfg, args.backend)
    info = preprocess.build_processed(cfg)
    sample = preprocess.make_sample(cfg)
    img, _ = preprocess.read_stack(cfg["paths"]["processed"] / "reflectance.tif")
    paths = viz.quicklooks(img, meta, cfg["paths"]["figures"])
    print(json.dumps({k: meta[k] for k in ["backend", "product_id", "acquisition_date",
                                            "aoi_cloud_metrics_pct", "aoi_aot_mean"]}, indent=2))
    print("no-data pixels filled:", info["nodata_pixels"], "| sample window:", sample)
    print("metadata:", cfg["paths"]["raw"] / "scene_metadata.json")
    for p in paths:
        print("quick-look:", p)


def cmd_filter(args, cfg):
    """Apply one filter to every band of a GeoTIFF and keep its georeferencing."""
    from . import bank, preprocess

    img, profile = preprocess.read_stack(Path(args.input))
    if args.filter in FUNCTIONS:
        fn, _ = FUNCTIONS[args.filter]
        out = fn(img, **_parse_params(args.filter, args.params))
    else:
        out = bank.get(cfg, args.filter).fn(img)
    preprocess.write_stack(Path(args.out), out.astype(np.float32), profile)
    print("wrote", args.out)


def _evaluate(cfg, input_path: Path, labels_path: Path, out_dir: Path, quick: bool):
    from . import data, experiments, preprocess

    img, profile = preprocess.read_stack(input_path)
    wc = preprocess.read_labels(labels_path)
    sample = cfg["paths"]["sample"] / "reflectance.tif"
    if sample.exists() and not quick:
        clean, _ = preprocess.read_stack(sample)
    else:  # centre crop of the input
        s = min(cfg["sample_crop_size"], *img.shape[1:])
        r0, c0 = (img.shape[1] - s) // 2, (img.shape[2] - s) // 2
        clean = img[:, r0:r0 + s, c0:c0 + s]
    full_grid = data.aoi_grid(cfg)
    insets = cfg["insets"] if img.shape[1:] == (full_grid["height"], full_grid["width"]) else None
    return experiments.run_all(img, wc, profile, cfg, out_dir, clean, insets, quick)


def cmd_evaluate(args, cfg):
    """Run E1 to E5 and the decision matrix on the given image and labels."""
    t0 = time.perf_counter()
    res = _evaluate(cfg, Path(args.input), Path(args.labels), Path(args.out), args.quick)
    print(res["decision_matrix"].to_string(index=False))
    print(f"done in {time.perf_counter() - t0:.0f} s; tables in {Path(args.out) / 'tables'}")


def cmd_figures(args, cfg):
    """Regenerate every figure from the processed data and the CSV tables."""
    from . import viz

    for p in viz.make_all(cfg):
        print(p)


def cmd_demo(args, cfg):
    """Offline demonstration on data/sample: reduced experiments plus figures."""
    from . import viz

    t0 = time.perf_counter()
    s, out = cfg["paths"]["sample"], cfg["paths"]["demo"]
    print(f"Demo on {s / 'reflectance.tif'} (512 x 512, 4 bands); outputs in {out}")
    res = _evaluate(cfg, s / "reflectance.tif", s / "worldcover.tif", out, quick=True)
    for p in viz.make_demo(cfg, out):
        print("figure:", p)
    print("\nDecision matrix (demo crop, reduced filter set for E2):")
    print(res["decision_matrix"][["application", "best_filter", "key_metric", "key_metric_value"]]
          .to_string(index=False))
    print(f"\nDemo finished in {time.perf_counter() - t0:.0f} s")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run one command."""
    p = argparse.ArgumentParser(prog="rsfilter", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true", help="show progress messages")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download scene and WorldCover for the AOI")
    f.add_argument("--backend", default="auto", choices=["auto", "gee", "pc", "earthsearch"])

    f = sub.add_parser("filter", help="apply one filter to a GeoTIFF")
    f.add_argument("--input", required=True)
    f.add_argument("--filter", required=True,
                   help=f"a base filter ({', '.join(FUNCTIONS)}) or a filter-bank name such as gauss_s2")
    f.add_argument("--params", nargs="*", default=[], help="key=value pairs, e.g. sigma=2")
    f.add_argument("--out", required=True)

    cfg_paths = load_config()["paths"]
    f = sub.add_parser("evaluate", help="run experiments E1-E5 and the decision matrix")
    f.add_argument("--input", default=str(cfg_paths["processed"] / "reflectance.tif"))
    f.add_argument("--labels", default=str(cfg_paths["processed"] / "worldcover.tif"))
    f.add_argument("--out", default=str(cfg_paths["tables"].parent))
    f.add_argument("--quick", action="store_true", help="reduced E2 and E5 (used by demo)")

    sub.add_parser("figures", help="regenerate all figures")
    sub.add_parser("demo", help="offline run on data/sample")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("rasterio").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_config()
    {"fetch": cmd_fetch, "filter": cmd_filter, "evaluate": cmd_evaluate,
     "figures": cmd_figures, "demo": cmd_demo}[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
