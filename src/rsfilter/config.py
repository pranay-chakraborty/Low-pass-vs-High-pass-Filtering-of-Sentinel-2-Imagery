"""Configuration loading and project paths.

All tunable values (AOI, dates, filter parameters, seeds) live in ``config.yaml``
at the project root. This module finds that file and turns the relative paths in
it into absolute ``pathlib.Path`` objects, so the code runs the same way on
Windows and Linux.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml


def project_root() -> Path:
    """Return the project root directory.

    Search order: the ``RSFILTER_ROOT`` environment variable, then the current
    directory and its parents (first one holding ``config.yaml``), then the
    source tree this file lives in. A PyInstaller executable carries its own
    copy of ``config.yaml`` and ``data/sample`` in ``sys._MEIPASS``; that copy is
    used only when no project folder is found, so the executable also works on
    its own.
    """
    if os.environ.get("RSFILTER_ROOT"):
        return Path(os.environ["RSFILTER_ROOT"]).resolve()
    for folder in [Path.cwd(), *Path.cwd().parents]:
        if (folder / "config.yaml").exists():
            return folder
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "config.yaml").exists():
        return source_root
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    raise FileNotFoundError("config.yaml not found; run from the project folder or set RSFILTER_ROOT")


def load_config() -> dict:
    """Read ``config.yaml`` and add absolute paths under ``cfg['paths']``.

    Output paths are made relative to the current directory when the project
    root is the read-only bundle of an executable.
    """
    root = project_root()
    with open(root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["root"] = root
    out_root = Path.cwd() if hasattr(sys, "_MEIPASS") and root == Path(sys._MEIPASS) else root
    for key, rel in cfg["paths"].items():
        base = root if key == "sample" else out_root
        cfg["paths"][key] = base / Path(rel)
    return cfg
