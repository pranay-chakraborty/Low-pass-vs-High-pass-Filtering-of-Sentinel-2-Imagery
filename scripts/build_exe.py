"""Build a one-file executable of the CLI with PyInstaller: python scripts/build_exe.py

Output: dist/rsfilter (Linux/macOS) or dist/rsfilter.exe (Windows). PyInstaller
builds only for the operating system it runs on. The executable carries
config.yaml and data/sample, so `rsfilter demo` works from any folder.
"""
import os
import sys
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parents[1]
SEP = os.pathsep  # ';' on Windows, ':' elsewhere, as --add-data expects

if __name__ == "__main__":
    entry = ROOT / "build" / "rsfilter_entry.py"
    entry.parent.mkdir(exist_ok=True)
    entry.write_text("from rsfilter.cli import main\nmain()\n", encoding="utf-8")
    PyInstaller.__main__.run([
        str(entry), "--onefile", "--name", "rsfilter", "--noconfirm", "--clean",
        "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build"), "--specpath", str(ROOT / "build"),
        "--paths", str(ROOT / "src"),
        "--add-data", f"{ROOT / 'config.yaml'}{SEP}.",
        "--add-data", f"{ROOT / 'data' / 'sample'}{SEP}data/sample",
        "--collect-all", "rasterio", "--collect-submodules", "skimage", "--collect-submodules", "sklearn",
        "--exclude-module", "tkinter", "--exclude-module", "pytest",
    ])
    print("built", ROOT / "dist" / ("rsfilter.exe" if sys.platform == "win32" else "rsfilter"))
