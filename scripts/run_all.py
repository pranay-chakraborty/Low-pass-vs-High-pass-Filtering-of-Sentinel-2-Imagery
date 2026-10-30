"""Reproduce every result with one command: python scripts/run_all.py

Steps: fetch (skipped if data/processed already exists, unless --refetch),
tests, experiments E1-E5, figures, report and slides (LaTeX).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print("\n>>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refetch", action="store_true", help="download the scene again")
    args = p.parse_args()
    if args.refetch or not (ROOT / "data" / "processed" / "reflectance.tif").exists():
        run("-m", "rsfilter", "-v", "fetch")
    run("-m", "pytest", "-q")
    run("-m", "rsfilter", "-v", "evaluate")
    run("-m", "rsfilter", "figures")
    run("scripts/build_latex.py")
