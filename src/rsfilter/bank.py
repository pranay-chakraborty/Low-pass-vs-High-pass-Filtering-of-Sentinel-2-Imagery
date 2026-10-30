"""The filter bank: every filter configuration used in the experiments, by name.

Each entry is a :class:`FilterSpec`. ``family`` groups the filters for tables
and figures; ``output`` says what the result is:

* ``"image"``  a reflectance-like image (low-pass and high-boost results),
* ``"edge"``   a zero-mean or non-negative edge layer (HPF, Laplacian, Sobel,
  Prewitt, frequency-domain HPF), which is not a reflectance image.

Names are short and stable (``gauss_s2``, ``boost_n5_k1.0``, ``fft_butter_lp_0.2``)
because they are used as row keys in every CSV.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

from . import filters_freq as ff
from . import filters_spatial as fs


@dataclass(frozen=True)
class FilterSpec:
    name: str        # key used in CSVs and the CLI
    label: str       # human-readable label for figures
    family: str      # LPF-spatial, LPF-freq, HPF-spatial, HPF-freq, HPF-enhanced
    output: str      # "image" or "edge"
    fn: Callable[[np.ndarray], np.ndarray]


FREQ_NAMES = {"ideal": "Ideal", "butterworth": "Butterworth", "gaussian": "Gaussian"}


def filter_bank(cfg: dict) -> list[FilterSpec]:
    """Build the full list of filter configurations from ``config.yaml``."""
    f = cfg["filters"]
    bank: list[FilterSpec] = []
    for n in f["box_sizes"]:
        bank.append(FilterSpec(f"box_{n}", f"Mean {n}x{n}", "LPF-spatial", "image",
                               partial(fs.mean_filter, n=n)))
    for s in f["gaussian_sigmas"]:
        bank.append(FilterSpec(f"gauss_s{s}", f"Gaussian sigma={s}", "LPF-spatial", "image",
                               partial(fs.gaussian_filter, sigma=s)))
    for n in f["median_sizes"]:
        bank.append(FilterSpec(f"median_{n}", f"Median {n}x{n}", "LPF-spatial", "image",
                               partial(fs.median_filter, n=n)))
    for n in f["hpf_sizes"]:
        bank.append(FilterSpec(f"hpf_{n}", f"HPF (orig - mean {n}x{n})", "HPF-spatial", "edge",
                               partial(fs.high_pass, n=n)))
    for nb in (4, 8):
        bank.append(FilterSpec(f"laplace_{nb}", f"Laplacian {nb}-neighbour", "HPF-spatial", "edge",
                               partial(fs.laplacian, neighbours=nb)))
    bank.append(FilterSpec("sobel", "Sobel magnitude", "HPF-spatial", "edge", fs.sobel))
    bank.append(FilterSpec("prewitt", "Prewitt magnitude", "HPF-spatial", "edge", fs.prewitt))
    for n in f["hpf_sizes"]:
        for k in f["boost_k"]:
            bank.append(FilterSpec(f"boost_n{n}_k{k}", f"High-boost n={n}, k={k}", "HPF-enhanced", "image",
                                   partial(fs.high_boost, n=n, k=k)))
    for kind, short in [("ideal", "ideal"), ("butterworth", "butter"), ("gaussian", "gauss")]:
        for hp in (False, True):
            for d0 in f["freq_cutoffs"]:
                bank.append(FilterSpec(
                    f"fft_{short}_{'hp' if hp else 'lp'}_{d0}",
                    f"{FREQ_NAMES[kind]} {'HPF' if hp else 'LPF'} D0={d0}",
                    "HPF-freq" if hp else "LPF-freq", "edge" if hp else "image",
                    partial(ff.freq_filter, kind=kind, d0=d0, highpass=hp,
                            order=f["butterworth_order"], pad=f["fft_pad"])))
    return bank


def get(cfg: dict, name: str) -> FilterSpec:
    """Look up one filter by name (raises ``KeyError`` with the valid names)."""
    bank = {s.name: s for s in filter_bank(cfg)}
    if name not in bank:
        raise KeyError(f"unknown filter {name!r}; choose from: {', '.join(bank)}")
    return bank[name]
