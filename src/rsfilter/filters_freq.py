"""Frequency-domain filters using the 2-D discrete Fourier transform.

Pipeline for one band (``freq_filter``)::

    f (H, W) --reflect pad--> fp --fft2--> F --fftshift--> Fc (zero frequency at centre)
    G = Fc * H(u, v) --ifftshift--> --ifft2--> real part --crop padding--> g (H, W)

Why pad: the DFT treats the image as one tile of an infinitely repeating
pattern, so the left edge "touches" the right edge. Without padding, a filter
mixes pixels from opposite borders (wrap-around). Mirror padding makes the
repeated pattern continuous at the borders and the padding is cropped off at
the end.

Distance in the frequency plane. For a padded size M x N the frequencies along
each axis are ``u = fftfreq(M)`` and ``v = fftfreq(N)`` in cycles per pixel,
ranging over [-0.5, 0.5). The Nyquist frequency is 0.5 cycles/pixel. We use::

    D(u, v) = sqrt(u^2 + v^2) / 0.5

so D = 1 at Nyquist along an axis, and a cutoff D0 = 0.2 means "20 % of
Nyquist". This keeps the cutoff meaning the same for any image size.

Transfer functions H(u, v) (Gonzalez and Woods, ch. 4)::

    Ideal LPF        H = 1 if D <= D0 else 0
    Butterworth LPF  H = 1 / (1 + (D / D0)^(2n))
    Gaussian LPF     H = exp(-D^2 / (2 D0^2))
    any HPF          H_hp = 1 - H_lp

The ideal filter's sharp cut makes its spatial kernel a sinc-like function
with oscillating side lobes, which shows up as ringing (Gibbs effect) beside
edges. Butterworth and Gaussian roll off smoothly and ring much less or not at
all.

A Gaussian LPF with cutoff D0 equals a spatial Gaussian of
``sigma = 1 / (pi * D0)`` pixels; the tests use that to check both domains agree.
"""
from __future__ import annotations

import numpy as np

from .filters_spatial import _per_band, pad_reflect


def distance_grid(shape: tuple[int, int]) -> np.ndarray:
    """Normalised radial frequency D (1.0 = Nyquist) for a centred (fftshifted) spectrum."""
    u = np.fft.fftshift(np.fft.fftfreq(shape[0]))
    v = np.fft.fftshift(np.fft.fftfreq(shape[1]))
    return np.sqrt(u[:, None] ** 2 + v[None, :] ** 2) / 0.5


def lowpass_mask(kind: str, shape: tuple[int, int], d0: float, order: int = 2) -> np.ndarray:
    """Low-pass transfer function H(u, v) of type 'ideal', 'butterworth' or 'gaussian'."""
    d = distance_grid(shape)
    if kind == "ideal":
        return (d <= d0).astype(np.float64)
    if kind == "butterworth":
        return 1.0 / (1.0 + (d / d0) ** (2 * order))
    if kind == "gaussian":
        return np.exp(-(d ** 2) / (2 * d0 ** 2))
    raise ValueError(f"unknown mask type {kind!r}")


def make_mask(kind: str, shape: tuple[int, int], d0: float, highpass: bool = False,
              order: int = 2) -> np.ndarray:
    """Return the LPF mask, or ``1 - LPF`` when ``highpass`` is True."""
    h = lowpass_mask(kind, shape, d0, order)
    return 1.0 - h if highpass else h


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multiply the centred spectrum of a 2-D array by ``mask`` and transform back (no padding)."""
    spectrum = np.fft.fftshift(np.fft.fft2(img))
    return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * mask)))


def _freq_band(img: np.ndarray, kind: str, d0: float, highpass: bool, order: int, pad: int) -> np.ndarray:
    padded = pad_reflect(img.astype(np.float64), pad)
    mask = make_mask(kind, padded.shape, d0, highpass, order)
    out = apply_mask(padded, mask)
    return out[pad:pad + img.shape[0], pad:pad + img.shape[1]].astype(np.float32)


def freq_filter(img: np.ndarray, kind: str, d0: float, highpass: bool = False,
                order: int = 2, pad: int = 64) -> np.ndarray:
    """Filter an image (or each band of a stack) in the frequency domain.

    ``kind``: 'ideal', 'butterworth' or 'gaussian'; ``d0``: cutoff as a fraction
    of Nyquist; ``highpass``: use ``1 - H``; ``order``: Butterworth order;
    ``pad``: mirror padding in pixels on each side.
    """
    return _per_band(_freq_band, img, kind, d0, highpass, order, pad)


def square_mask(shape: tuple[int, int], side: int) -> np.ndarray:
    """Square ideal low-pass mask: ones in a centred ``side`` x ``side`` block (as on the course slide)."""
    m = np.zeros(shape)
    cy, cx = shape[0] // 2, shape[1] // 2
    h = side // 2
    m[cy - h:cy + h + 1, cx - h:cx + h + 1] = 1.0
    return m


def square_lowpass(img: np.ndarray, side: int) -> np.ndarray:
    """Slide replication: keep a ``side`` x ``side`` block of the centred spectrum of a 2-D image.

    No padding is used here, so the mask is defined on the image's own
    frequency grid exactly as in the lecture example (512 x 512 image, masks
    61, 91 and 121 pixels wide).
    """
    return apply_mask(img.astype(np.float64), square_mask(img.shape, side)).astype(np.float32)


def log_spectrum(img: np.ndarray) -> np.ndarray:
    """Centred log-magnitude spectrum ``log(1 + |F(u, v)|)`` for display.

    The image mean is removed first so the zero-frequency spike does not
    swamp the colour scale.
    """
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))))


def step_response(kind: str, d0: float, order: int, pad: int, lo: float = 0.1, hi: float = 0.3) -> np.ndarray:
    """Filter a synthetic vertical step edge (lo -> hi) and return the middle row.

    A clean step isolates the filter's own behaviour from scene texture, so the
    overshoot it produces is the ringing of the filter itself.
    """
    img = np.full((256, 256), lo, np.float32)
    img[:, 128:] = hi
    return freq_filter(img, kind, d0, order=order, pad=pad)[128]


def overshoot_pct(profile: np.ndarray, lo: float = 0.1, hi: float = 0.3) -> float:
    """Largest excursion beyond the step levels, as a percentage of the step height."""
    return float(max(profile.max() - hi, lo - profile.min(), 0) / (hi - lo) * 100)
