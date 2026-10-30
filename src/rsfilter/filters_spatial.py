"""Spatial-domain filters, written directly in NumPy.

Every filter accepts a single band ``(H, W)`` or a stack ``(B, H, W)`` of
float32 reflectance and returns an array of the same shape and dtype. Stacks
are processed band by band.

Border handling: the image is padded by mirror reflection without repeating the
edge pixel (NumPy ``mode="reflect"``: ``d c b | a b c d | c b a``). This equals
``scipy.ndimage`` ``mode="mirror"``, which the tests use as the reference.

Convolution
-----------
Discrete 2-D convolution of image f with kernel w of size (2a+1) x (2b+1)::

    g(x, y) = sum_{s=-a..a} sum_{t=-b..b} w(s, t) * f(x - s, y - t)

The minus signs mean the kernel is rotated by 180 degrees before it slides
over the image. For symmetric kernels (box, Gaussian, Laplacian) rotation
changes nothing; for Sobel and Prewitt it flips the sign of the response,
which disappears in the magnitude.

Implementation ("shift and accumulate"): pad once, then for every kernel cell
(i, j) add ``w_rot[i, j] * padded[i:i+H, j:j+W]`` to the output. This needs
only one extra image of memory and no Python loop over pixels.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _per_band(func, img: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Apply a single-band function to every band of a ``(B, H, W)`` stack."""
    if img.ndim == 2:
        return func(img, *args, **kwargs)
    return np.stack([func(band, *args, **kwargs) for band in img])


def pad_reflect(img: np.ndarray, ry: int, rx: int | None = None) -> np.ndarray:
    """Mirror-pad a 2-D array by ``ry`` rows and ``rx`` columns on each side."""
    rx = ry if rx is None else rx
    return np.pad(img, ((ry, ry), (rx, rx)), mode="reflect")


def _convolve_band(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ry, rx = kh // 2, kw // 2
    h, w = img.shape
    padded = pad_reflect(img.astype(np.float32), ry, rx)
    rot = kernel[::-1, ::-1].astype(np.float32)  # convolution rotates the kernel by 180 degrees
    out = np.zeros((h, w), np.float32)
    for i in range(kh):
        for j in range(kw):
            if rot[i, j] != 0:
                out += rot[i, j] * padded[i:i + h, j:j + w]
    return out


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an image (or each band of a stack) with an odd-sized 2-D kernel."""
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel sides must be odd")
    return _per_band(_convolve_band, img, kernel)


# ----------------------------------------------------------------- low-pass kernels
def box_kernel(n: int) -> np.ndarray:
    """n x n mean kernel, every weight 1 / n^2 (weights sum to 1, so mean brightness is kept)."""
    return np.full((n, n), 1.0 / (n * n), np.float32)


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Normalised 2-D Gaussian kernel.

    ``w(x, y) = exp(-(x^2 + y^2) / (2 sigma^2))`` divided by its sum.
    Side length ``2 * ceil(3 sigma) + 1`` keeps more than 99.7 % of the weight.
    """
    r = math.ceil(3 * sigma)
    x = np.arange(-r, r + 1, dtype=np.float64)
    g1 = np.exp(-x ** 2 / (2 * sigma ** 2))
    k = np.outer(g1, g1)
    return (k / k.sum()).astype(np.float32)


def mean_filter(img: np.ndarray, n: int) -> np.ndarray:
    """Box (mean) low-pass filter: each pixel becomes the average of its n x n neighbourhood."""
    return convolve2d(img, box_kernel(n))


def gaussian_filter(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass filter. Weights fall off smoothly, so it blurs with less ringing than the box."""
    return convolve2d(img, gaussian_kernel(sigma))


def _median_band(img: np.ndarray, n: int, chunk_rows: int) -> np.ndarray:
    r = n // 2
    padded = pad_reflect(img.astype(np.float32), r)
    out = np.empty(img.shape, np.float32)
    for y0 in range(0, img.shape[0], chunk_rows):
        y1 = min(y0 + chunk_rows, img.shape[0])
        windows = sliding_window_view(padded[y0:y1 + 2 * r], (n, n))  # (rows, W, n, n), a view
        out[y0:y1] = np.median(windows.reshape(*windows.shape[:2], n * n), axis=-1)
    return out


def median_filter(img: np.ndarray, n: int, chunk_rows: int = 128) -> np.ndarray:
    """Median filter (non-linear): each pixel becomes the median of its n x n neighbourhood.

    Isolated extreme values (salt-and-pepper noise) never reach the median, so
    they are removed while step edges survive. ``sliding_window_view`` builds the
    windows as a view; processing ``chunk_rows`` rows at a time bounds the
    memory taken by ``np.median``'s working copy.
    """
    return _per_band(_median_band, img, n, chunk_rows)


# ----------------------------------------------------------------- high-pass
LAPLACIAN_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
LAPLACIAN_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)


def high_pass(img: np.ndarray, n: int) -> np.ndarray:
    """Classic remote-sensing high-pass: ``HPF = original - mean_filter(original, n)``.

    The box filter keeps the low frequencies; subtracting it leaves the local
    deviation from the neighbourhood mean, which is large at edges and near zero
    in uniform areas (Lillesand, Kiefer and Chipman).
    """
    return img - mean_filter(img, n)


def laplacian(img: np.ndarray, neighbours: int = 4) -> np.ndarray:
    """Second-derivative (Laplacian) edge response with the 4- or 8-neighbour kernel.

    4-neighbour: ``[[0,1,0],[1,-4,1],[0,1,0]]``; 8-neighbour adds the diagonals.
    The response is zero on flat ground and on constant slopes, and changes sign
    across an edge.
    """
    return convolve2d(img, LAPLACIAN_4 if neighbours == 4 else LAPLACIAN_8)


def _gradient_magnitude(img: np.ndarray, kx: np.ndarray) -> np.ndarray:
    gx = convolve2d(img, kx)
    gy = convolve2d(img, kx.T)
    return np.sqrt(gx ** 2 + gy ** 2)


def sobel(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude ``sqrt(Gx^2 + Gy^2)``; centre row/column weighted 2."""
    return _gradient_magnitude(img, SOBEL_X)


def prewitt(img: np.ndarray) -> np.ndarray:
    """Prewitt gradient magnitude ``sqrt(Gx^2 + Gy^2)``; equal weights across the kernel."""
    return _gradient_magnitude(img, PREWITT_X)


def high_boost(img: np.ndarray, n: int, k: float) -> np.ndarray:
    """Edge enhancement: ``enhanced = original + k * HPF_n``.

    The original brightness is kept and the edge detail is added back k times,
    so the result still looks like the scene but with sharper boundaries.
    ``k = 1`` is the usual unsharp mask.
    """
    return img + k * high_pass(img, n)
