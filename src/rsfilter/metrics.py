"""Quality metrics used by experiments E1 to E4.

All inputs are float reflectance arrays; ``(B, H, W)`` for multi-band metrics,
``(H, W)`` for single-band ones. SSIM and Canny come from scikit-image (they
are measurement tools here, not the filters under test).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.metrics import structural_similarity


# ----------------------------------------------------------------- E1: noise
def psnr(ref: np.ndarray, test: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB: ``10 log10(R^2 / MSE)``, R = data range."""
    mse = float(np.mean((ref.astype(np.float64) - test) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(data_range ** 2 / mse)


def ssim(ref: np.ndarray, test: np.ndarray, data_range: float = 1.0) -> float:
    """Mean structural similarity over bands (Wang et al. 2004), Gaussian window sigma 1.5."""
    if ref.ndim == 2:
        ref, test = ref[None], test[None]
    return float(np.mean([structural_similarity(r, t, data_range=data_range, gaussian_weights=True,
                                                sigma=1.5, use_sample_covariance=False)
                          for r, t in zip(ref, test)]))


# ----------------------------------------------------------------- E3: detail and edges
def average_gradient(band: np.ndarray) -> float:
    """Mean of ``sqrt((dx^2 + dy^2) / 2)`` with forward differences; higher = more local contrast."""
    dx = np.diff(band, axis=1)[:-1, :]
    dy = np.diff(band, axis=0)[:, :-1]
    return float(np.mean(np.sqrt((dx ** 2 + dy ** 2) / 2)))


def laplacian_variance(band: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian response; a standard sharpness measure."""
    return float(np.var(ndi.laplace(band.astype(np.float64), mode="mirror")))


def shannon_entropy(band: np.ndarray, bins: int = 256) -> float:
    """Entropy (bits) of the grey-level histogram, ``-sum p log2 p``.

    256 bins span the band's own 0.1-99.9 percentile range, so a few extreme
    pixels do not squeeze all other values into a handful of bins.
    """
    lo, hi = np.percentile(band, [0.1, 99.9])
    hist, _ = np.histogram(np.clip(band, lo, hi), bins=bins, range=(lo, hi + 1e-12))
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def canny_edges(band: np.ndarray, sigma: float, low: float, high: float) -> np.ndarray:
    """Canny edge map (bool) with fixed absolute thresholds, so edge counts are comparable."""
    return canny(band.astype(np.float64), sigma=sigma, low_threshold=low, high_threshold=high)


def class_boundaries(labels: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Pixels whose label differs from the pixel to the right or below (bool)."""
    b = np.zeros(labels.shape, bool)
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return b if valid is None else b & valid


def boundary_prf(edges: np.ndarray, reference: np.ndarray, tol: int) -> tuple[float, float, float]:
    """Precision, recall and F1 of an edge map against reference boundaries.

    An edge pixel counts as correct if a reference boundary lies within ``tol``
    pixels (square neighbourhood), and a reference pixel counts as found if an
    edge lies within ``tol`` pixels. This tolerance absorbs the half-pixel
    offsets and the 10 m positional uncertainty of the reference map.
    """
    se = np.ones((2 * tol + 1, 2 * tol + 1), bool)
    near_ref = ndi.binary_dilation(reference, se)
    near_edge = ndi.binary_dilation(edges, se)
    p = (edges & near_ref).sum() / max(edges.sum(), 1)
    r = (reference & near_edge).sum() / max(reference.sum(), 1)
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return float(p), float(r), float(f)


def coherence(band: np.ndarray, rho: float, margin: int = 0) -> float:
    """Energy-weighted structure-tensor coherence (Bigun and Granlund 1987), in [0, 1].

    With gradients gx, gy and a Gaussian window G_rho, the structure tensor is
    ``J = G_rho * [[gx^2, gx gy], [gx gy, gy^2]]``. Its eigenvalues l1 >= l2 give
    ``coherence = (l1 - l2) / (l1 + l2) = sqrt((Jxx - Jyy)^2 + 4 Jxy^2) / (Jxx + Jyy)``.
    1 means all gradients in the window share one orientation (a linear
    feature such as a ridge); 0 means no preferred orientation (speckle).
    Weighting by ``Jxx + Jyy`` keeps flat, low-contrast pixels from dominating.
    ``margin`` pixels on each side only feed the Gaussian window and are left
    out of the sums.
    """
    gy, gx = np.gradient(band.astype(np.float64))
    jxx = ndi.gaussian_filter(gx * gx, rho)
    jyy = ndi.gaussian_filter(gy * gy, rho)
    jxy = ndi.gaussian_filter(gx * gy, rho)
    num = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    energy = jxx + jyy
    if margin:
        num, energy = num[margin:-margin, margin:-margin], energy[margin:-margin, margin:-margin]
    return float(num.sum() / (energy.sum() + 1e-12))


# ----------------------------------------------------------------- E2: map noise
def salt_pepper_index(pred: np.ndarray, n_classes: int, valid: np.ndarray | None = None) -> float:
    """Fraction of pixels whose class differs from the majority class of their 3 x 3 neighbourhood.

    A clean map has large uniform patches and scores near 0; a speckled
    per-pixel classification scores higher.
    """
    counts = np.stack([ndi.uniform_filter((pred == c).astype(np.float32), 3, mode="mirror")
                       for c in range(n_classes)])
    majority = counts.argmax(axis=0)
    diff = majority != pred
    return float(diff[valid].mean() if valid is not None else diff.mean())


# ----------------------------------------------------------------- E4: fidelity
def ndvi(img: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red), bands 3 and 2 of a (4, H, W) stack, clipped to [-1, 1].

    The clip matters for high-pass outputs, whose 'reflectance' can be near zero
    or negative and would otherwise give unbounded ratios.
    """
    nir, red = img[3].astype(np.float64), img[2].astype(np.float64)
    den = nir + red
    out = np.divide(nir - red, den, out=np.zeros_like(den), where=np.abs(den) > 1e-6)
    return np.clip(out, -1, 1)


def pearson_per_band(a: np.ndarray, b: np.ndarray) -> list[float]:
    """Pearson correlation between original and filtered, one value per band."""
    return [float(np.corrcoef(x.ravel(), y.ravel())[0, 1]) for x, y in zip(a, b)]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square difference over all bands and pixels (reflectance units)."""
    return float(np.sqrt(np.mean((a.astype(np.float64) - b) ** 2)))


def spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Mean spectral angle in degrees, ``arccos(<x, y> / (|x| |y|))`` per pixel over the 4 bands.

    The angle ignores overall brightness and measures only the change in
    spectral shape, which is what a classifier relies on.
    """
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    cos = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())
