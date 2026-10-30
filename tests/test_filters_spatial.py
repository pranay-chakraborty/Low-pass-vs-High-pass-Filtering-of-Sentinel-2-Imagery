"""Our NumPy spatial filters must match scipy.ndimage (mode="mirror") to within 1e-5."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from rsfilter import filters_spatial as fs

TOL = 1e-5
RNG = np.random.default_rng(42)
IMG = RNG.random((61, 83)).astype(np.float32)  # odd, non-square size catches index mistakes


def close(a, b):
    return np.max(np.abs(a - b)) < TOL


def test_convolve_matches_scipy_with_asymmetric_kernel():
    kernel = RNG.random((5, 3)).astype(np.float32)  # asymmetric, so a missing flip would fail
    assert close(fs.convolve2d(IMG, kernel), ndi.convolve(IMG, kernel, mode="mirror"))


def test_even_kernel_rejected():
    with pytest.raises(ValueError):
        fs.convolve2d(IMG, np.ones((2, 2)))


@pytest.mark.parametrize("n", [3, 5, 7, 9])
def test_mean(n):
    assert close(fs.mean_filter(IMG, n), ndi.uniform_filter(IMG, n, mode="mirror"))


@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_gaussian(sigma):
    k = fs.gaussian_kernel(sigma)
    assert k.shape == (2 * int(np.ceil(3 * sigma)) + 1,) * 2 and abs(k.sum() - 1) < 1e-6
    assert close(fs.gaussian_filter(IMG, sigma), ndi.gaussian_filter(IMG, sigma, mode="mirror", truncate=3))


@pytest.mark.parametrize("n", [3, 5])
def test_median(n):
    assert close(fs.median_filter(IMG, n, chunk_rows=7), ndi.median_filter(IMG, n, mode="mirror"))


def test_laplacians():
    assert close(fs.laplacian(IMG, 4), ndi.laplace(IMG, mode="mirror"))
    assert close(fs.laplacian(IMG, 8), ndi.convolve(IMG, fs.LAPLACIAN_8, mode="mirror"))


def test_gradient_magnitudes():
    sob = np.hypot(ndi.sobel(IMG, 0, mode="mirror"), ndi.sobel(IMG, 1, mode="mirror"))
    pre = np.hypot(ndi.prewitt(IMG, 0, mode="mirror"), ndi.prewitt(IMG, 1, mode="mirror"))
    assert close(fs.sobel(IMG), sob)
    assert close(fs.prewitt(IMG), pre)


def test_high_pass_and_boost():
    hpf = IMG - ndi.uniform_filter(IMG, 5, mode="mirror")
    assert close(fs.high_pass(IMG, 5), hpf)
    assert close(fs.high_boost(IMG, 5, 1.5), IMG + 1.5 * hpf)


def test_flat_image_behaviour():
    flat = np.full((20, 20), 0.3, np.float32)
    assert close(fs.gaussian_filter(flat, 2), flat)           # LPF keeps a constant
    assert np.max(np.abs(fs.high_pass(flat, 3))) < TOL         # HPF of a constant is zero
    assert np.max(np.abs(fs.sobel(flat))) < TOL


def test_stack_shape_kept():
    stack = RNG.random((4, 30, 40)).astype(np.float32)
    out = fs.median_filter(stack, 3)
    assert out.shape == stack.shape and out.dtype == np.float32
    assert close(out[2], ndi.median_filter(stack[2], 3, mode="mirror"))
