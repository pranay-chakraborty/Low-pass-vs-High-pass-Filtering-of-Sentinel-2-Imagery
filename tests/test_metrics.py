"""Metrics and classification helpers on small inputs with known answers."""
import numpy as np

from rsfilter import classify, metrics

RNG = np.random.default_rng(42)


def test_psnr_known_value():
    a = np.zeros((10, 10))
    b = np.full((10, 10), 0.1)          # MSE = 0.01 -> PSNR = 20 dB for range 1
    assert abs(metrics.psnr(a, b) - 20.0) < 1e-9


def test_ssim_identity_and_drop():
    img = RNG.random((3, 64, 64)).astype(np.float32)
    assert abs(metrics.ssim(img, img) - 1) < 1e-6
    assert metrics.ssim(img, img + RNG.normal(0, 0.2, img.shape)) < 0.9


def test_spectral_angle():
    a = RNG.random((4, 8, 8)) + 0.1
    assert metrics.spectral_angle(a, 2 * a) < 1e-4    # brightness scaling does not change the angle
    x = np.zeros((4, 1, 1)); x[0] = 1
    y = np.zeros((4, 1, 1)); y[1] = 1
    assert abs(metrics.spectral_angle(x, y) - 90) < 1e-6


def test_ndvi_and_clip():
    img = np.zeros((4, 1, 2))
    img[2] = [[0.1, 0.0]]
    img[3] = [[0.3, 0.0]]
    assert np.allclose(metrics.ndvi(img), [[0.5, 0.0]])  # (0.3-0.1)/(0.3+0.1); zero denominator -> 0


def test_boundary_prf_with_tolerance():
    ref = np.zeros((20, 20), bool); ref[:, 10] = True
    edges = np.zeros((20, 20), bool); edges[:, 12] = True   # 2 pixels away
    assert metrics.boundary_prf(edges, ref, 2) == (1.0, 1.0, 1.0)
    assert metrics.boundary_prf(edges, ref, 1)[2] == 0.0


def test_salt_pepper_index():
    clean = np.zeros((10, 10), int)
    assert metrics.salt_pepper_index(clean, 2) == 0.0
    noisy = clean.copy(); noisy[5, 5] = 1
    assert abs(metrics.salt_pepper_index(noisy, 2) - 0.01) < 1e-9


def test_block_split_whole_blocks():
    mask = classify.block_split((300, 260), 128, 0.7, 42)
    assert mask.shape == (300, 260)
    blocks = mask[::128, ::128]
    for i in range(blocks.shape[0]):           # every block is all-train or all-test
        for j in range(blocks.shape[1]):
            b = mask[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
            assert b.all() or not b.any()


def test_regroup_drops_rare_class():
    wc = np.full((100, 100), 40, np.uint8)
    wc[0, :10] = 80                          # 0.1 % water -> dropped at 0.5 %
    labels, kept, dropped = classify.regroup(wc, {"Cropland": [40], "Water": [80, 90]}, 0.005)
    assert kept == ["Cropland"] and "Water" in dropped and (labels[0, :10] == -1).all()


def test_coherence_lines_vs_speckle():
    x = np.arange(64)
    stripes = np.tile(np.sin(x / 3.0), (64, 1))          # vertical stripes: one orientation
    assert metrics.coherence(stripes, 2, 6) > 0.95
    assert metrics.coherence(RNG.random((64, 64)), 2, 6) < 0.5
