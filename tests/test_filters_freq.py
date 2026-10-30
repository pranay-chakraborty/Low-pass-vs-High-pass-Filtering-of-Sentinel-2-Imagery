"""Frequency-domain masks and filtering."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from rsfilter import filters_freq as ff
from rsfilter import filters_spatial as fs

RNG = np.random.default_rng(42)
IMG = ndi.gaussian_filter(RNG.random((96, 128)), 1.5).astype(np.float32)


@pytest.mark.parametrize("kind", ["ideal", "butterworth", "gaussian"])
def test_highpass_is_one_minus_lowpass(kind):
    lp = ff.make_mask(kind, (64, 80), 0.2)
    hp = ff.make_mask(kind, (64, 80), 0.2, highpass=True)
    assert np.allclose(lp + hp, 1.0)
    assert lp[32, 40] == 1.0  # zero frequency sits at the centre after fftshift


def test_mask_values_at_cutoff():
    d = ff.distance_grid((200, 200))
    i = np.unravel_index(np.argmin(np.abs(d - 0.3)), d.shape)
    assert abs(ff.lowpass_mask("butterworth", (200, 200), d[i])[i] - 0.5) < 1e-9
    assert abs(ff.lowpass_mask("gaussian", (200, 200), d[i])[i] - np.exp(-0.5)) < 1e-9
    assert np.isclose(d.max(), np.sqrt(2))  # corner = Nyquist on both axes


def test_lowpass_keeps_mean_and_allpass_is_identity():
    for kind in ["ideal", "butterworth", "gaussian"]:
        out = ff.freq_filter(IMG, kind, 0.2)
        assert abs(out.mean() - IMG.mean()) < 1e-3
    same = ff.apply_mask(IMG.astype(float), np.ones(IMG.shape))
    assert np.max(np.abs(same - IMG)) < 1e-6


def test_frequency_gaussian_matches_spatial_gaussian():
    """Gaussian LPF with cutoff D0 equals a spatial Gaussian with sigma = 1 / (pi D0)."""
    sigma = 2.0
    d0 = 1 / (np.pi * sigma)
    freq = ff.freq_filter(IMG, "gaussian", d0, pad=32)
    spat = fs.gaussian_filter(IMG, sigma)
    assert np.max(np.abs(freq - spat)) < 1e-3


def test_padding_prevents_wraparound():
    img = np.zeros((64, 64), np.float32)
    img[:, :4] = 1.0  # bright strip on the left border only
    no_pad = ff.freq_filter(img, "gaussian", 0.1, pad=0)
    padded = ff.freq_filter(img, "gaussian", 0.1, pad=32)
    assert no_pad[:, -1].mean() > 0.05   # without padding, brightness leaks to the right edge
    assert padded[:, -1].mean() < 1e-3


def test_square_mask():
    m = ff.square_mask((512, 512), 61)
    assert m.sum() == 61 * 61 and m[256, 256] == 1
    out = ff.square_lowpass(IMG, 21)
    assert out.shape == IMG.shape


def test_ringing_order():
    """Ideal LPF overshoots a step by roughly the Gibbs 9 %; Butterworth less; Gaussian not at all."""
    ideal = ff.overshoot_pct(ff.step_response("ideal", 0.1, 2, 64))
    butter = ff.overshoot_pct(ff.step_response("butterworth", 0.1, 2, 64))
    gauss = ff.overshoot_pct(ff.step_response("gaussian", 0.1, 2, 64))
    assert 7 < ideal < 11 and 0 < butter < ideal and gauss < 0.1
