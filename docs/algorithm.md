# Algorithm description

This document describes every processing step in the order the program runs it, with the formulas and kernels used. Function names refer to the package in `src/rsfilter/`.

## 1. Spatial frequency

Spatial frequency is the number of changes in brightness (DN or reflectance) per unit distance. A field of uniform wheat changes slowly across many pixels: low frequency. A street grid, a quarry rim or a quartzite ridge changes abruptly over one or two pixels: high frequency.

* A **low-pass filter (LPF)** keeps the slow changes and suppresses the fast ones. The image looks smoother, noise falls, edges blur.
* A **high-pass filter (HPF)** keeps the fast changes and suppresses the slow ones. Uniform areas go to zero; edges, lines and texture remain.

Every image can be split exactly as `image = low-pass part + high-pass part`. This identity is the basis of the remote-sensing HPF used here (section 4.3).

## 2. Data preparation

### 2.1 Scene search and selection (`data.py`)

1. Search Sentinel-2 L2A scenes between 1 Nov 2024 and 28 Feb 2025 whose footprint contains the whole AOI (77.05-77.30 E, 28.30-28.50 N) and whose scene-wide cloud cover is below 5 %. If none qualify, the window widens to Oct 2024 - Mar 2025.
2. For each candidate, measure inside the AOI only (20 m, the SCL resolution):
   * no data: SCL class 0, or no reflectance value,
   * cloud shadow: SCL class 3,
   * cloud: SCL classes 8 and 9,
   * cirrus: SCL class 10,
   * s2cloudless probability above 40 % (Earth Engine only),
   * `bad_pct` = percentage of AOI pixels with any of the flags above,
   * mean aerosol optical thickness (AOT, 550 nm).
3. Scenes within 0.5 percentage points of the lowest `bad_pct` are treated as equally clean. Among them the lowest AOT wins. Winter smog over Delhi NCR can leave a scene cloud-free by SCL but hazy, and AOT separates those cases.
4. Thumbnails of the top eight candidates are saved side by side (`results/figures/scene_candidates.png`) for a visual check.

### 2.2 Download and grid

The output grid is EPSG:32643 (UTM 43N) with 10 m pixels. The AOI box is projected to UTM and snapped outwards to whole 10 m pixels. Sentinel-2 tiles use the same 10 m alignment, so the grid pixels coincide with native pixels and the four 10 m bands (B2, B3, B4, B8) are copied without resampling.

Earth Engine refuses single downloads above 32 MB, so the grid is cut into 2 x 2 tiles, each tile is downloaded as GeoTIFF, and the tiles are mosaicked with `rasterio.merge`.

ESA WorldCover 2021 (v200) is downloaded in its own EPSG:4326 grid and reprojected onto the Sentinel-2 grid with **nearest-neighbour** resampling (`preprocess.warp_to_grid`). Class codes must never be averaged.

### 2.3 Reflectance

```
rho = (DN - offset) / 10000,   clipped to [0, 1]
```

`offset` is 1000 for scenes with processing baseline 04.00 or later read from raw STAC assets, and 0 for the Earth Engine `_HARMONIZED` collection (already corrected). A pixel is no-data only when **all** bands are 0. A single band at 0 is a very dark value that the harmonised collection clamps to zero, not missing data.

All filtering and all metrics use reflectance in float32. The 2-98 % percentile stretch is used **only** to draw figures.

## 3. Border handling

Filters need pixels outside the image near its edges. The image is padded by mirror reflection without repeating the edge pixel:

```
original row:  a b c d e
padded (r=2):  c b | a b c d e | d c
```

This is NumPy `np.pad(mode="reflect")` and equals `scipy.ndimage` `mode="mirror"`. It avoids the dark frame that zero padding would create.

## 4. Spatial-domain filters (`filters_spatial.py`)

### 4.1 Convolution

For an image f and a kernel w of size (2a+1) x (2b+1):

```
g(x, y) = sum_{s=-a..a} sum_{t=-b..b} w(s, t) f(x - s, y - t)
```

The minus signs rotate the kernel by 180 degrees. Symmetric kernels are unaffected; for Sobel and Prewitt the rotation flips the sign of the response, which the magnitude removes.

**Shift-and-accumulate implementation.** Pad once by (a, b). For every kernel cell (i, j) of the rotated kernel, add `w[i, j] * padded[i : i+H, j : j+W]` to the output. The loop runs over kernel cells (9 for 3 x 3, 81 for 9 x 9), never over pixels, and needs only one extra image of memory. The cost is proportional to H x W x (kernel area).

### 4.2 Low-pass filters

| Filter | Kernel / rule | Sizes used |
|---|---|---|
| Mean (box) | every weight `1 / n^2` | 3, 5, 7, 9 |
| Gaussian | `w(x, y) = exp(-(x^2 + y^2) / (2 sigma^2))`, normalised to sum 1, side `2 * ceil(3 sigma) + 1` | sigma = 1, 2, 3 (sides 7, 13, 19) |
| Median | median of the n x n neighbourhood (non-linear, no kernel) | 3, 5 |

Weights that sum to 1 keep the mean brightness, so a low-passed image stays in reflectance units.

3 x 3 mean kernel:

```
1/9 * | 1 1 1 |
      | 1 1 1 |
      | 1 1 1 |
```

The median is computed with `numpy.lib.stride_tricks.sliding_window_view`, which presents every n x n window as a view without copying. The image is processed in blocks of 128 rows so that the copy `np.median` makes stays small. A median ignores isolated extreme values, so it removes salt-and-pepper noise without blurring a straight step edge.

### 4.3 High-pass filters

**Remote-sensing HPF** (Lillesand, Kiefer and Chipman):

```
HPF_n = original - mean_n(original)        n = 3, 5, 7
```

The result is the local departure from the neighbourhood mean: near zero on uniform ground, large and signed at edges. A larger n passes more of the middle frequencies, so wider features such as ridges show up.

**Laplacian** (second derivative):

```
4-neighbour:  | 0  1  0 |      8-neighbour:  | 1  1  1 |
              | 1 -4  1 |                    | 1 -8  1 |
              | 0  1  0 |                    | 1  1  1 |
```

**Gradient magnitude** (first derivative), `|G| = sqrt(Gx^2 + Gy^2)`:

```
Sobel Gx:  | -1 0 1 |      Prewitt Gx:  | -1 0 1 |
           | -2 0 2 |                   | -1 0 1 |
           | -1 0 1 |                   | -1 0 1 |
```

Gy is the transpose of Gx.

**High-boost / edge enhancement**:

```
enhanced = original + k * HPF_n          k = 0.5, 1.0, 1.5
```

The original brightness stays and edge detail is added back k times. With k = 1 this is the unsharp mask. Unlike the pure HPF outputs, the enhanced image is still a usable reflectance image.

## 5. Frequency-domain filters (`filters_freq.py`)

### 5.1 Pipeline (per band)

1. Mirror-pad by 64 pixels. The discrete Fourier transform assumes the image repeats endlessly, so its left edge meets its right edge. Without padding a filter mixes opposite borders (wrap-around). A test (`test_padding_prevents_wraparound`) shows this leak and its removal.
2. `F = fft2(padded)`.
3. `Fc = fftshift(F)` moves zero frequency to the centre.
4. `G = Fc * H(u, v)`.
5. `g = ifft2(ifftshift(G))`, keep the real part (the imaginary part is rounding noise because H is real and symmetric).
6. Crop the padding.

### 5.2 Frequency distance

For a padded image of M x N pixels, `u = fftfreq(M)` and `v = fftfreq(N)` in cycles per pixel, range [-0.5, 0.5). The Nyquist frequency, the highest an image sampled at 10 m can hold, is 0.5 cycles per pixel (a 20 m period). We use

```
D(u, v) = sqrt(u^2 + v^2) / 0.5
```

so D = 1 is Nyquist and the cutoff D0 is a fraction of Nyquist. D0 = 0.1, 0.2, 0.3 keep periods longer than 200 m, 100 m and 67 m. The meaning is the same for any image size.

### 5.3 Transfer functions

| Type | Low-pass H(u, v) | Behaviour |
|---|---|---|
| Ideal | 1 if D <= D0, else 0 | sharp cut, rings (Gibbs effect) |
| Butterworth, order n = 2 | 1 / (1 + (D / D0)^(2n)) | smooth, H = 0.5 at D0, mild overshoot |
| Gaussian | exp(-D^2 / (2 D0^2)) | smoothest, no ringing |

High-pass: `H_hp = 1 - H_lp` for each type.

**Ringing.** The ideal mask's spatial equivalent is a sinc-like function with side lobes. Each lobe adds a ripple beside every edge. The program measures this on a clean synthetic step edge passed through the same filter code (`ringing_overshoot.csv`, figure 5).

**Link between domains.** A Gaussian LPF with cutoff D0 equals a spatial Gaussian of

```
sigma = 1 / (pi * D0)   pixels
```

because a spatial Gaussian of width sigma has the transfer function `exp(-2 pi^2 sigma^2 f^2)`, and with `D = 2 f` this equals `exp(-D^2 / (2 D0^2))` when `sigma = 1 / (pi D0)`. The test suite checks that the two implementations agree.

### 5.4 Slide replication

On the 512 x 512 sample crop, without padding, square ideal masks of 61, 91 and 121 pixels are centred on the shifted spectrum (`square_mask`, `square_lowpass`). A side of 61 pixels on a 512-pixel grid passes |u|, |v| <= 30/512 = 0.059 cycles per pixel, i.e. periods longer than about 170 m.

## 6. Evaluation

### E1 Noise suppression
Clean crop (512 x 512, the sample area). Two noisy versions (seed 42): Gaussian noise with sigma = 0.02 reflectance, and 2 % salt-and-pepper (each hit pixel set to 0 or 1 in that band). Every LPF is applied and scored against the clean crop:

```
PSNR = 10 log10(1 / MSE)          (data range 1)
SSIM = mean over bands of structural similarity (Gaussian window, sigma 1.5)
```

### E2 Land-cover classification
WorldCover codes are regrouped: Trees (10), Shrub/Grass (20, 30), Cropland (40), Built-up (50), Bare/Mining (60), Water (80, 90). Classes below 0.5 % of the AOI are dropped and reported.

The AOI is split into 128 x 128 blocks and 70 % of the blocks (seed 42) are used for training. Neighbouring pixels are highly correlated. A random pixel split would put near-copies of each test pixel in the training set and inflate accuracy, so whole blocks are held out instead.

Up to 2,000 training pixels per class are drawn from training blocks. Features are the four bands and NDVI, computed after filtering (plus four HPF layers in the "raw + HPF" inputs). A Random Forest (200 trees, seed 42) predicts the whole AOI. Scores use every labelled pixel in the test blocks: overall accuracy, Cohen's kappa, macro-F1, per-class F1 and the salt-and-pepper index (share of pixels whose class differs from the majority of their 3 x 3 neighbourhood).

### E3 Edges and detail (NIR band)
* Average gradient: mean of `sqrt((dx^2 + dy^2) / 2)`.
* Laplacian variance: variance of the 4-neighbour Laplacian.
* Shannon entropy: `-sum p log2 p` over a 256-bin histogram.
* Canny edges (sigma 1, fixed hysteresis thresholds 0.10 / 0.20 on the Sobel magnitude, reflectance units). Fixed thresholds keep edge counts comparable between filters. Only image-type outputs are run through Canny.
* Boundary precision, recall and F1 of the Canny edges against WorldCover class boundaries, with a 2-pixel tolerance; once for all classes and once for the built-up boundary only.
* Orientation coherence (structure tensor, Bigun and Granlund 1987) in the ridge, urban and cropland windows. With gradients gx, gy smoothed by a Gaussian window of sigma = 2 pixels (20 m, below the 100-300 m ridge spacing):

  ```
  J = G * [[gx^2, gx gy], [gx gy, gy^2]]
  coherence = sum sqrt((Jxx - Jyy)^2 + 4 Jxy^2) / sum (Jxx + Jyy)
  ```

  It is 1 when all local gradients share one direction (parallel ridges) and 0 when there is no preferred direction (speckle). No lineament reference map exists for the AOI, so coherence on the ridge window stands in for "how well the filter brings out linear structure".

### E4 Spectral fidelity
Against the original stack: Pearson r per band, RMSE, mean spectral angle

```
SAM = arccos( <x, y> / (|x| |y|) )      per pixel, over the 4 bands, in degrees
```

and the mean absolute change in NDVI.

### E5 Run time
Median of 5 runs on the full-AOI NIR band: spatial box (n = 3 to 31), median (3, 5), Gaussian (sigma 1 to 8), and frequency-domain ideal, Butterworth and Gaussian filters. The Gaussian is run in both domains with matched parameters (D0 = 1 / (pi sigma)).

### Decision matrix
Each application row is filled by one sort on one measured column (see `experiments.decision_matrix`). The table also lists the value without any filter, so the size of the gain is visible:

| Application | Rule |
|---|---|
| Noise removal (Gaussian / salt-and-pepper) | highest PSNR in E1 |
| Land-cover classification | highest Cohen's kappa in E2 |
| Lineament / structural mapping | highest orientation coherence on the ridge window among HPF outputs (E3) |
| Urban boundary delineation | highest built-up boundary F1 (E3) |
| Visual interpretation | highest NIR average gradient among image outputs with SAM <= 5 deg and mean abs NDVI change <= 0.05 (E3 + E4) |
