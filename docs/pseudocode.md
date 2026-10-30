# Pseudo-code and data structures

## 1. Data structures

| Name | Type | Shape / content | Where |
|---|---|---|---|
| `cfg` | `dict` (from `config.yaml`) | AOI, dates, bands, filter parameters, experiment settings, seed 42; `cfg["paths"]` holds absolute `pathlib.Path` objects | `config.load_config` |
| `grid` | `dict` | `crs` "EPSG:32643", `transform` (affine, 10 m), `width` 2491, `height` 2261, `bounds` | `data.aoi_grid` |
| candidate table | `pandas.DataFrame` | one row per scene: `date, image_id, product_id, tile, scene_cloud_pct, nodata_pct, shadow_pct, cloud_pct, cirrus_pct, prob_pct, bad_pct, aot_mean, clean_group, rank` | `data.gee_candidates`, `data.rank_candidates` |
| `meta` | `dict` -> `scene_metadata.json` | backend, product ID, date/time, cloud metrics in the AOI, AOT, processing baseline, DN offset, CRS, size, selection rationale, download time | `data.fetch_scene` |
| `dn` | `numpy.ndarray uint16` | `(5, H, W)`: B2, B3, B4, B8, SCL | `data/raw/s2_aoi_dn.tif` |
| `img` | `numpy.ndarray float32` | `(4, H, W)` reflectance in [0, 1]: blue, green, red, NIR | `data/processed/reflectance.tif` |
| `wc` | `numpy.ndarray uint8` | `(H, W)` WorldCover codes 10..100 | `data/processed/worldcover.tif` |
| `profile` | `dict` | rasterio profile: driver, dtype, count, width, height, crs, transform | read with every GeoTIFF |
| `kernel` | `numpy.ndarray float32` | `(k, k)`, k odd (3 to 19) | `filters_spatial` |
| `H` (mask) | `numpy.ndarray float64` | `(M, N)` = padded image size, values in [0, 1], zero frequency at the centre | `filters_freq.make_mask` |
| `FilterSpec` | dataclass | `name` (CSV key), `label`, `family` (LPF-spatial, LPF-freq, HPF-spatial, HPF-freq, HPF-enhanced), `output` ("image" / "edge"), `fn: (4,H,W) -> (4,H,W)` | `bank.filter_bank` (43 entries) |
| `labels` | `numpy.ndarray int16` | `(H, W)` class index 0..K-1, -1 = not used | `classify.regroup` |
| `train_mask` | `numpy.ndarray bool` | `(H, W)`, True in training blocks | `classify.block_split` |
| `features` | `numpy.ndarray float32` | `(H*W, F)`, F = 5 (bands + NDVI) or 9 (+ 4 HPF layers) | `classify.features` |
| result tables | `pandas.DataFrame` -> CSV | one row per filter (or input) per experiment | `results/tables/*.csv` |

`H = 2261`, `W = 2491` for the full AOI; `H = W = 512` for the sample crop.

## 2. Main flow

```
PROGRAM rsfilter
    cfg <- load_config("config.yaml")

    COMMAND fetch:
        FOR backend IN [gee, pc, earthsearch]:          # first that works wins
            FOR window IN [Nov-Feb, Oct-Mar]:           # widen only if needed
                candidates <- search scenes over AOI, scene cloud < 5 %
                IF candidates is empty: CONTINUE
                FOR each scene: measure nodata, cloud, cirrus, shadow, s2cloudless, AOT inside AOI
                ranked <- rank_candidates(candidates, tolerance = 0.5 %)
                save thumbnails of top 8, save candidate CSV
                download best scene on grid (2 x 2 tiles -> mosaic), WorldCover -> nearest-neighbour regrid
                write scene_metadata.json
                GOTO prepared
        prepared:
        img <- clip((dn - offset) / 10000, 0, 1); nodata <- all bands == 0
        write reflectance.tif, worldcover.tif
        sample window <- 512 x 512 window with highest WorldCover class entropy
        write data/sample/*, quick-look PNGs

    COMMAND evaluate:
        img, wc <- read processed files
        E1 <- e1_noise(sample crop)
        E2 <- e2_classification(img, wc)
        E3, E4 <- e3_e4(img, wc)
        ringing <- ringing_table()
        E5 <- e5_runtime(img[NIR])
        D <- decision_matrix(E1, E2, E3, E4)
        write every table to results/tables/*.csv

    COMMAND figures:   read tables + images, write results/figures/*.png
    COMMAND demo:      evaluate on data/sample with reduced E2/E5, write results/demo/
```

## 3. Spatial convolution

```
FUNCTION convolve2d(f[H, W], w[kh, kw]):
    ry <- kh div 2;  rx <- kw div 2
    p  <- mirror_pad(f, ry, rx)                 # (H + 2ry, W + 2rx)
    w  <- rotate_180(w)                         # convolution, not correlation
    g  <- zeros(H, W)
    FOR i IN 0 .. kh-1:
        FOR j IN 0 .. kw-1:
            IF w[i, j] != 0:
                g <- g + w[i, j] * p[i : i+H, j : j+W]   # whole-image shifted slice
    RETURN g
```

```
FUNCTION median_filter(f[H, W], n, chunk = 128):
    r <- n div 2;  p <- mirror_pad(f, r)
    FOR y0 IN 0, chunk, 2*chunk, ... < H:
        y1 <- min(y0 + chunk, H)
        windows <- sliding_window_view(p[y0 : y1 + 2r, :], (n, n))   # view, (y1-y0, W, n, n)
        g[y0:y1, :] <- median(windows over its last two axes)
    RETURN g
```

```
high_pass(f, n)       = f - convolve2d(f, box(n))
high_boost(f, n, k)   = f + k * high_pass(f, n)
sobel(f)              = sqrt(convolve2d(f, Sx)^2 + convolve2d(f, Sx^T)^2)
```

## 4. Frequency-domain filter

```
FUNCTION freq_filter(f[H, W], kind, D0, highpass, order = 2, pad = 64):
    p   <- mirror_pad(f, pad)                    # (M, N) = (H + 128, W + 128)
    u   <- fftshift(fftfreq(M));  v <- fftshift(fftfreq(N))
    D   <- sqrt(u[:,None]^2 + v[None,:]^2) / 0.5           # 1.0 = Nyquist
    IF kind = ideal:        Hlp <- (D <= D0)
    IF kind = butterworth:  Hlp <- 1 / (1 + (D / D0)^(2 order))
    IF kind = gaussian:     Hlp <- exp(-D^2 / (2 D0^2))
    H   <- 1 - Hlp IF highpass ELSE Hlp
    F   <- fftshift(fft2(p))
    g   <- real(ifft2(ifftshift(F * H)))
    RETURN g[pad : pad+H, pad : pad+W]
```

## 5. Classification experiment

```
FUNCTION e2(img, wc):
    labels, classes, dropped <- regroup(wc, min_fraction = 0.005)
    train_mask <- block_split(128 x 128 blocks, 70 % train, seed 42)
    FOR input IN [raw, each LPF, each high-boost, raw + HPF_n]:
        image, extra <- input(img)
        X <- features(image, extra)                       # (H*W, 5 or 9)
        idx <- up to 2000 random training-block pixels per class
        rf  <- RandomForest(200 trees, seed 42).fit(X[idx], labels[idx])
        pred <- rf.predict(X)  reshaped to (H, W)
        test <- NOT train_mask AND labels >= 0
        record OA, kappa, macro-F1, per-class F1 on test; salt-and-pepper index on pred
```

## 6. Boundary agreement (E3)

```
FUNCTION boundary_prf(edges, reference, tol = 2):
    se <- ones(2 tol + 1, 2 tol + 1)
    precision <- |edges AND dilate(reference, se)| / |edges|
    recall    <- |reference AND dilate(edges, se)| / |reference|
    F1        <- 2 P R / (P + R)
```

## 7. Decision matrix

```
FOR each application:
    candidates <- rows of the relevant table that meet the application's filter (e.g. HPF only)
    sort candidates by the application's metric (descending)
    best, runner_up <- first two rows
    trade_off <- text built from E3 (gradient vs original) and E4 (SAM, NDVI change) of best
```
