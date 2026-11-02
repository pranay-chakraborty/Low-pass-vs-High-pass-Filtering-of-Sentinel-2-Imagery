# Code reference

Generated from the docstrings by `scripts/make_code_reference.py`. Read the source for the full explanation and formulas.

| Module | Purpose |
|---|---|
| `config.py` | Configuration loading and project paths. |
| `data.py` | Scene search, selection and download for the study area. |
| `preprocess.py` | Pre-processing: reflectance scaling, WorldCover regridding, sample crop, display stretch. |
| `filters_spatial.py` | Spatial-domain filters, written directly in NumPy. |
| `filters_freq.py` | Frequency-domain filters using the 2-D discrete Fourier transform. |
| `bank.py` | The filter bank: every filter configuration used in the experiments, by name. |
| `metrics.py` | Quality metrics used by experiments E1 to E4. |
| `classify.py` | Land-cover classification used in experiment E2. |
| `experiments.py` | Experiments E1 to E5 and the decision matrix. |
| `viz.py` | Figures for the report and slides. Each function saves one PNG at 200 dpi and returns its path. |
| `cli.py` | Command-line interface: ``python -m rsfilter <command>`` (the executable exposes the same commands). |

## `rsfilter/config.py`

Configuration loading and project paths.

- **`project_root() -> 'Path'`** Return the project root directory.
- **`load_config() -> 'dict'`** Read ``config.yaml`` and add absolute paths under ``cfg['paths']``.

## `rsfilter/data.py`

Scene search, selection and download for the study area.

- **`aoi_grid(cfg: 'dict') -> 'dict'`** Return the output grid for the AOI: CRS, affine transform, width, height.
- **`rank_candidates(df: 'pd.DataFrame', tolerance: 'float') -> 'pd.DataFrame'`** Order candidate scenes from best to worst.
- **`selection_rationale(ranked: 'pd.DataFrame', tolerance: 'float', window: 'str') -> 'str'`** Plain-language explanation of why the top-ranked scene was chosen.
- **`gee_candidates(cfg: 'dict', start: 'str', end: 'str') -> 'pd.DataFrame'`** Score every GEE Sentinel-2 L2A scene whose footprint contains the AOI.
- **`gee_thumbnail(cfg: 'dict', image_id: 'str', bands: 'list[str]', size: 'int' = 420) -> 'np.ndarray'`** Download a small RGB PNG of the AOI for visual checking of a candidate scene.
- **`download_gee(cfg: 'dict', row: 'pd.Series', raw_dir: 'Path') -> 'dict'`** Download the selected scene (B2, B3, B4, B8, SCL) and WorldCover via GEE.
- **`stac_candidates(backend: 'str', cfg: 'dict', start: 'str', end: 'str') -> 'tuple[pd.DataFrame, dict]'`** Score STAC items over the AOI using their SCL and AOT assets (no s2cloudless).
- **`download_stac(backend: 'str', cfg: 'dict', item, raw_dir: 'Path') -> 'dict'`** Write the selected STAC scene and WorldCover onto the output grid.
- **`download_worldcover_pc(cfg: 'dict', raw_dir: 'Path') -> 'Path'`** Read ESA WorldCover 2021 (v200) from Planetary Computer onto the output grid.
- **`save_candidate_sheet(cfg: 'dict', ranked: 'pd.DataFrame') -> 'Path'`** Save thumbnails of the top-ranked GEE candidates side by side for visual checking.
- **`fetch_scene(cfg: 'dict', backend: 'str' = 'auto') -> 'dict'`** Search, rank, download and document one Sentinel-2 L2A scene for the AOI.

## `rsfilter/preprocess.py`

Pre-processing: reflectance scaling, WorldCover regridding, sample crop, display stretch.

- **`warp_to_grid(src_path: 'Path', grid: 'dict', out_path: 'Path') -> 'Path'`** Reproject a categorical raster onto ``grid`` with nearest-neighbour resampling.
- **`to_reflectance(dn: 'np.ndarray', offset: 'int' = 0) -> 'tuple[np.ndarray, np.ndarray]'`** Convert L2A digital numbers to surface reflectance.
- **`build_processed(cfg: 'dict') -> 'dict'`** Turn ``data/raw`` downloads into analysis-ready files in ``data/processed``.
- **`read_stack(path: 'Path') -> 'tuple[np.ndarray, dict]'`** Read a multi-band GeoTIFF as float32 ``(bands, H, W)`` plus its profile.
- **`read_labels(path: 'Path') -> 'np.ndarray'`** Read a single-band label raster (WorldCover codes) as ``uint8 (H, W)``.
- **`write_stack(path: 'Path', arr: 'np.ndarray', profile: 'dict') -> 'None'`** Write ``(bands, H, W)`` or ``(H, W)`` array with the georeferencing in ``profile``.
- **`class_entropy(labels: 'np.ndarray') -> 'float'`** Shannon entropy (bits) of the class frequencies in ``labels``.
- **`choose_sample_window(wc: 'np.ndarray', size: 'int', step: 'int' = 64) -> 'tuple[int, int]'`** Pick the ``size`` x ``size`` window with the most varied land cover.
- **`make_sample(cfg: 'dict') -> 'dict'`** Export the 512 x 512 demo crop (reflectance + WorldCover + metadata) to ``data/sample``.
- **`stretch(img: 'np.ndarray', lo: 'float' = 2, hi: 'float' = 98, ref: 'np.ndarray | None' = None) -> 'np.ndarray'`** Percentile stretch to [0, 1] for display only (never used before filtering or metrics).

## `rsfilter/filters_spatial.py`

Spatial-domain filters, written directly in NumPy.

- **`pad_reflect(img: 'np.ndarray', ry: 'int', rx: 'int | None' = None) -> 'np.ndarray'`** Mirror-pad a 2-D array by ``ry`` rows and ``rx`` columns on each side.
- **`convolve2d(img: 'np.ndarray', kernel: 'np.ndarray') -> 'np.ndarray'`** Convolve an image (or each band of a stack) with an odd-sized 2-D kernel.
- **`box_kernel(n: 'int') -> 'np.ndarray'`** n x n mean kernel, every weight 1 / n^2 (weights sum to 1, so mean brightness is kept).
- **`gaussian_kernel(sigma: 'float') -> 'np.ndarray'`** Normalised 2-D Gaussian kernel.
- **`mean_filter(img: 'np.ndarray', n: 'int') -> 'np.ndarray'`** Box (mean) low-pass filter: each pixel becomes the average of its n x n neighbourhood.
- **`gaussian_filter(img: 'np.ndarray', sigma: 'float') -> 'np.ndarray'`** Gaussian low-pass filter. Weights fall off smoothly, so it blurs with less ringing than the box.
- **`median_filter(img: 'np.ndarray', n: 'int', chunk_rows: 'int' = 128) -> 'np.ndarray'`** Median filter (non-linear): each pixel becomes the median of its n x n neighbourhood.
- **`high_pass(img: 'np.ndarray', n: 'int') -> 'np.ndarray'`** Classic remote-sensing high-pass: ``HPF = original - mean_filter(original, n)``.
- **`laplacian(img: 'np.ndarray', neighbours: 'int' = 4) -> 'np.ndarray'`** Second-derivative (Laplacian) edge response with the 4- or 8-neighbour kernel.
- **`sobel(img: 'np.ndarray') -> 'np.ndarray'`** Sobel gradient magnitude ``sqrt(Gx^2 + Gy^2)``; centre row/column weighted 2.
- **`prewitt(img: 'np.ndarray') -> 'np.ndarray'`** Prewitt gradient magnitude ``sqrt(Gx^2 + Gy^2)``; equal weights across the kernel.
- **`high_boost(img: 'np.ndarray', n: 'int', k: 'float') -> 'np.ndarray'`** Edge enhancement: ``enhanced = original + k * HPF_n``.

## `rsfilter/filters_freq.py`

Frequency-domain filters using the 2-D discrete Fourier transform.

- **`distance_grid(shape: 'tuple[int, int]') -> 'np.ndarray'`** Normalised radial frequency D (1.0 = Nyquist) for a centred (fftshifted) spectrum.
- **`lowpass_mask(kind: 'str', shape: 'tuple[int, int]', d0: 'float', order: 'int' = 2) -> 'np.ndarray'`** Low-pass transfer function H(u, v) of type 'ideal', 'butterworth' or 'gaussian'.
- **`make_mask(kind: 'str', shape: 'tuple[int, int]', d0: 'float', highpass: 'bool' = False, order: 'int' = 2) -> 'np.ndarray'`** Return the LPF mask, or ``1 - LPF`` when ``highpass`` is True.
- **`apply_mask(img: 'np.ndarray', mask: 'np.ndarray') -> 'np.ndarray'`** Multiply the centred spectrum of a 2-D array by ``mask`` and transform back (no padding).
- **`freq_filter(img: 'np.ndarray', kind: 'str', d0: 'float', highpass: 'bool' = False, order: 'int' = 2, pad: 'int' = 64) -> 'np.ndarray'`** Filter an image (or each band of a stack) in the frequency domain.
- **`square_mask(shape: 'tuple[int, int]', side: 'int') -> 'np.ndarray'`** Square ideal low-pass mask: ones in a centred ``side`` x ``side`` block (as on the assignment slide).
- **`square_lowpass(img: 'np.ndarray', side: 'int') -> 'np.ndarray'`** Slide replication: keep a ``side`` x ``side`` block of the centred spectrum of a 2-D image.
- **`log_spectrum(img: 'np.ndarray') -> 'np.ndarray'`** Centred log-magnitude spectrum ``log(1 + |F(u, v)|)`` for display.
- **`step_response(kind: 'str', d0: 'float', order: 'int', pad: 'int', lo: 'float' = 0.1, hi: 'float' = 0.3) -> 'np.ndarray'`** Filter a synthetic vertical step edge (lo -> hi) and return the middle row.
- **`overshoot_pct(profile: 'np.ndarray', lo: 'float' = 0.1, hi: 'float' = 0.3) -> 'float'`** Largest excursion beyond the step levels, as a percentage of the step height.

## `rsfilter/bank.py`

The filter bank: every filter configuration used in the experiments, by name.

- **`FilterSpec`** FilterSpec(name: 'str', label: 'str', family: 'str', output: 'str', fn: 'Callable[[np.ndarray], np.ndarray]')
- **`filter_bank(cfg: 'dict') -> 'list[FilterSpec]'`** Build the full list of filter configurations from ``config.yaml``.
- **`get(cfg: 'dict', name: 'str') -> 'FilterSpec'`** Look up one filter by name (raises ``KeyError`` with the valid names).

## `rsfilter/metrics.py`

Quality metrics used by experiments E1 to E4.

- **`psnr(ref: 'np.ndarray', test: 'np.ndarray', data_range: 'float' = 1.0) -> 'float'`** Peak signal-to-noise ratio in dB: ``10 log10(R^2 / MSE)``, R = data range.
- **`ssim(ref: 'np.ndarray', test: 'np.ndarray', data_range: 'float' = 1.0) -> 'float'`** Mean structural similarity over bands (Wang et al. 2004), Gaussian window sigma 1.5.
- **`average_gradient(band: 'np.ndarray') -> 'float'`** Mean of ``sqrt((dx^2 + dy^2) / 2)`` with forward differences; higher = more local contrast.
- **`laplacian_variance(band: 'np.ndarray') -> 'float'`** Variance of the 4-neighbour Laplacian response; a standard sharpness measure.
- **`shannon_entropy(band: 'np.ndarray', bins: 'int' = 256) -> 'float'`** Entropy (bits) of the grey-level histogram, ``-sum p log2 p``.
- **`canny_edges(band: 'np.ndarray', sigma: 'float', low: 'float', high: 'float') -> 'np.ndarray'`** Canny edge map (bool) with fixed absolute thresholds, so edge counts are comparable.
- **`class_boundaries(labels: 'np.ndarray', valid: 'np.ndarray | None' = None) -> 'np.ndarray'`** Pixels whose label differs from the pixel to the right or below (bool).
- **`boundary_prf(edges: 'np.ndarray', reference: 'np.ndarray', tol: 'int') -> 'tuple[float, float, float]'`** Precision, recall and F1 of an edge map against reference boundaries.
- **`coherence(band: 'np.ndarray', rho: 'float', margin: 'int' = 0) -> 'float'`** Energy-weighted structure-tensor coherence (Bigun and Granlund 1987), in [0, 1].
- **`salt_pepper_index(pred: 'np.ndarray', n_classes: 'int', valid: 'np.ndarray | None' = None) -> 'float'`** Fraction of pixels whose class differs from the majority class of their 3 x 3 neighbourhood.
- **`ndvi(img: 'np.ndarray') -> 'np.ndarray'`** NDVI = (NIR - Red) / (NIR + Red), bands 3 and 2 of a (4, H, W) stack, clipped to [-1, 1].
- **`pearson_per_band(a: 'np.ndarray', b: 'np.ndarray') -> 'list[float]'`** Pearson correlation between original and filtered, one value per band.
- **`rmse(a: 'np.ndarray', b: 'np.ndarray') -> 'float'`** Root-mean-square difference over all bands and pixels (reflectance units).
- **`spectral_angle(a: 'np.ndarray', b: 'np.ndarray') -> 'float'`** Mean spectral angle in degrees, ``arccos(<x, y> / (|x| |y|))`` per pixel over the 4 bands.

## `rsfilter/classify.py`

Land-cover classification used in experiment E2.

- **`regroup(wc: 'np.ndarray', classes: 'dict[str, list[int]]', min_fraction: 'float') -> 'tuple[np.ndarray, list[str], dict[str, float]]'`** Map WorldCover codes to class indices; drop rare classes.
- **`block_split(shape: 'tuple[int, int]', block: 'int', train_fraction: 'float', seed: 'int') -> 'np.ndarray'`** Boolean (H, W) mask, True for pixels in training blocks.
- **`features(img: 'np.ndarray', extra: 'np.ndarray | None' = None) -> 'np.ndarray'`** Per-pixel features: the 4 bands, NDVI, and optional extra layers. Shape (H*W, F).
- **`sample_training(labels: 'np.ndarray', train_mask: 'np.ndarray', per_class: 'int', seed: 'int') -> 'np.ndarray'`** Flat indices of up to ``per_class`` random training pixels for each class.
- **`run_classification(feat: 'np.ndarray', labels: 'np.ndarray', train_mask: 'np.ndarray', class_names: 'list[str]', cfg: 'dict') -> 'tuple[dict, np.ndarray]'`** Train, predict the full grid, and score on the test blocks.

## `rsfilter/experiments.py`

Experiments E1 to E5 and the decision matrix.

- **`add_noise(clean: 'np.ndarray', cfg: 'dict') -> 'dict[str, np.ndarray]'`** Return the two noisy versions of ``clean`` used in E1 (seed from config).
- **`e1_noise(clean: 'np.ndarray', cfg: 'dict') -> 'pd.DataFrame'`** Apply every low-pass filter to both noisy crops and score against the clean crop.
- **`e2_inputs(cfg: 'dict', quick: 'bool' = False) -> 'list[tuple[str, str, object]]'`** List of (name, family, function returning (image, extra_layers)) compared in E2.
- **`e2_classification(img: 'np.ndarray', wc: 'np.ndarray', cfg: 'dict', quick: 'bool' = False) -> 'tuple[pd.DataFrame, dict, dict]'`** Train and score one Random Forest per input. Returns (table, predictions, class info).
- **`e3_row(name: 'str', family: 'str', output: 'str', band: 'np.ndarray', boundaries: 'dict', cfg: 'dict', insets: 'dict | None') -> 'dict'`** E3 metrics for one NIR band.
- **`e4_row(name: 'str', family: 'str', output: 'str', orig: 'np.ndarray', out: 'np.ndarray', ndvi_orig: 'np.ndarray') -> 'dict'`** E4 metrics: how far a filtered stack moved from the original radiometry.
- **`e3_e4(img: 'np.ndarray', wc: 'np.ndarray', cfg: 'dict', insets: 'dict | None') -> 'tuple[pd.DataFrame, pd.DataFrame]'`** Run E3 and E4 together so each filter is applied to the full stack only once.
- **`e5_runtime(band: 'np.ndarray', cfg: 'dict', quick: 'bool' = False) -> 'pd.DataFrame'`** Median wall-clock time of spatial and frequency implementations on one band.
- **`ringing_table(cfg: 'dict') -> 'pd.DataFrame'`** Overshoot of each frequency-domain LPF on a synthetic 0.1 -> 0.3 step edge.
- **`decision_matrix(e1: 'pd.DataFrame', e2: 'pd.DataFrame', e3: 'pd.DataFrame', e4: 'pd.DataFrame', cfg: 'dict') -> 'pd.DataFrame'`** Pick the best and runner-up filter per application from E1 to E4.
- **`run_all(img: 'np.ndarray', wc: 'np.ndarray', profile: 'dict', cfg: 'dict', out_dir: 'Path', clean_crop: 'np.ndarray', insets: 'dict | None', quick: 'bool' = False) -> 'dict[str, pd.DataFrame]'`** Run E1 to E5 and the decision matrix; write CSVs and E2 class maps under ``out_dir``.

## `rsfilter/viz.py`

Figures for the report and slides. Each function saves one PNG at 200 dpi and returns its path.

- **`rgb(img: 'np.ndarray', bands: 'tuple[int, int, int]') -> 'np.ndarray'`** Stack three bands of a ``(4, H, W)`` array into ``(H, W, 3)`` for display.
- **`crop(arr: 'np.ndarray', box: 'dict') -> 'np.ndarray'`** Cut a square window (centre row/col, size) from the last two axes.
- **`scale_bar(ax, width_px: 'int', height_px: 'int', pixel_m: 'float' = 10, length_m: 'float | None' = None) -> 'None'`** Black/white scale bar in the lower-left corner of an image axis.
- **`quicklooks(img: 'np.ndarray', meta: 'dict', out_dir: 'Path') -> 'list[Path]'`** True-colour (R-G-B) and false-colour (NIR-R-G) quick-looks of the whole AOI.
- **`candidate_sheet(thumbs: 'list[np.ndarray]', rows: 'list[dict]', out: 'Path') -> 'Path'`** Grid of candidate-scene thumbnails labelled with date, flagged AOI % and AOT.
- **`fig_study_area(cfg, img, meta, out: 'Path') -> 'Path'`** India locator, true-colour and false-colour composites with the inset windows marked.
- **`fig_lpf_grid(cfg, img, out)`** Low-pass results (spatial and frequency domain) with the three zoomed insets.
- **`fig_hpf_grid(cfg, img, out)`** High-pass edge layers on the NIR band with the three zoomed insets.
- **`fig_boost_grid(cfg, img, out)`** High-boost (original + k x HPF) results with the three zoomed insets.
- **`fig_slide_grid(cfg, img, names, out, mode, regions=('ridge', 'urban'))`** Landscape grid for slides: rows = sub-areas, columns = filters (same display rules as figure 2).
- **`fig_spectra(band: 'np.ndarray', cfg, out: 'Path') -> 'Path'`** Log-magnitude spectra of the NIR crop before and after low- and high-pass filtering.
- **`fig_square_masks(band: 'np.ndarray', cfg, out: 'Path') -> 'Path'`** Square ideal low-pass masks of 61, 91, 121 pixels on the 512 x 512 crop (slide layout).
- **`fig_ringing(cfg, img, out: 'Path') -> 'Path'`** Ideal vs Butterworth vs Gaussian LPF (D0 = 0.1): real zoom (top) and synthetic step response (bottom).
- **`fig_line_profile(cfg, img, out: 'Path') -> 'Path'`** NIR line profile across a sharp edge: original vs low-pass vs high-boost.
- **`fig_class_maps(cfg, out: 'Path') -> 'Path'`** WorldCover (regrouped) next to the raw, best-LPF and best-HPF Random Forest maps.
- **`fig_e1(e1: 'pd.DataFrame', out: 'Path') -> 'Path'`** E1 chart: PSNR of every low-pass filter for both noise types.
- **`fig_e2(e2: 'pd.DataFrame', out: 'Path') -> 'Path'`** E2 chart: Cohen's kappa and salt-and-pepper index per classification input.
- **`fig_e3(e3: 'pd.DataFrame', out: 'Path') -> 'Path'`** E3 chart: boundary F1 scores and orientation coherence on the ridge window.
- **`fig_e4(e4: 'pd.DataFrame', out: 'Path') -> 'Path'`** E4 chart: spectral angle and NDVI change per filter (log scale).
- **`fig_e5(e5: 'pd.DataFrame', out: 'Path') -> 'Path'`** E5 chart: run time against kernel size, with the FFT range as a band.
- **`make_all(cfg) -> 'list[Path]'`** Regenerate every figure of section 7 from processed data and the CSV tables.
- **`make_demo(cfg, out_dir: 'Path') -> 'list[Path]'`** Figures for the offline demo, drawn from the sample crop and the demo tables only.

## `rsfilter/cli.py`

Command-line interface: ``python -m rsfilter <command>`` (the executable exposes the same commands).

- **`cmd_fetch(args, cfg)`** Download the scene and WorldCover, build processed data, sample crop and quick-looks.
- **`cmd_filter(args, cfg)`** Apply one filter to every band of a GeoTIFF and keep its georeferencing.
- **`cmd_evaluate(args, cfg)`** Run E1 to E5 and the decision matrix on the given image and labels.
- **`cmd_figures(args, cfg)`** Regenerate every figure from the processed data and the CSV tables.
- **`cmd_demo(args, cfg)`** Offline demonstration on data/sample: reduced experiments plus figures.
- **`main(argv: 'list[str] | None' = None) -> 'None'`** Parse arguments and run one command.
