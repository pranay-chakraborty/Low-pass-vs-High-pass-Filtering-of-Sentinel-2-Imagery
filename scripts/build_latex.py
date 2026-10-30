"""Build the LaTeX report and the Beamer slides (IISER Mohali templates) from results/tables.

The prose lives in two templates:
    report/latex/report_template.tex        -> report/Group1_EES403_Report.pdf
    presentation/beamer/slides_template.tex -> presentation/Group1_EES403_Slides.pdf
                                               presentation/Group1_EES403_Slides_notes.pdf
Placeholders are written <<key>> (value as text) or <<p:key>> (a filter code such as
gauss_s1 printed as a readable name). Every value comes from values() or from the CSV
tables read here, so no result is typed by hand. The script also redraws the
processing flowchart (report/latex/flowchart_template.tex -> docs/flowchart.pdf/.png)
and refreshes the results block in README.md.

Run: python scripts/build_latex.py   (needs pdflatex, bibtex and latexmk on PATH)
"""
import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
REPORT_DIR = ROOT / "report" / "latex"
SLIDES_DIR = ROOT / "presentation" / "beamer"


def f(x, n=2):
    return f"{x:.{n}f}"


def read(name):
    return pd.read_csv(T / f"{name}.csv")


def num(v: float) -> str:
    """PSNR-sized values with 2 decimals, scores in [0, 1] and gradients with 3."""
    return f"{v:.2f}" if abs(v) >= 10 else f"{v:.3f}"


def values() -> dict:
    """Every number the report and slides quote, read from results/tables and the scene metadata."""
    meta = json.loads((ROOT / "data" / "raw" / "scene_metadata.json").read_text(encoding="utf-8"))
    cand = read("scene_candidates")
    e1, e2, e3, e4, e5 = (read(n) for n in ["e1_noise", "e2_classification", "e3_edges", "e4_fidelity",
                                             "e5_runtime"])
    ring = read("ringing_overshoot").set_index(["mask", "d0"])["overshoot_pct_of_step"]
    classes = json.loads((T / "e2_classes.json").read_text(encoding="utf-8"))
    dm = read("decision_matrix").set_index("application")
    m = meta["aoi_cloud_metrics_pct"]
    v = {"report_date": date.today().strftime("%d %B %Y"),
         "n_candidates": meta["n_candidates"], "tile": meta["mgrs_tile"],
         "bad_min": f(cand["bad_pct"].min()), "bad_max": f(cand["bad_pct"].max(), 1),
         "aot_max": f(cand["aot_mean"].max(), 2),
         "product_id": meta["product_id"], "acq_date": meta["acquisition_date"],
         "acq_time": meta["acquisition_datetime_utc"][11:16], "baseline": meta["processing_baseline"],
         "bad_pct": f(m["bad_pct"]), "cloud_pct": f(m["cloud_pct"]), "cirrus_pct": f(m["cirrus_pct"]),
         "shadow_pct": f(m["shadow_pct"]), "nodata_pct": f(m["nodata_pct"]), "prob_pct": f(m["prob_pct"]),
         "aot": f(meta["aoi_aot_mean"], 3), "width": meta["width"], "height": meta["height"],
         "n_filters": len(e4)}

    # ringing
    v |= {"ring_ideal": f(ring[("ideal", 0.1)], 1), "ring_butter": f(ring[("butterworth", 0.1)], 1),
          "ring_gauss": f(ring[("gaussian", 0.1)], 1)}

    # E1
    for noise, key in [("gaussian", "g"), ("salt_pepper", "sp")]:
        d = e1[e1["noise"] == noise]
        noisy = d[d["family"] == "-"].iloc[0]
        lp = d[d["family"] != "-"].sort_values("psnr_db", ascending=False)
        v[f"e1_{key}_noisy"] = f(noisy["psnr_db"])
        v[f"e1_{key}_best_psnr"], v[f"e1_{key}_best_ssim"] = f(lp.iloc[0]["psnr_db"]), f(lp.iloc[0]["ssim"], 3)
        if key == "g":
            v |= {"e1_g_best": lp.iloc[0]["filter"], "e1_g_second": lp.iloc[1]["filter"],
                  "e1_g_second_psnr": f(lp.iloc[1]["psnr_db"]), "e1_g_n_lpf": len(lp),
                  "e1_g_n_worse": int((lp["psnr_db"] < noisy["psnr_db"]).sum())}
        else:
            lin = lp[~lp["filter"].str.startswith("median")].iloc[0]
            v |= {"e1_sp_lin": lin["filter"], "e1_sp_lin_psnr": f(lin["psnr_db"]),
                  "e1_sp_gap": f(lp.iloc[0]["psnr_db"] - lin["psnr_db"])}

    # E2
    raw = e2.set_index("input").loc["raw"]
    lpf = e2[e2["family"].str.startswith("LPF")].sort_values("kappa", ascending=False)
    hb = e2[e2["family"] == "HPF-enhanced"].sort_values("kappa", ascending=False)
    stack = e2[e2["family"] == "raw + HPF layers"].sort_values("kappa", ascending=False)
    n_better = int((lpf["kappa"] > raw["kappa"]).sum())
    v |= {"water_pct": f(100 * classes["dropped_classes_fraction"].get("Water", 0)),
          "n_classes": len(classes["classes"]),
          "e2_raw_oa": f(raw["overall_accuracy"], 3), "e2_raw_kappa": f(raw["kappa"], 3),
          "e2_raw_sp": f(raw["salt_pepper_index"], 3),
          "e2_lpf_phrase": ("Every low-pass input improved on that." if n_better == len(lpf)
                            else f"{n_better} of {len(lpf)} low-pass inputs improved on that."),
          "e2_best": lpf.iloc[0]["input"], "e2_best_oa": f(lpf.iloc[0]["overall_accuracy"], 3),
          "e2_best_kappa": f(lpf.iloc[0]["kappa"], 3), "e2_best_sp": f(lpf.iloc[0]["salt_pepper_index"], 3),
          "e2_second": lpf.iloc[1]["input"], "e2_second_kappa": f(lpf.iloc[1]["kappa"], 3),
          "e2_best_gap": f(lpf.iloc[0]["kappa"] - lpf.iloc[1]["kappa"], 3),
          "e2_kappa_gain": f(lpf.iloc[0]["kappa"] - raw["kappa"], 3),
          "e2_hb_best": hb.iloc[0]["input"], "e2_hb_best_kappa": f(hb.iloc[0]["kappa"], 3),
          "e2_hb_worst_kappa": f(hb.iloc[-1]["kappa"], 3),
          "e2_stack_best_kappa": f(stack.iloc[0]["kappa"], 3),
          "bare_pct": f(100 * classes["class_fraction"]["Bare/Mining"], 1),
          "bare_f1_max": f(e2["f1_Bare/Mining"].max(), 2)}

    # E3
    s3 = e3.set_index("filter")
    hb3 = e3[e3["family"] == "HPF-enhanced"].sort_values("boundary_f1", ascending=False).iloc[0]
    img3 = e3[(e3["output"] == "image") & (e3["filter"] != "original")]
    bu = img3.sort_values("builtup_boundary_f1", ascending=False).iloc[0]
    fft_hp = e3[e3["family"] == "HPF-freq"].sort_values("coherence_ridge", ascending=False).iloc[0]
    v |= {"e3_orig_f1": f(s3.loc["original", "boundary_f1"], 3), "e3_hb_best_f1": f(hb3["boundary_f1"], 3),
          "e3_hb_best": hb3["filter"], "e3_box5_f1": f(s3.loc["box_5", "boundary_f1"], 3),
          "e3_bu_best_f1": f(bu["builtup_boundary_f1"], 3), "e3_bu_best": bu["filter"],
          "e3_bu_orig": f(s3.loc["original", "builtup_boundary_f1"], 3),
          "e3_coh_orig": f(s3.loc["original", "coherence_ridge"], 3),
          "e3_coh_prewitt": f(s3.loc["prewitt", "coherence_ridge"], 3),
          "e3_coh_sobel": f(s3.loc["sobel", "coherence_ridge"], 3),
          "e3_coh_lap4": f(s3.loc["laplace_4", "coherence_ridge"], 3),
          "e3_coh_g3": f(s3.loc["gauss_s3", "coherence_ridge"], 3),
          "e3_coh_ideal": f(s3.loc["fft_ideal_lp_0.1", "coherence_ridge"], 3),
          "e3_coh_fft_best": f(fft_hp["coherence_ridge"], 3), "e3_coh_fft_best_name": fft_hp["filter"]}

    # E4
    s4 = e4.set_index("filter")
    hpf_sam = e4[e4["filter"].str.match(r"^(hpf_|fft_.*_hp_)")]["sam_deg"].median()
    v |= {"e4_med3_sam": f(s4.loc["median_3", "sam_deg"]), "e4_g1_sam": f(s4.loc["gauss_s1", "sam_deg"]),
          "e4_med3_ndvi": f(s4.loc["median_3", "ndvi_abs_change"], 3),
          "e4_g1_ndvi": f(s4.loc["gauss_s1", "ndvi_abs_change"], 3),
          "e4_hb05_sam": f(s4.loc["boost_n3_k0.5", "sam_deg"]), "e4_hpf_sam": f(hpf_sam, 0)}

    # E5
    sp = e5[e5["domain"] == "spatial"]
    fq = e5[e5["domain"] == "frequency"]
    box = sp[sp["method"] == "box mean"].set_index("kernel_side")["seconds"]
    gau = sp[sp["method"] == "gaussian"].set_index("kernel_side")["seconds"]
    lin = sp[sp["method"].isin(["box mean", "gaussian"])]
    slower = lin[lin["seconds"] > fq["seconds"].median()]["kernel_side"]
    v |= {"e5_shape": e5["image_shape"].iloc[0].replace("x", " x "), "e5_box3": f(box[3]), "e5_box31": f(box[31]),
          "e5_g8": f(gau.iloc[-1]), "e5_fft_min": f(fq["seconds"].min()), "e5_fft_max": f(fq["seconds"].max()),
          "e5_cross": int(slower.min()) if len(slower) else "n/a"}

    # decision matrix
    vis = dm.loc["Visual interpretation"]
    g0 = s3.loc["original", "avg_gradient"]
    v |= {"vis_best": vis["best_filter"],
          "vis_gain": f(s3.loc[vis["best_filter"], "avg_gradient"] / g0),
          "vis_sam": f(s4.loc[vis["best_filter"], "sam_deg"])}
    return v


def readme_block(v: dict) -> str:
    """Markdown results summary written between the RESULTS markers in README.md."""
    dm = read("decision_matrix")
    rows = "\n".join(f"| {r.application} | `{r.best_filter}` | {r.key_metric} {num(r.key_metric_value)} "
                     f"(no filter {num(r.no_filter_value)}) | `{r.runner_up}` |" for r in dm.itertuples())
    return f"""## Results at a glance

Scene: **{v['product_id']}** ({v['acq_date']}), chosen from {v['n_candidates']} candidates; {v['bad_pct']} % of AOI pixels flagged as cloud, shadow or no data, mean AOT {v['aot']}.

| Application | Best filter | Evidence | Runner-up |
|---|---|---|---|
{rows}

Smoothing raised Random Forest kappa from {v['e2_raw_kappa']} to {v['e2_best_kappa']} and cut map speckle from {v['e2_raw_sp']} to {v['e2_best_sp']}. High-boost lowered kappa, but it gave the sharpest image and the best agreement between Canny edges and mapped boundaries. So low-pass filtering is the better choice when per-pixel values feed a calculation, and high-pass filtering when the analyst cares about edges and structure.

On salt-and-pepper noise a 3 x 3 median reached {v['e1_sp_best_psnr']} dB, {v['e1_sp_gap']} dB ahead of the best linear filter. On a clean step edge the ideal frequency-domain LPF overshoots by {v['ring_ideal']} %, Butterworth by {v['ring_butter']} % and Gaussian by {v['ring_gauss']} %. Every frequency-domain filter took {v['e5_fft_min']}-{v['e5_fft_max']} s on a full band whatever its cutoff, while spatial convolution was faster only for kernels below about {v['e5_cross']} pixels.

All values come from `results/tables/`. Report: [`report/Group1_EES403_Report.pdf`](report/Group1_EES403_Report.pdf); slides: [`presentation/Group1_EES403_Slides.pdf`](presentation/Group1_EES403_Slides.pdf).

| | |
|---|---|
| ![LPF](results/figures/fig02a_lpf_grid.png) | ![High-boost](results/figures/fig02c_highboost_grid.png) |
| ![Ringing](results/figures/fig05_ringing.png) | ![Classification](results/figures/fig07_classification_maps.png) |
"""


def pretty(code: str) -> str:
    """Readable LaTeX name for a filter code used in the CSV tables."""
    mask = {"ideal": "Ideal", "butter": "Butterworth", "gauss": "Gaussian"}
    rules = [
        (r"box_(\d+)", lambda m: rf"Mean ${m[1]}\times{m[1]}$"),
        (r"gauss_s(\d+)", lambda m: rf"Gaussian $\sigma={m[1]}$"),
        (r"median_(\d+)", lambda m: rf"Median ${m[1]}\times{m[1]}$"),
        (r"hpf_(\d+)", lambda m: rf"HPF ${m[1]}\times{m[1]}$"),
        (r"raw\+hpf_(\d+)", lambda m: rf"Raw + HPF ${m[1]}\times{m[1]}$ layers"),
        (r"laplace_(\d)", lambda m: rf"Laplacian ({m[1]}-neighbour)"),
        (r"sobel", lambda m: "Sobel"),
        (r"prewitt", lambda m: "Prewitt"),
        (r"boost_n(\d+)_k([\d.]+)", lambda m: rf"High-boost $n={m[1]}$, $k={m[2]}$"),
        (r"fft_(\w+?)_(lp|hp)_([\d.]+)", lambda m: rf"{mask[m[1]]} {m[2].upper()}F $D_0={m[3]}$"),
        (r"raw|original", lambda m: "No filter"),
    ]
    for pat, fn in rules:
        m = re.fullmatch(pat, code)
        if m:
            return fn(m)
    return tex(code)


def tex(s) -> str:
    """Escape LaTeX special characters in plain text values."""
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(a, b)
    return s


def tabular_rows(rows) -> str:
    """Join table rows (lists of cell strings) into LaTeX body lines."""
    return "\n".join(" & ".join(r) + r" \\" for r in rows)


def bold_max(values_: list, fmt, higher=True) -> list:
    """Format a column and bold the best value (ignoring missing ones)."""
    present = [v for v in values_ if pd.notna(v)]
    best = (max if higher else min)(present) if present else None
    return [("--" if pd.isna(v) else (rf"\textbf{{{fmt(v)}}}" if v == best else fmt(v))) for v in values_]


def e1_table() -> str:
    e1 = read("e1_noise")
    e1["filter"] = e1["filter"].replace({"none (noisy input)": "raw"})
    keep = ["raw", "box_3", "box_5", "gauss_s1", "gauss_s2", "median_3", "median_5",
            "fft_ideal_lp_0.3", "fft_butter_lp_0.3", "fft_gauss_lp_0.3"]
    cols = {}
    for noise in ["gaussian", "salt_pepper"]:
        d = e1[e1["noise"] == noise].set_index("filter").loc[keep]
        cols[noise + "_psnr"] = bold_max(list(d["psnr_db"]), lambda v: f"{v:.2f}")
        cols[noise + "_ssim"] = bold_max(list(d["ssim"]), lambda v: f"{v:.3f}")
    rows = []
    for i, k in enumerate(keep):
        name = "Noisy input (no filter)" if k == "raw" else pretty(k)
        rows.append([name] + [cols[c][i] for c in cols])
    return tabular_rows(rows)


def e2_table() -> str:
    e2 = read("e2_classification")
    pick = ["raw"]
    for fam in ["LPF-spatial", "LPF-freq", "HPF-enhanced", "raw + HPF layers"]:
        d = e2[e2["family"] == fam].sort_values("kappa", ascending=False)
        pick += [d.iloc[0]["input"]] + ([d.iloc[-1]["input"]] if fam == "HPF-enhanced" else [])
    d = e2.set_index("input").loc[pick]
    fam_name = {"none": "none", "LPF-spatial": "LPF, spatial", "LPF-freq": "LPF, frequency",
                "HPF-enhanced": "high-boost", "raw + HPF layers": "raw + HPF stack"}
    oa = bold_max(list(d["overall_accuracy"]), lambda v: f"{v:.3f}")
    ka = bold_max(list(d["kappa"]), lambda v: f"{v:.3f}")
    f1 = bold_max(list(d["macro_f1"]), lambda v: f"{v:.3f}")
    sp = bold_max(list(d["salt_pepper_index"]), lambda v: f"{v:.3f}", higher=False)
    rows = [[pretty(k), fam_name.get(d.loc[k, "family"], tex(d.loc[k, "family"])), oa[i], ka[i], f1[i], sp[i]]
            for i, k in enumerate(pick)]
    return tabular_rows(rows)


def e3e4_table() -> str:
    e3 = read("e3_edges").set_index("filter")
    e4 = read("e4_fidelity").set_index("filter")
    keep = ["original", "gauss_s1", "median_3", "box_5", "fft_gauss_lp_0.2", "hpf_7", "laplace_4", "sobel",
            "prewitt", "fft_gauss_hp_0.1", "boost_n3_k0.5", "boost_n3_k1.0", "boost_n5_k1.5"]
    d = e3.loc[keep]
    sam = [e4.loc[k, "sam_deg"] if k in e4.index else float("nan") for k in keep]
    ndvi = [e4.loc[k, "ndvi_abs_change"] if k in e4.index else float("nan") for k in keep]
    cols = [bold_max(list(d["avg_gradient"]), lambda v: f"{v:.4f}"),
            bold_max(list(d["boundary_f1"]), lambda v: f"{v:.3f}"),
            bold_max(list(d["builtup_boundary_f1"]), lambda v: f"{v:.3f}"),
            bold_max(list(d["coherence_ridge"]), lambda v: f"{v:.3f}"),
            ["--" if pd.isna(v) else f"{v:.2f}" for v in sam],
            ["--" if pd.isna(v) else f"{v:.3f}" for v in ndvi]]
    rows = [[pretty(k)] + [c[i] for c in cols] for i, k in enumerate(keep)]
    # rule between the low-pass, high-pass and high-boost groups
    body = tabular_rows(rows).split("\n")
    for at in (10, 5, 1):
        body.insert(at, r"\midrule")
    return "\n".join(body)


def decision_rows() -> list:
    dm = read("decision_matrix")
    e4 = read("e4_fidelity").set_index("filter")
    short = {"Noise removal (Gaussian noise)": "Noise removal, random noise",
             "Noise removal (salt-and-pepper noise)": "Noise removal, impulse noise",
             "Lineament / structural mapping (Aravalli ridges)": "Lineament mapping (ridges)"}
    metric = {"PSNR (dB)": "PSNR (dB)", "Cohen's kappa": r"Cohen's $\kappa$",
              "orientation coherence on the ridge": "ridge coherence",
              "built-up boundary F1": "built-up edge F1", "NIR average gradient": "NIR mean gradient"}

    def num(v):
        return f"{v:.2f}" if abs(v) >= 10 else (f"{v:.3f}" if abs(v) >= 0.1 else f"{v:.4f}")

    rows = []
    for r in dm.itertuples():
        rows.append(dict(app=short.get(r.application, r.application), best=pretty(r.best_filter),
                         metric=metric.get(r.key_metric, tex(r.key_metric)), val=num(r.key_metric_value),
                         none=num(r.no_filter_value), runner=pretty(r.runner_up), runner_val=num(r.runner_up_value),
                         sam=f"{e4.loc[r.best_filter, 'sam_deg']:.2f}"))
    return rows


def decision_table_report() -> str:
    return tabular_rows([[r["app"], r["best"], r["metric"], r["val"], r["none"],
                          f"{r['runner']} ({r['runner_val']})", r["sam"]] for r in decision_rows()])


def decision_table_slides() -> str:
    return tabular_rows([[r["app"], r["best"], f"{r['metric']}: \\textbf{{{r['val']}}} vs {r['none']}"]
                         for r in decision_rows()])


def all_values(raw: dict) -> dict:
    """Escaped text values for the templates, plus the LaTeX tables and a few extra numbers."""
    v = {k: (val if isinstance(val, (int, float)) else tex(val)) for k, val in raw.items()}
    v["e5_shape"] = v["e5_shape"].replace(" x ", r"$\times$")
    e2 = read("e2_classification").set_index("input")
    e3 = read("e3_edges").set_index("filter")
    e4 = read("e4_fidelity").set_index("filter")
    classes = json.loads((T / "e2_classes.json").read_text(encoding="utf-8"))
    v |= {"e1_table": e1_table(), "e2_table": e2_table(), "e3e4_table": e3e4_table(),
          "decision_table_report": decision_table_report(), "decision_table_slides": decision_table_slides(),
          "e2_best_mf1": f"{e2.loc[raw['e2_best'], 'macro_f1']:.3f}",
          "e2_raw_mf1": f"{e2.loc['raw', 'macro_f1']:.3f}",
          "e3_g1_f1": f"{e3.loc['gauss_s1', 'boundary_f1']:.3f}",
          "e3_coh_hpf7": f"{e3.loc['hpf_7', 'coherence_ridge']:.3f}",
          "e4_sobel_sam": f"{e4.loc['sobel', 'sam_deg']:.2f}",
          "e4_hpf7_sam": f"{e4.loc['hpf_7', 'sam_deg']:.2f}",
          "n_test_pix": f"{int(e2.loc['raw', 'n_test']):,}".replace(",", r"\,"),
          "n_train_pix": f"{int(e2.loc['raw', 'n_train']):,}".replace(",", r"\,"),
          **{f"cls_{c.split('/')[0].split('-')[0].lower()}": f"{100 * f:.1f}"
             for c, f in classes["class_fraction"].items()}}
    return v


def fill(template: Path, v: dict, raw: dict) -> str:
    text = template.read_text(encoding="utf-8")
    keys = set(re.findall(r"<<(?:p:)?(\w+)>>", text))
    missing = keys - set(v)
    if missing:
        raise KeyError(f"{template.name}: placeholders without a value: {sorted(missing)}")
    text = re.sub(r"<<p:(\w+)>>", lambda m: pretty(str(raw[m[1]])), text)
    return re.sub(r"<<(\w+)>>", lambda m: str(v[m[1]]), text)


def latexmk(folder: Path, name: str) -> Path:
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-quiet", name + ".tex"],
                   cwd=folder, check=True, stdout=subprocess.DEVNULL)
    return folder / f"{name}.pdf"


def main():
    raw = values()
    sp = read("e1_noise").query("noise == 'salt_pepper' and family != '-'")
    raw["e1_sp_best"] = sp.sort_values("psnr_db", ascending=False).iloc[0]["filter"]
    v = all_values(raw)

    # flowchart: one TikZ source for the report figure and docs/flowchart.pdf/.png
    (REPORT_DIR / "flowchart.tex").write_text(fill(REPORT_DIR / "flowchart_template.tex", v, raw), encoding="utf-8")
    chart = latexmk(REPORT_DIR, "flowchart")
    shutil.copy(chart, ROOT / "docs" / "flowchart.pdf")
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", str(chart), str(ROOT / "docs" / "flowchart")],
                       check=True)
    else:
        print("pdftoppm not found: docs/flowchart.png not refreshed (docs/flowchart.pdf is)")
    print("wrote docs/flowchart.pdf")

    report = fill(REPORT_DIR / "report_template.tex", v, raw)
    (REPORT_DIR / "report.tex").write_text(report, encoding="utf-8")
    shutil.copy(latexmk(REPORT_DIR, "report"), ROOT / "report" / "Group1_EES403_Report.pdf")
    print("wrote report/Group1_EES403_Report.pdf")

    slides = fill(SLIDES_DIR / "slides_template.tex", v, raw)
    for name, notes in [("slides", r"\setbeameroption{hide notes}"),
                        ("slides_notes", r"\setbeameroption{show notes on second screen=bottom}")]:
        (SLIDES_DIR / f"{name}.tex").write_text(slides.replace("%NOTES-OPTION%", notes), encoding="utf-8")
        out = ROOT / "presentation" / ("Group1_EES403_Slides" + ("_notes" if "notes" in name else "") + ".pdf")
        shutil.copy(latexmk(SLIDES_DIR, name), out)
        print("wrote", out.relative_to(ROOT))

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                  lambda _: f"<!-- RESULTS:START -->\n{readme_block(raw)}<!-- RESULTS:END -->", text, flags=re.S)
    readme.write_text(text, encoding="utf-8")
    print("updated README results block")


if __name__ == "__main__":
    main()
