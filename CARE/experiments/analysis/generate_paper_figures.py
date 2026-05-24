from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
SRC_DIR = FIG_DIR / "src"
BUILD_DIR = FIG_DIR / "build"


def load_json(rel_path: str):
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)


def mean_metric(dataset: str, bucket: str, method: str, metric: str) -> float:
    data = load_json(f"results/{dataset}_care_results.json")
    return mean(seed["gate_strat"][bucket][method][metric] for seed in data["per_seed"])


def mean_top(dataset: str, key: str, metric: str = "ndcg@10") -> float:
    data = load_json(f"results/{dataset}_care_results.json")
    return mean(seed[key][metric] for seed in data["per_seed"])


def parse_numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text)]


def sanitize_label(text: str) -> str:
    return (
        text.replace("μ", "mu")
        .replace("α", "alpha")
        .replace("β", "beta")
        .replace("=", "=")
        .replace(" ", "")
    )


def coord_str(labels: list[str], values: list[float]) -> str:
    return " ".join(f"({label},{value:.4f})" for label, value in zip(labels, values))


def doc(body: str, extra_preamble: str = "") -> str:
    return (
        "\\documentclass[tikz,border=2pt]{standalone}\n"
        "\\usepackage{times}\n"
        "\\usepackage{pgfplots}\n"
        "\\usepgfplotslibrary{groupplots,colormaps}\n"
        "\\pgfplotsset{compat=1.18}\n"
        "\\definecolor{careblue}{HTML}{46B5E5}\n"
        "\\definecolor{careteal}{HTML}{4CB963}\n"
        "\\definecolor{careorange}{HTML}{FF6B6B}\n"
        "\\definecolor{caregold}{HTML}{F4E36A}\n"
        "\\definecolor{carepurple}{HTML}{E07A8B}\n"
        "\\definecolor{caregray}{HTML}{556270}\n"
        "\\definecolor{gridgray}{HTML}{7A7A7A}\n"
        "\\definecolor{textgray}{HTML}{111111}\n"
        "\\pgfplotsset{\n"
        "tkdeAxis/.style={\n"
        "font=\\fontsize{8.6}{10}\\selectfont,\n"
        "tick label style={font=\\fontsize{8}{9}\\selectfont},\n"
        "label style={font=\\fontsize{8.6}{10}\\selectfont},\n"
        "title style={font=\\fontsize{8.8}{10}\\selectfont},\n"
        "legend style={font=\\fontsize{7.8}{9}\\selectfont, draw=none, fill=none, /tikz/every even column/.append style={column sep=0.35cm}},\n"
        "axis line style={draw=black, line width=0.75pt},\n"
        "axis lines=box,\n"
        "tick style={draw=black, line width=0.55pt},\n"
        "x tick label style={text=textgray},\n"
        "y tick label style={text=textgray},\n"
        "ylabel style={text=textgray},\n"
        "xlabel style={text=textgray},\n"
        "title style={text=textgray},\n"
        "grid=major,\n"
        "grid style={draw=gridgray, densely dotted, line width=0.35pt},\n"
        "axis background/.style={fill=white},\n"
        "minor tick num=0,\n"
        "tick align=outside,\n"
        "major tick length=1.8pt,\n"
        "every axis plot/.append style={line join=round},\n"
        "clip=false\n"
        "},\n"
        "tkdeBar/.style={ybar, bar width=6.8pt},\n"
        "tkdeLine/.style={line width=1.15pt, mark size=1.9pt},\n"
        "tkdeSurf/.style={view={38}{24}, colormap/jet, shader=flat, colorbar, mesh/ordering=y varies}\n"
        "}\n"
        f"{extra_preamble}\n"
        "\\begin{document}\n"
        f"{body}\n"
        "\\end{document}\n"
    )


def write_and_compile(name: str, tex_content: str) -> None:
    tex_path = SRC_DIR / f"{name}.tex"
    tex_path.write_text(tex_content, encoding="utf-8")
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={BUILD_DIR}",
            str(tex_path),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    pdf_src = BUILD_DIR / f"{name}.pdf"
    pdf_dst = FIG_DIR / f"{name}.pdf"
    shutil.copyfile(pdf_src, pdf_dst)


def coldness_spectrum(dataset: str) -> tuple[str, str]:
    bucket_labels = ["L0", "L1", "L2", "L3"]
    bucket_keys = ["0 (zero-shot)", "1-4 (cold)", "5-20 (warm)", ">20 (hot)"]
    cf = [mean_metric(dataset, bucket, "cf_only", "ndcg@10") for bucket in bucket_keys]
    text = [mean_metric(dataset, bucket, "text_only", "ndcg@10") for bucket in bucket_keys]
    care = [mean_metric(dataset, bucket, "gate", "ndcg@10") for bucket in bucket_keys]
    ymax = max(cf + text + care) * 1.10
    body = f"""
\\begin{{tikzpicture}}
\\begin{{axis}}[
tkdeAxis,
tkdeBar,
width=2.18in,
height=1.62in,
ymin=0,
ymax={ymax:.3f},
symbolic x coords={{L0,L1,L2,L3}},
xtick=data,
xticklabel style={{font=\\fontsize{{7.6}}{{8.5}}\\selectfont}},
ylabel={{NDCG@10}},
enlarge x limits=0.18,
legend columns=3,
legend to name=legend:{dataset},
legend style={{/tikz/every even column/.append style={{column sep=0.22cm}}}}
]
\\addplot+[tkdeBar, fill=careblue, draw=careblue] coordinates {{{coord_str(bucket_labels, cf)}}};
\\addplot+[tkdeBar, fill=careorange, draw=careorange] coordinates {{{coord_str(bucket_labels, text)}}};
\\addplot+[tkdeBar, fill=careteal, draw=careteal] coordinates {{{coord_str(bucket_labels, care)}}};
\\legend{{CF, Text, CARE}}
\\end{{axis}}
\\end{{tikzpicture}}
"""
    return f"coldness_spectrum_{dataset}", doc(body)


def fixed_fusion_tradeoff() -> tuple[str, str]:
    beta_labels = ["0.0", "0.5", "0.7", "1.0"]
    datasets = ["baby", "office", "sports"]
    colors = {"baby": "careteal", "office": "careblue", "sports": "careorange"}
    lines = []
    for title, prefix in [("Text channel", ""), ("Image channel", "cold_img_")]:
        plots = []
        for ds in datasets:
            values = [
                mean_top(ds, "cold_cf"),
                mean_top(ds, f"{prefix}beta05" if prefix else "cold_beta05"),
                mean_top(ds, f"{prefix}beta07" if prefix else "cold_beta07"),
                mean_top(ds, f"{prefix}beta10" if prefix else "cold_beta10"),
            ]
            plots.append(
                f"\\addplot+[tkdeLine, color={colors[ds]}, mark=*] coordinates {{{coord_str(beta_labels, values)}}};"
            )
        lines.append((title, "\n".join(plots)))
    body = f"""
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
group style={{group size=2 by 1, horizontal sep=0.72in}},
tkdeAxis,
width=3.18in,
height=2.10in,
ymin=0,
ymax=0.24,
symbolic x coords={{0.0,0.5,0.7,1.0}},
xtick=data,
xlabel={{Semantic weight beta}},
ylabel={{NDCG@10}},
legend columns=3,
legend to name=legend:fixedfusion
]
\\nextgroupplot[title={{Text channel}}]
{lines[0][1]}
\\nextgroupplot[title={{Image channel}}]
{lines[1][1]}
\\legend{{Baby, Office, Sports}}
\\end{{groupplot}}
\\end{{tikzpicture}}
"""
    return "fixed_fusion_tradeoff", doc(body)


def zero_shot_modality_profile() -> tuple[str, str]:
    datasets = ["baby", "office", "sports"]
    metrics = ["hr@5", "ndcg@5", "hr@10", "ndcg@10", "hr@20", "ndcg@20"]
    labels = ["T", "I", "C+T", "C+I"]
    method_keys = ["text_only", "image_only", "gate", "gate_image"]
    colors = ["careorange", "careblue", "careteal", "carepurple"]
    groupplots = []
    for metric in metrics:
        raw = [mean(mean_metric(ds, "0 (zero-shot)", key, metric) for ds in datasets) for key in method_keys]
        m = max(raw)
        values = [v / m if m > 0 else 0.0 for v in raw]
        plots = "\n".join(
            f"\\addplot+[tkdeBar, fill={color}, draw={color}] coordinates {{({label},{value:.4f})}};"
            for label, value, color in zip(labels, values, colors)
        )
        groupplots.append(
            f"\\nextgroupplot[title={{{metric.upper()}}}, ymin=0, ymax=1.08, symbolic x coords={{T,I,C+T,C+I}}, xtick=data, xticklabel style={{font=\\fontsize{{7}}{{8}}\\selectfont}}, enlarge x limits=0.28]\n{plots}"
        )
    body = f"""
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
group style={{group size=3 by 2, horizontal sep=0.30in, vertical sep=0.34in}},
tkdeAxis,
tkdeBar,
width=1.02in,
height=0.94in,
ylabel={{Norm.}},
legend columns=4,
legend to name=legend:modalityprofile
]
{chr(10).join(groupplots)}
\\legend{{Text, Image, CARE+Text, CARE+Image}}
\\end{{groupplot}}
\\end{{tikzpicture}}
"""
    return "zero_shot_modality_radar", doc(body)


def modality_delta_bars() -> tuple[str, str]:
    datasets = ["Baby", "Office", "Sports"]
    keys = ["baby", "office", "sports"]
    text_delta = [mean_metric(ds, "0 (zero-shot)", "gate", "ndcg@10") - mean_metric(ds, "0 (zero-shot)", "text_only", "ndcg@10") for ds in keys]
    image_delta = [mean_metric(ds, "0 (zero-shot)", "gate_image", "ndcg@10") - mean_metric(ds, "0 (zero-shot)", "image_only", "ndcg@10") for ds in keys]
    ymax = max(text_delta + image_delta) * 1.14
    body = f"""
\\begin{{tikzpicture}}
\\begin{{axis}}[
tkdeAxis,
tkdeBar,
width=3.20in,
height=2.12in,
ymin=0,
ymax={ymax:.3f},
symbolic x coords={{Baby,Office,Sports}},
xtick=data,
ylabel={{Delta NDCG@10}},
enlarge x limits=0.24,
legend columns=2,
legend style={{at={{(0.5,1.04)}}, anchor=south, draw=none}}
]
\\addplot+[tkdeBar, fill=careteal, draw=careteal] coordinates {{{coord_str(datasets, text_delta)}}};
\\addplot+[tkdeBar, fill=carepurple, draw=carepurple] coordinates {{{coord_str(datasets, image_delta)}}};
\\legend{{Text to CARE+Text, Image to CARE+Image}}
\\end{{axis}}
\\end{{tikzpicture}}
"""
    return "modality_delta_bars", doc(body)


def aggregate_vs_stratified_gap() -> tuple[str, str]:
    baseline_files = {
        "baby": load_json("experiments/baselines/results/baby_baselines.json")["results"],
        "office": load_json("experiments/baselines/results/office_baselines.json")["results"],
        "sports": load_json("experiments/baselines/results/sports_baselines.json")["results"],
    }
    datasets = ["Baby", "Office", "Sports"]
    keys = ["baby", "office", "sports"]
    full_cf = [mean_top(ds, "full") for ds in keys]
    full_bm3 = [baseline_files[ds]["BM3"]["test"]["ndcg@10"] for ds in keys]
    full_care = [mean_top(ds, "gate_full") for ds in keys]
    l0_cf = [mean_metric(ds, "0 (zero-shot)", "cf_only", "ndcg@10") for ds in keys]
    l0_text = [mean_metric(ds, "0 (zero-shot)", "text_only", "ndcg@10") for ds in keys]
    l0_care = [mean_metric(ds, "0 (zero-shot)", "gate", "ndcg@10") for ds in keys]
    body = f"""
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
group style={{group size=2 by 1, horizontal sep=0.78in}},
tkdeAxis,
tkdeBar,
width=3.06in,
height=2.10in,
symbolic x coords={{Baby,Office,Sports}},
xtick=data,
enlarge x limits=0.20,
ylabel={{NDCG@10}}
]
\\nextgroupplot[title={{Full test set}}, ymin=0, ymax=0.42, legend columns=3, legend style={{at={{(0.5,1.04)}}, anchor=south}}]
\\addplot+[tkdeBar, fill=careblue, draw=careblue] coordinates {{{coord_str(datasets, full_cf)}}};
\\addplot+[tkdeBar, fill=caregold, draw=caregold] coordinates {{{coord_str(datasets, full_bm3)}}};
\\addplot+[tkdeBar, fill=careteal, draw=careteal] coordinates {{{coord_str(datasets, full_care)}}};
\\legend{{CF backbone, BM3, CARE}}
\\nextgroupplot[title={{Zero-shot bucket L0}}, ymin=0, ymax=0.80, legend columns=3, legend style={{at={{(0.5,1.04)}}, anchor=south}}]
\\addplot+[tkdeBar, fill=careblue, draw=careblue] coordinates {{{coord_str(datasets, l0_cf)}}};
\\addplot+[tkdeBar, fill=careorange, draw=careorange] coordinates {{{coord_str(datasets, l0_text)}}};
\\addplot+[tkdeBar, fill=careteal, draw=careteal] coordinates {{{coord_str(datasets, l0_care)}}};
\\legend{{CF, Text, CARE}}
\\end{{groupplot}}
\\end{{tikzpicture}}
"""
    return "aggregate_vs_stratified_gap", doc(body)


def care_vs_mamex_zero_shot() -> tuple[str, str]:
    care_baby = load_json("results/baby_care_results.json")["per_seed"][0]
    care_office = load_json("results/office_care_results.json")["per_seed"][0]
    mamex_baby = load_json("experiments/baselines/results/baby_mamex.json")[0]
    mamex_office = load_json("experiments/baselines/results/office_mamex.json")[0]
    labels = ["Full CF", "L0 Text", "L0 Gate"]
    baby_care = [care_baby["full"]["ndcg@10"], care_baby["gate_strat"]["0 (zero-shot)"]["text_only"]["ndcg@10"], care_baby["gate_strat"]["0 (zero-shot)"]["gate"]["ndcg@10"]]
    baby_mamex = [mamex_baby["full_cf"]["ndcg@10"], mamex_baby["stratified"]["0 (zero-shot)"]["text_only"]["ndcg@10"], mamex_baby["stratified"]["0 (zero-shot)"]["gate"]["ndcg@10"]]
    office_care = [care_office["full"]["ndcg@10"], care_office["gate_strat"]["0 (zero-shot)"]["text_only"]["ndcg@10"], care_office["gate_strat"]["0 (zero-shot)"]["gate"]["ndcg@10"]]
    office_mamex = [mamex_office["full_cf"]["ndcg@10"], mamex_office["stratified"]["0 (zero-shot)"]["text_only"]["ndcg@10"], mamex_office["stratified"]["0 (zero-shot)"]["gate"]["ndcg@10"]]
    body = f"""
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
group style={{group size=2 by 1, horizontal sep=0.78in}},
tkdeAxis,
tkdeBar,
width=3.00in,
height=2.10in,
symbolic x coords={{FullCF,L0Text,L0Gate}},
xtick=data,
xticklabels={{Full CF,L0 Text,L0 Gate}},
xticklabel style={{font=\\fontsize{{7}}{{8}}\\selectfont}},
enlarge x limits=0.22,
ylabel={{NDCG@10}},
legend columns=2,
legend style={{at={{(0.5,1.04)}}, anchor=south}}
]
\\nextgroupplot[title={{Baby}}, ymin=0, ymax=0.46]
\\addplot+[tkdeBar, fill=careteal, draw=careteal] coordinates {{{coord_str(['FullCF','L0Text','L0Gate'], baby_care)}}};
\\addplot+[tkdeBar, fill=careorange, draw=careorange] coordinates {{{coord_str(['FullCF','L0Text','L0Gate'], baby_mamex)}}};
\\legend{{CARE, MAMEX}}
\\nextgroupplot[title={{Office}}, ymin=0, ymax=0.60]
\\addplot+[tkdeBar, fill=careteal, draw=careteal] coordinates {{{coord_str(['FullCF','L0Text','L0Gate'], office_care)}}};
\\addplot+[tkdeBar, fill=careorange, draw=careorange] coordinates {{{coord_str(['FullCF','L0Text','L0Gate'], office_mamex)}}};
\\end{{groupplot}}
\\end{{tikzpicture}}
"""
    return "care_vs_mamex_zero_shot", doc(body)


def gate_family_sweep() -> tuple[str, str]:
    results = load_json("experiments/ablations/results/baby_gate_functions.json")["results"]
    functions = results["function"]
    params = results["params"]
    ndcg10 = results["ndcg10"]
    hr10 = results["hr10"]
    inverse, exponential = [], []
    sigmoid_ndcg, sigmoid_hr = [], []
    refs = {}
    for fn_name, param, score_ndcg, score_hr in zip(functions, params, ndcg10, hr10):
        nums = parse_numbers(param)
        if fn_name == "Inverse":
            inverse.append((nums[0], score_ndcg))
        elif fn_name == "Exponential":
            exponential.append((nums[0], score_ndcg))
        elif fn_name == "Sigmoid":
            mu, temp = nums[0], nums[1]
            sigmoid_ndcg.append((temp, mu, score_ndcg))
            sigmoid_hr.append((temp, mu, score_hr))
        else:
            refs[fn_name] = score_ndcg
    inverse.sort()
    exponential.sort()
    ndcg_coords = " ".join(f"({t:g},{m:g},{v:.6f})" for t, m, v in sorted(sigmoid_ndcg, key=lambda x: (x[0], x[1])))
    hr_coords = " ".join(f"({t:g},{m:g},{v:.6f})" for t, m, v in sorted(sigmoid_hr, key=lambda x: (x[0], x[1])))
    inv_line = " ".join(f"({x:.2f},{y:.4f})" for x, y in inverse)
    exp_line = " ".join(f"({x:.2f},{y:.4f})" for x, y in exponential)
    body = f"""
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
group style={{group size=2 by 1, horizontal sep=0.72in}},
width=3.06in,
height=2.18in
]
\\nextgroupplot[
tkdeAxis,
tkdeSurf,
title={{Sigmoid landscape: NDCG@10}},
xlabel={{Diffusion steps T}},
ylabel={{Inference scale mu}},
zlabel={{NDCG@10}},
xmin=0.5, xmax=5.0,
ymin=0.5, ymax=20,
point meta min=0.249,
point meta max=0.354,
mesh/cols=5
]
\\addplot3[surf] coordinates {{{ndcg_coords}}};
\\nextgroupplot[
tkdeAxis,
tkdeSurf,
title={{Sigmoid landscape: HR@10}},
xlabel={{Diffusion steps T}},
ylabel={{Inference scale mu}},
zlabel={{HR@10}},
xmin=0.5, xmax=5.0,
ymin=0.5, ymax=20,
point meta min=0.46,
point meta max=0.60,
mesh/cols=5
]
\\addplot3[surf] coordinates {{{hr_coords}}};
\\end{{groupplot}}
\\node[anchor=north west, font=\\fontsize{{6.9}}{{8.4}}\\selectfont, text width=6.5in, text=textgray] at (0,-0.78) {{Reference curves not shown here remain tightly clustered: inverse peaks at {max(v for _, v in inverse):.4f}, exponential at {max(v for _, v in exponential):.4f}, CF-only at {refs['CF only']:.4f}, and the hard piecewise rule drops to {refs['Piecewise']:.4f}.}};
\\end{{tikzpicture}}
"""
    return "gate_family_sweep", doc(body)


def yelp_cross_domain_dashboard() -> tuple[str, str]:
    seed = load_json("results/yelp_care_results.json")["per_seed"][0]
    labels = ["1--4", ">20"]
    cf = [seed["gate_strat"]["1-4 (cold)"]["cf_only"]["ndcg@10"], seed["gate_strat"][">20 (hot)"]["cf_only"]["ndcg@10"]]
    txt = [seed["gate_strat"]["1-4 (cold)"]["text_only"]["ndcg@10"], seed["gate_strat"][">20 (hot)"]["text_only"]["ndcg@10"]]
    care = [seed["gate_strat"]["1-4 (cold)"]["gate"]["ndcg@10"], seed["gate_strat"][">20 (hot)"]["gate"]["ndcg@10"]]
    full_labels = ["CF", "CARE"]
    full_vals = [seed["full"]["ndcg@10"], seed["gate_full"]["ndcg@10"]]
    body = f"""
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
group style={{group size=2 by 1, horizontal sep=0.86in}},
tkdeAxis,
width=3.00in,
height=2.08in
]
\\nextgroupplot[
title={{Cold and hot strata}},
tkdeBar,
ymin=0,
ymax=1.05,
symbolic x coords={{1--4,>20}},
xtick=data,
ylabel={{NDCG@10}},
legend columns=3,
legend style={{at={{(0.5,1.04)}}, anchor=south}}
]
\\addplot+[tkdeBar, fill=careblue, draw=careblue] coordinates {{{coord_str(labels, cf)}}};
\\addplot+[tkdeBar, fill=careorange, draw=careorange] coordinates {{{coord_str(labels, txt)}}};
\\addplot+[tkdeBar, fill=careteal, draw=careteal] coordinates {{{coord_str(labels, care)}}};
\\legend{{CF, Text, CARE}}
\\nextgroupplot[
title={{Full test set}},
tkdeBar,
ymin=0,
ymax=0.34,
symbolic x coords={{CF,CARE}},
xtick=data,
ylabel={{NDCG@10}}
]
\\addplot+[tkdeBar, fill=careblue, draw=careblue] coordinates {{{coord_str(full_labels, [full_vals[0]])}}};
\\addplot+[tkdeBar, fill=careteal, draw=careteal] coordinates {{{coord_str(['CARE'], [full_vals[1]])}}};
\\end{{groupplot}}
\\end{{tikzpicture}}
"""
    return "yelp_cross_domain_dashboard", doc(body)


def full_ranking_consistency() -> tuple[str, str]:
    sampled = load_json("results/baby_care_results.json")["per_seed"][0]
    full = load_json("results/full_rank/baby_seed42_fullrank.json")["result"]
    labels = ["L0", "L3"]
    sampled_cf = [sampled["gate_strat"]["0 (zero-shot)"]["cf_only"]["ndcg@10"], sampled["gate_strat"][">20 (hot)"]["cf_only"]["ndcg@10"]]
    sampled_text = [sampled["gate_strat"]["0 (zero-shot)"]["text_only"]["ndcg@10"], sampled["gate_strat"][">20 (hot)"]["text_only"]["ndcg@10"]]
    sampled_care = [sampled["gate_strat"]["0 (zero-shot)"]["gate"]["ndcg@10"], sampled["gate_strat"][">20 (hot)"]["gate"]["ndcg@10"]]
    full_cf = [full["stratified"]["0 (zero-shot)"]["cf_only"]["ndcg@10"], full["stratified"][">20 (hot)"]["cf_only"]["ndcg@10"]]
    full_text = [full["stratified"]["0 (zero-shot)"]["text_only"]["ndcg@10"], full["stratified"][">20 (hot)"]["text_only"]["ndcg@10"]]
    full_care = [full["stratified"]["0 (zero-shot)"]["gate"]["ndcg@10"], full["stratified"][">20 (hot)"]["gate"]["ndcg@10"]]
    body = f"""
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
group style={{group size=2 by 1, horizontal sep=0.86in}},
tkdeAxis,
tkdeBar,
width=3.00in,
height=2.08in,
symbolic x coords={{L0,L3}},
xtick=data,
enlarge x limits=0.24,
legend columns=3,
legend style={{at={{(0.5,1.04)}}, anchor=south}}
]
\\nextgroupplot[title={{Sampled ranking}}, ymin=0, ymax=0.60, ylabel={{NDCG@10}}]
\\addplot+[tkdeBar, fill=careblue, draw=careblue] coordinates {{{coord_str(labels, sampled_cf)}}};
\\addplot+[tkdeBar, fill=careorange, draw=careorange] coordinates {{{coord_str(labels, sampled_text)}}};
\\addplot+[tkdeBar, fill=careteal, draw=careteal] coordinates {{{coord_str(labels, sampled_care)}}};
\\legend{{CF, Text, CARE}}
\\nextgroupplot[title={{Full ranking}}, ymin=0, ymax=0.012, ylabel={{NDCG@10}}]
\\addplot+[tkdeBar, fill=careblue, draw=careblue] coordinates {{{coord_str(labels, full_cf)}}};
\\addplot+[tkdeBar, fill=careorange, draw=careorange] coordinates {{{coord_str(labels, full_text)}}};
\\addplot+[tkdeBar, fill=careteal, draw=careteal] coordinates {{{coord_str(labels, full_care)}}};
\\end{{groupplot}}
\\end{{tikzpicture}}
"""
    return "full_ranking_consistency", doc(body)


def main() -> None:
    ensure_dirs()
    jobs = [
        coldness_spectrum("baby"),
        coldness_spectrum("office"),
        coldness_spectrum("sports"),
        fixed_fusion_tradeoff(),
        zero_shot_modality_profile(),
        modality_delta_bars(),
        aggregate_vs_stratified_gap(),
        care_vs_mamex_zero_shot(),
        gate_family_sweep(),
        yelp_cross_domain_dashboard(),
        full_ranking_consistency(),
    ]
    for name, tex in jobs:
        write_and_compile(name, tex)


if __name__ == "__main__":
    main()
