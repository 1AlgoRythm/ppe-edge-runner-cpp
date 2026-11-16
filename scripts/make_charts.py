"""Charts for the README, drawn only from results/*.json.  usage: .venv/bin/python scripts/make_charts.py"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from chart_style import setup, bars, save, legend_top, INK2
from render_readme import load_bench, agg, IMPLS

ROOT = Path(__file__).resolve().parents[1]
runs = load_bench()
(ROOT / "docs").mkdir(exist_ok=True)

fig, ax = setup(1, 3, w=12, h=4.2)
for a, (metric, label) in zip(ax, [("preprocess_ms", "preprocess"), ("inference_ms", "inference"), ("postprocess_ms", "postprocess")]):
    vals = [[agg(runs, ("cascade", v, 0, impl), metric, "median") for v in ("fp32", "int8")] for impl, _ in IMPLS]
    bars(a, ["fp32", "int8"], [l for _, l in IMPLS], vals, "{:.2f}", width=0.26)
    a.set_title(f"{label}, ms per image (median)", fontsize=10, color=INK2, loc="left")
legend_top(fig, [l for _, l in IMPLS])
save(fig, ROOT / "docs/latency_by_stage.png", "C++ vs Python, cascade, ONNX Runtime default threads",
     "Source: results/bench/*.json (median over 3 rounds). Each panel has its own scale: inference is ~98% of the total.")

def tot(groups):
    return (sum(g["matched"] for g in groups.values()), sum(max(g["n_cpp"], g["n_py"]) for g in groups.values()))
vals = []
for mode in ("sameinput", "native"):
    row = []
    for v in ("fp32", "int8"):
        m, n = tot(json.load(open(ROOT / f"results/correctness_cascade_{v}_{mode}.json"))["groups"])
        row.append(100 * m / n)
    vals.append(row)
fig, ax = setup(w=10.5, h=4.2)
bars(ax, ["fp32", "int8"], ["identical network input", "each side's own OpenCV resize"], vals, "{:.1f}%")
ax.set_ylim(0, 108); ax.set_ylabel("% of boxes matched (IoU >= 0.9, same class)"); legend_top(fig, ["identical network input", "each side's own OpenCV resize"])
save(fig, ROOT / "docs/box_agreement.png", "C++ vs Python detections on 40 held-out images",
     "Source: results/correctness_cascade_*.json (persons + PPE items).")
print("wrote docs/latency_by_stage.png, docs/box_agreement.png")
