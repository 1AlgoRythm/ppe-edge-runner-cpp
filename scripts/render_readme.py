"""Render the results tables in README.md from results/*.json. Nothing in the tables is typed by hand.

usage: .venv/bin/python scripts/render_readme.py        (rewrites the block between the RESULTS markers)
"""
import glob
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [("preprocess_ms", "preprocess"), ("inference_ms", "inference"), ("postprocess_ms", "postprocess"),
          ("e2e_ms", "total (incl. crops)")]


IMPLS = [("cpp-brew", "C++ (Homebrew ORT)"), ("cpp-wheel", "C++ (pip-wheel ORT)"), ("python", "Python (pip-wheel ORT)")]


def impl_of(meta):
    """The C++ runner can be linked against Homebrew's libonnxruntime or the one inside the Python wheel."""
    if meta["lang"] == "python":
        return "python"
    return "cpp-wheel" if "site-packages" in meta.get("ort_library", "") else "cpp-brew"


def load_bench():
    """{(mode, variant, threads, impl): [json per round]}"""
    runs = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "results/bench/*.json"))):
        d = json.load(open(f))
        m = d["meta"]
        runs[(m["mode"], m["variant"], m["threads"], impl_of(m))].append(d)
    return runs


def agg(runs, key, metric, stat):
    """median over rounds of the per-round `stat` (median, p95 or min)"""
    return st.median(r[metric][stat] for r in runs[key])


def table_impls(runs, mode, threads):
    out = ["| variant | stage | " + " | ".join(f"{label} median (p95)" for _, label in IMPLS) +
           " | Python / C++ pip-wheel ORT | Python / C++ Homebrew ORT |", "|---|---|" + "---|" * (len(IMPLS) + 2)]
    for variant in ("fp32", "int8"):
        keys = {i: (mode, variant, threads, i) for i, _ in IMPLS}
        if any(k not in runs for k in keys.values()):
            out.append(f"| {variant} | FAILED / not run | " + " | ".join("" for _ in range(len(IMPLS) + 2)) + " |")
            continue
        for metric, label in STAGES:
            cells = [f"{agg(runs, keys[i], metric, 'median'):.2f} ({agg(runs, keys[i], metric, 'p95'):.2f})" for i, _ in IMPLS]
            py = agg(runs, keys["python"], metric, "median")
            out.append(f"| {variant} | {label} | " + " | ".join(cells) +
                       f" | {py / agg(runs, keys['cpp-wheel'], metric, 'median'):.2f}x | {py / agg(runs, keys['cpp-brew'], metric, 'median'):.2f}x |")
    return "\n".join(out)


def table_fp32_vs_int8(runs):
    out = ["| implementation | mode | threads | fp32 total median (ms) | int8 total median (ms) | int8 speedup | fp32 inference median | int8 inference median |", "|---|---|---|---|---|---|---|---|"]
    for impl, label in IMPLS:
        for mode, threads in (("cascade", 1), ("cascade", 0), ("person", 0)):
            a, b = (mode, "fp32", threads, impl), (mode, "int8", threads, impl)
            if a in runs and b in runs:
                e32, e8 = agg(runs, a, "e2e_ms", "median"), agg(runs, b, "e2e_ms", "median")
                i32, i8 = agg(runs, a, "inference_ms", "median"), agg(runs, b, "inference_ms", "median")
                out.append(f"| {label} | {mode} | {threads} | {e32:.2f} | {e8:.2f} | {e32 / e8:.2f}x | {i32:.2f} | {i8:.2f} |")
    return "\n".join(out)


def table_spread(runs):
    out = ["| run | rounds | total median per round (ms) | 1-min load average at start of each round |", "|---|---|---|---|"]
    labels = dict(IMPLS)
    for key in sorted(runs):
        mode, variant, threads, impl = key
        rs = runs[key]
        meds = ", ".join(f"{r['e2e_ms']['median']:.1f}" for r in rs)
        loads = ", ".join(f"{r['meta'].get('loadavg_before_after', [float('nan')])[0]:.1f}" for r in rs)
        out.append(f"| {labels[impl]} {mode} {variant} t{threads} | {len(rs)} | {meds} | {loads} |")
    return "\n".join(out)


def table_ort_diag():
    f = ROOT / "results/ort_build_diag.json"
    if not f.exists():
        return "FAILED: results/ort_build_diag.json missing"
    d = json.load(open(f))
    out = ["| implementation | ORT setting | total median (ms) | inference median (ms) |", "|---|---|---|---|"]
    for r in d["runs"]:
        out.append(f"| {r['label']} | {r['setting']} | {r['e2e_median_ms']:.2f} | {r['inference_median_ms']:.2f} |")
    return "\n".join(out)


def table_correctness():
    out = ["| variant | comparison | group | matched / total boxes | C++ boxes | Python boxes | agreement | matched at IoU >= 0.5 | min IoU of matches | max score diff of matches |", "|---|---|---|---|---|---|---|---|---|---|"]
    for variant in ("fp32", "int8"):
        for kind, label in (("sameinput", "same letterboxed input (gated)"), ("native", "each side's own OpenCV resize")):
            f = ROOT / f"results/correctness_cascade_{variant}_{kind}.json"
            if not f.exists():
                out.append(f"| {variant} | {label} | FAILED: {f.name} missing | | | | | | | |")
                continue
            d = json.load(open(f))
            for g, gl in (("persons", "person (stage 1)"), ("items", "PPE items (stage 2)")):
                x = d["groups"][g]
                tot = max(x["n_cpp"], x["n_py"])
                out.append(f"| {variant} | {label} | {gl} | {x['matched']} / {tot} | {x['n_cpp']} | {x['n_py']} | {x['agreement_pct']:.2f}% | "
                           f"{x['matched_iou50']} | {x['min_matched_iou']:.4f} | {x['max_matched_score_diff']:.1e} |")
    return "\n".join(out)


def versions(runs):
    first = lambda impl: next(r for k, rs in runs.items() if k[3] == impl for r in rs)["meta"]
    c, w, p = first("cpp-brew"), first("cpp-wheel"), first("python")
    rd = ROOT / "results/resize_diff.json"
    rz = json.load(open(rd)) if rd.exists() else None
    lines = [f"- C++: OpenCV {c['opencv']}, {c['compiler']}, {c['build']} build; ONNX Runtime {c['onnxruntime']} from Homebrew (`{c['ort_library']}`), "
             f"and a second build of the same code linked against the ONNX Runtime {w['onnxruntime']} inside the Python wheel (`.venv/.../onnxruntime/capi/`)",
             f"- Python reference: ONNX Runtime {p['onnxruntime']} (pip wheel), opencv-python {p['opencv']}, numpy {p['numpy']}, Python {p['python']}",
             f"- CPU: {c['cpu']}; execution provider: CPU only; thresholds conf {c['conf']}, NMS IoU {c['iou']}"]
    if rz:
        lines.append(f"- Letterboxed pixels, C++ (OpenCV {c['opencv']}) vs Python (opencv-python {rz['opencv_python']}): "
                     f"{rz['images_with_any_difference']}/{rz['n_images']} images differ, max |difference| {rz['max_abs_diff_overall']} grey level, "
                     f"{100 * rz['mean_frac_values_differing']:.1f}% of values differ on average (results/resize_diff.json)")
    loads = [x for rs in runs.values() for r in rs for x in r["meta"].get("loadavg_before_after", [])]
    if loads:
        lines.append(f"- Machine load while benchmarking (1-minute load average, 10 cores, other jobs were running): {min(loads):.1f} to {max(loads):.1f}")
    return "\n".join(lines)


def headlines(runs):
    g = lambda mode, v, t, impl, metric="e2e_ms": agg(runs, (mode, v, t, impl), metric, "median")
    pp = lambda v, t, impl: g("cascade", v, t, impl, "preprocess_ms") + g("cascade", v, t, impl, "postprocess_ms")
    out = []
    for v in ("fp32", "int8"):
        out.append(f"- Cascade, {v}, default threads, total per image: C++ pip-wheel ORT {g('cascade', v, 0, 'cpp-wheel'):.1f} ms, "
                   f"Python {g('cascade', v, 0, 'python'):.1f} ms, C++ Homebrew ORT {g('cascade', v, 0, 'cpp-brew'):.1f} ms.")
    out.append(f"- Preprocess + postprocess only, sum of the two medians (cascade, fp32, default threads): C++ {pp('fp32', 0, 'cpp-wheel'):.2f} ms, Python {pp('fp32', 0, 'python'):.2f} ms "
               f"({pp('fp32', 0, 'python') / pp('fp32', 0, 'cpp-wheel'):.1f}x). Inference is {100 * g('cascade', 'fp32', 0, 'cpp-wheel', 'inference_ms') / g('cascade', 'fp32', 0, 'cpp-wheel'):.0f}% "
               f"of the C++ total, so the language only changes the small part.")
    out.append(f"- int8 vs fp32 in C++ (Homebrew ORT, cascade, default threads): {g('cascade', 'fp32', 0, 'cpp-brew') / g('cascade', 'int8', 0, 'cpp-brew'):.2f}x faster; "
               f"with the pip-wheel ORT: {g('cascade', 'fp32', 0, 'cpp-wheel') / g('cascade', 'int8', 0, 'cpp-wheel'):.2f}x "
               f"(1 thread: {g('cascade', 'fp32', 1, 'cpp-wheel') / g('cascade', 'int8', 1, 'cpp-wheel'):.2f}x, i.e. int8 is slower there).")
    for variant in ("fp32", "int8"):
        d = json.load(open(ROOT / f"results/correctness_cascade_{variant}_sameinput.json"))["groups"]
        n = json.load(open(ROOT / f"results/correctness_cascade_{variant}_native.json"))["groups"]
        tot = lambda x: (sum(x[k]["matched"] for k in x), sum(max(x[k]["n_cpp"], x[k]["n_py"]) for k in x))
        a, b = tot(d), tot(n)
        out.append(f"- Box agreement C++ vs Python, {variant}, persons + PPE items together: {a[0]}/{a[1]} = {100 * a[0] / a[1]:.2f}% with identical network input; "
                   f"{b[0]}/{b[1]} = {100 * b[0] / b[1]:.2f}% with each side's own OpenCV resize.")
    return "\n".join(out)


def main():
    runs = load_bench()
    n = next(iter(runs.values()))[0]
    meta = f"{n['n_images']} images x {n['timed_passes']} timed passes (+{n['warmup_passes']} warmup) = {n['n_timings']} timings per run"
    rounds = max(len(v) for v in runs.values())
    block = f"""<!-- RESULTS:BEGIN (generated by scripts/render_readme.py; do not edit by hand) -->
Each run: {meta}, repeated in {rounds} interleaved rounds (the three implementations back to back); every cell is the median over
rounds of that round's median (p95 = median over rounds of that round's 95th percentile). Times are milliseconds per image on the 40 held-out person test images; image decoding is not timed.
In cascade mode a stage is summed over the person run and every PPE crop of that image.

{versions(runs)}

### Headlines

{headlines(runs)}

### Latency, cascade, 1 thread (`--threads 1`)

{table_impls(runs, "cascade", 1)}

### Latency, cascade, ONNX Runtime default threads (`--threads 0`)

{table_impls(runs, "cascade", 0)}

### Latency, person model only, default threads

{table_impls(runs, "person", 0)}

### fp32 vs int8

{table_fp32_vs_int8(runs)}

### Why Homebrew's ONNX Runtime is slower for fp32 (diagnostic)

Person model, fp32, 1 thread, one run each (`scripts/diag_ort_build.sh` -> `results/ort_build_diag.json`).
`mlas.disable_kleidiai=1` is an ONNX Runtime session option that turns off the Arm KleidiAI matrix kernels.

{table_ort_diag()}

### Round-to-round spread (how noisy the machine was)

{table_spread(runs)}

### Box agreement, C++ vs Python reference (cascade, 40 images)

{table_correctness()}
<!-- RESULTS:END -->"""
    readme = ROOT / "README.md"
    text = readme.read_text()
    text = re.sub(r"<!-- RESULTS:BEGIN.*?<!-- RESULTS:END -->", lambda _: block, text, flags=re.S)
    readme.write_text(text)
    print("README results tables rendered from", sum(len(v) for v in runs.values()), "bench files")


if __name__ == "__main__":
    main()
