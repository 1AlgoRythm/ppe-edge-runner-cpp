#!/usr/bin/env bash
# Why is the C++ build on Homebrew's ONNX Runtime slower than Python for fp32? Same model, 1 thread, with and without
# the KleidiAI kernels (session option mlas.disable_kleidiai), for all three implementations.
# Writes results/diag/*.json and the summary results/ort_build_diag.json.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/diag
for setting in default mlas.disable_kleidiai=1; do
  extra=(); [ "$setting" != default ] && extra=(--ort-config "$setting")
  t="${setting//[=.]/_}"
  common=(--variant fp32 --bench --warmup 1 --passes 2 --threads 1)
  build/ppe_runner "${common[@]}" "${extra[@]+"${extra[@]}"}" --out "results/diag/cppbrew_$t.json" | tail -1
  build-wheelort/ppe_runner "${common[@]}" "${extra[@]+"${extra[@]}"}" --out "results/diag/cppwheel_$t.json" | tail -1
  .venv/bin/python python/ref.py "${common[@]}" "${extra[@]+"${extra[@]}"}" --out "results/diag/py_$t.json" | tail -1
done
.venv/bin/python - <<'PY'
import glob, json
labels = {"cppbrew": "C++ (Homebrew ORT)", "cppwheel": "C++ (pip-wheel ORT)", "py": "Python (pip-wheel ORT)"}
runs = []
for impl in labels:
    for t, setting in (("default", "default"), ("mlas_disable_kleidiai_1", "mlas.disable_kleidiai=1")):
        d = json.load(open(f"results/diag/{impl}_{t}.json"))
        runs.append({"label": labels[impl], "setting": setting, "e2e_median_ms": d["e2e_ms"]["median"],
                     "inference_median_ms": d["inference_ms"]["median"], "ort_library": d["meta"]["ort_library"],
                     "loadavg_before_after": d["meta"].get("loadavg_before_after")})
json.dump({"model": "person_fp32", "threads": 1, "runs": runs}, open("results/ort_build_diag.json", "w"), indent=1)
PY
