#!/usr/bin/env bash
# Cross-check of the C++ runner against the Python reference: same models, same 40 images, same preprocessing code,
# same thresholds. Two comparisons per variant (fp32, int8), both cascade mode:
#   sameinput  Python is fed the C++ runner's letterboxed pixels. Isolates the pipeline logic (decode, NMS, crops,
#              ORT). GATED: boxes must agree (same class, IoU >= 0.9) for >= 95% of boxes or the test fails.
#   native     each side uses its own OpenCV for the resize (Homebrew 5.1.0 vs opencv-python wheel). Reported, not gated:
#              it measures how sensitive the detections are to a +-1 pixel difference in the input.
# usage: tests/run_correctness.sh [path/to/ppe_runner]   (run from the repo root; needs .venv, see README)
set -euo pipefail
cd "$(dirname "$0")/.."
RUNNER="${1:-build/ppe_runner}"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "FAILED: .venv missing; run: uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r python/requirements.txt"; exit 2; }
mkdir -p results
LB="${TMPDIR:-/tmp}/ppe_runner_letterboxed"      # overwritten on every run
status=0
for variant in fp32 int8; do
  echo "== cascade $variant"
  "$RUNNER" --variant "$variant" --cascade --dump "results/dets_cpp_cascade_$variant.json" --save-letterbox "$LB/$variant" >/dev/null
  echo " sameinput (gated):"
  "$PY" python/ref.py --variant "$variant" --cascade --lb-dir "$LB/$variant" --dump "results/dets_py_cascade_${variant}_sameinput.json" >/dev/null
  "$PY" python/compare.py "results/dets_cpp_cascade_$variant.json" "results/dets_py_cascade_${variant}_sameinput.json" \
        --out "results/correctness_cascade_${variant}_sameinput.json" || status=1
  echo " native (reported, not gated):"
  "$PY" python/ref.py --variant "$variant" --cascade --dump "results/dets_py_cascade_${variant}_native.json" >/dev/null
  "$PY" python/compare.py "results/dets_cpp_cascade_$variant.json" "results/dets_py_cascade_${variant}_native.json" \
        --out "results/correctness_cascade_${variant}_native.json" --min-agreement 0 || true
done
"$PY" python/resize_diff.py "$LB/fp32" results/resize_diff.json
exit $status
