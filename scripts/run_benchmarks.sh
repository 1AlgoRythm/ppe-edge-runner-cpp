#!/usr/bin/env bash
# Benchmarks C++ (Homebrew ORT), C++ (pip-wheel ORT) and Python back to back, in ROUNDS interleaved rounds, so slow drifts in machine load hit both
# languages alike. One JSON per (round, language, mode, variant, threads) in results/bench/.
# Needs both builds:  cmake -S . -B build && cmake --build build
#                     cmake -S . -B build-wheelort -DORT_LIBRARY=$PWD/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.32.0.dylib && cmake --build build-wheelort
# usage: scripts/run_benchmarks.sh [rounds]     (run from the repo root; roughly 8 min per round)
set -euo pipefail
cd "$(dirname "$0")/.."
ROUNDS="${1:-3}"
PASSES=3; WARMUP=1
PY=.venv/bin/python; CPP=build/ppe_runner; CPPW=build-wheelort/ppe_runner
mkdir -p results/bench
for r in $(seq 1 "$ROUNDS"); do
  for threads in 1 0; do                                   # 1 = single core (embedded-like), 0 = ORT default (all cores)
    for variant in fp32 int8; do
      for mode in cascade person; do
        [ "$mode" = person ] && [ "$threads" = 1 ] && continue    # person-only is measured with default threads only
        flag=""; [ "$mode" = cascade ] && flag="--cascade"
        tag="r${r}_${mode}_${variant}_t${threads}"
        echo "== $tag   load: $(uptime | sed 's/.*load averages*: //')"
        $CPP --variant "$variant" $flag --bench --warmup $WARMUP --passes $PASSES --threads "$threads" --out "results/bench/cpp_$tag.json" | tail -1
        $CPPW --variant "$variant" $flag --bench --warmup $WARMUP --passes $PASSES --threads "$threads" --out "results/bench/cppwheel_$tag.json" | tail -1
        $PY python/ref.py --variant "$variant" $flag --bench --warmup $WARMUP --passes $PASSES --threads "$threads" --out "results/bench/py_$tag.json" | tail -1
      done
    done
  done
done
