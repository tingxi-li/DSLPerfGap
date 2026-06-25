#!/usr/bin/env bash
# regen_profile.sh -- regenerate the ViperBench profile for THIS GPU, non-destructively.
# Produces ViperBench/results/profile.<slug>.csv (untuned: pytorch/triton/tilelang) and,
# with --tuned, profile.<slug>.tuned.csv (adds triton_tuned/tilelang_tuned via an arch
# sweep). The live profile.csv is snapshotted and RESTORED afterward.
#
# GOTCHA: ViperBench/results/profile.csv is NOT per-arch namespaced -- benchmark*.py
# overwrite it in place. This script saves the named copy and restores the original.
#
# Run with clocks locked (see lock_clocks.sh) for stable numbers.
# Usage: bash experiments/repro/regen_profile.sh [--tuned]
set -u
DO_TUNED=0; [ "${1:-}" = "--tuned" ] && DO_TUNED=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VB="${REPO}/ViperBench"; RES="${VB}/results"
VENV="${PYTHON:-/home/ubuntu/dslperf-venv/bin/python}"
# ViperBench profile uses the SHORT device name (A100-SXM4-40GB, H100-80GB-HBM3),
# NOT the experiments/results _harness slug (NVIDIA_...). Keep this convention.
SLUG="$("$VENV" -c 'import torch,re;print(re.sub(r"\s+","-",torch.cuda.get_device_name(0).replace("NVIDIA ","")).strip("-"))')"
SNAP="$(mktemp)"
echo "### regen profile for ${SLUG}"
cp "${RES}/profile.csv" "$SNAP"; echo "snapshotted live profile.csv -> $SNAP"

( cd "$VB" && PYTHONPATH=. "$VENV" benchmark.py )            # pytorch + triton
( cd "$VB" && PYTHONPATH=. "$VENV" benchmark_tilelang.py )   # merge tilelang
cp "${RES}/profile.csv" "${RES}/profile.${SLUG}.csv"
echo "wrote ${RES}/profile.${SLUG}.csv ($(wc -l < "${RES}/profile.${SLUG}.csv") rows)"

if [ "$DO_TUNED" = 1 ]; then
  cp "${RES}/tuning_cache.json" "${SNAP}.cache" 2>/dev/null || true
  ( cd "$VB" && PYTHONPATH=. "$VENV" -m tuning.sweep --all )   # populates tuning_cache.json with <slug> keys
  ( cd "$VB" && PYTHONPATH=. "$VENV" benchmark_tuned.py )      # merges triton_tuned/tilelang_tuned into profile.csv
  cp "${RES}/profile.csv" "${RES}/profile.${SLUG}.tuned.csv"
  echo "wrote ${RES}/profile.${SLUG}.tuned.csv ($(wc -l < "${RES}/profile.${SLUG}.tuned.csv") rows)"
fi

cp "$SNAP" "${RES}/profile.csv"; rm -f "$SNAP"
echo "### restored live profile.csv; done"
