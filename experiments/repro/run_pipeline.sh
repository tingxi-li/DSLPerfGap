#!/usr/bin/env bash
# run_pipeline.sh -- full locked-clock measurement pipeline for one GPU.
# Produces the SAME artifact set the A100-SXM4 primary run produced, so a new arch
# (e.g. H100) is fully consistent. Run AFTER lock_clocks.sh discovery told you the
# sustained clock.
#
# Usage:
#   bash experiments/repro/run_pipeline.sh GR_MHZ MEM_MHZ
#   e.g. A100-SXM4: bash .../run_pipeline.sh 1215 1215
#        H100:      bash .../run_pipeline.sh <sustained_gr> <sustained_mem>   (from lock_clocks.sh)
#
# Steps: lock -> run_all (timing) -> ncu counters (sudo) -> consolidate -> significance -> reset.
set -u
GR="${1:?need GR_MHZ (run lock_clocks.sh first to find the sustained value)}"
MEM="${2:?need MEM_MHZ}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # experiments/
REPO="$(cd "${HERE}/.." && pwd)"
VENV="${PYTHON:-/home/ubuntu/dslperf-venv/bin/python}"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"
SLUG="$("$VENV" -c 'import torch,re;print(re.sub(r"[^A-Za-z0-9._-]+","_",torch.cuda.get_device_name(0)).strip("_"))')"
OUT="${REPO}/experiments/results/${SLUG}"
echo "### GPU slug: ${SLUG}  ->  ${OUT}"

echo "### [1/6] Lock clocks ${GR}/${MEM} (separate cmds; driver rejects combined) ###"
sudo nvidia-smi -i 0 -pm 1 >/dev/null
sudo nvidia-smi -i 0 -lgc "$GR"; sudo nvidia-smi -i 0 -lmc "$MEM"
nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader

echo "### [2/6] Timing + correctness suite (run_all.sh) ###"
( cd "$HERE" && PYTHON="$VENV" bash run_all.sh ); echo "run_all exit: $?"

echo "### [3/6] NCU hardware counters (sudo; root bypasses RmProfilingAdminOnly=1) ###"
( cd "$HERE" && sudo env PYTHON="$VENV" NCU="$NCU" bash ncu_counters.sh ); echo "ncu exit: $?"
sudo chown -R "$(id -un):$(id -gn)" "${OUT}/ncu" 2>/dev/null || true

echo "### [4/6] Consolidate ncu -> ncu_summary.csv ###"
( cd "$HERE" && "$VENV" consolidate_ncu.py ); echo "consolidate exit: $?"

echo "### [5/6] Locked-clock significance ###"
( cd "$HERE" && "$VENV" exp_significance.py --lock-gr-mhz "$GR" --lock-mem-mhz "$MEM" ); echo "significance exit: $?"

echo "### [6/6] Reset clocks ###"
sudo nvidia-smi -i 0 -rgc; sudo nvidia-smi -i 0 -rmc
echo "### DONE -> ${OUT}/  (run regen_profile.sh + retime_mitigation.py + verify_opt_kernels.py next, see runbook) ###"
ls -1 "${OUT}"/*.csv 2>/dev/null
