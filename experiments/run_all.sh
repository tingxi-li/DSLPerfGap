#!/usr/bin/env bash
# =============================================================================
# run_all.sh  --  master runner for the experiment suite
#
# Runs the full suite SERIALIZED on ONE PINNED GPU so the timing experiments
# never contend (contention would corrupt the median/std confidence intervals
# addressed by significance re-timing).  Default CUDA_VISIBLE_DEVICES=0; overridable.
#
# Order:
#   1. CORRECTNESS first (no timing sensitivity):
#        exp_fp32_gemm.py          (Exp 2: FP32 / TF32 root cause)
#        exp_correctness_edge.py   (edge-case numerical correctness)
#   2. TIMING (serialized, idle GPU):
#        exp_conv_filters.py       (Exp 3: 1/3/5/7 conv regs+latency; RC3)
#        exp_fused_baselines.py    (fused-op baselines)
#        exp_winograd_isolation.py (Exp 4: Winograd eligible-vs-ineligible; RC4)
#        exp_autotune_matmul.py    (autotuned matmul sweep)
#   3. NOTE: hardware-counter collection (Exp 1) is admin-gated and run
#        SEPARATELY via ncu_counters.sh (it needs NVreg_RestrictProfiling... off).
#
# Idempotent / re-runnable: each exp_*.py overwrites its own
# results/<gpu_slug>/<exp>.csv via _harness.write_csv. Logs tee to
# results/<gpu_slug>/run_all.log.
#
# Portable: nothing is sm_89-specific. The SAME script runs on A100/H100 and
# auto-writes to a NEW results/<gpu_slug>/ dir.
#
# Usage:
#   bash experiments/run_all.sh                 # full suite, GPU 0
#   CUDA_VISIBLE_DEVICES=1 bash .../run_all.sh  # pin a different GPU
#   bash experiments/run_all.sh --smoke         # quick smoke (passed to each exp)
# =============================================================================
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON="${PYTHON:-python}"

# --- Pin ONE GPU so timing experiments are serialized & uncontended. ---------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Pass-through args (e.g. --smoke) forwarded to every experiment script. ---
EXTRA_ARGS=("$@")

# --- Correctness scripts run before timing; timing scripts run on idle GPU. ---
CORRECTNESS_EXPS=(
  "exp_fp32_gemm.py"
  "exp_correctness_edge.py"
)
# Each entry is "script.py" or "script.py:extra args" (colon-separated args).
# Same script can fire multiple times with different flags; logs and CSV names
# are derived from the args by the experiment scripts themselves
# (exp_conv_filters.py auto-suffixes its CSV by --shape).
#
# Conv sweep covers four arms: {baseline, --mitigation} x {small, large}.
# This produces conv_{filters,mitigation}_{small,large}.csv -- the four files
# cited by the paper (mitigation.tex). On Ada the large
# shape OOM-guards per row; on A100/H100 (80 GB) all rows populate.
TIMING_EXPS=(
  "exp_conv_filters.py"
  "exp_conv_filters.py:--mitigation"
  "exp_conv_filters.py:--shape large"
  "exp_conv_filters.py:--shape large --mitigation"
  "exp_fused_baselines.py"
  "exp_winograd_isolation.py"
  "exp_autotune_matmul.py"
)

# --- Resolve the per-arch results dir + log path (GPU slug is queried). -------
GPU_SLUG="$(cd "${HERE}" && "${PYTHON}" -c \
  "from _harness import device_slug; print(device_slug())" 2>/dev/null)"
if [[ -z "${GPU_SLUG}" ]]; then
  GPU_SLUG="unknown_gpu"
fi
RESULTS_DIR="${HERE}/results/${GPU_SLUG}"
mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/run_all.log"

# Everything below is tee'd to the per-arch log (truncate at start -> idempotent).
exec > >(tee "${LOG}") 2>&1

echo "######################################################################"
echo "#  experiment suite -- run_all.sh"
echo "#  $(date -u +'%Y-%m-%dT%H:%M:%SZ')   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "#  results -> ${RESULTS_DIR}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "#  passthrough args: ${EXTRA_ARGS[*]}"
echo "######################################################################"

# --- Print the detected GPU via the shared harness banner. -------------------
( cd "${HERE}" && "${PYTHON}" -c "from _harness import banner; banner('run_all')" ) \
  || echo "[run_all] WARNING: could not print GPU banner (is torch+CUDA available?)"

# ---------------------------------------------------------------------------
# Run one experiment script; tolerate a missing sibling (other builders may
# not have landed it yet) and a non-zero rc (keep going, report at the end).
# ---------------------------------------------------------------------------
RC_SUMMARY=()
run_exp() {  # $1 = "script.py" or "script.py:extra args"
  local entry="$1"
  local script="${entry%%:*}"
  local extra=""
  if [[ "${entry}" == *:* ]]; then
    extra="${entry#*:}"
  fi
  local path="${HERE}/${script}"
  local label="${script}"
  [[ -n "${extra}" ]] && label="${script} ${extra}"
  echo ""
  echo "======================================================================"
  echo "[run_all] >>> ${label}  ${EXTRA_ARGS[*]:-}"
  echo "======================================================================"
  if [[ ! -f "${path}" ]]; then
    echo "[run_all] SKIP: ${script} not present yet (built by another author?)."
    RC_SUMMARY+=("SKIP  ${label}")
    return 0
  fi
  local t0 t1 rc
  t0="$(date +%s)"
  # shellcheck disable=SC2086  # intentional word-splitting on ${extra}
  ( cd "${HERE}" && "${PYTHON}" "${path}" ${extra} "${EXTRA_ARGS[@]}" )
  rc=$?
  t1="$(date +%s)"
  if [[ ${rc} -eq 0 ]]; then
    echo "[run_all] OK   ${label}  ($((t1 - t0))s)"
    RC_SUMMARY+=("OK    ${label}  ($((t1 - t0))s)")
  else
    echo "[run_all] FAIL ${label}  (rc=${rc}, $((t1 - t0))s)"
    RC_SUMMARY+=("FAIL  ${label}  (rc=${rc})")
  fi
  return 0   # never abort the suite on one experiment's failure
}

echo ""
echo "##### PHASE 1 -- CORRECTNESS (timing-insensitive) #####################"
for e in "${CORRECTNESS_EXPS[@]}"; do run_exp "${e}"; done

echo ""
echo "##### PHASE 2 -- TIMING (serialized on pinned GPU ${CUDA_VISIBLE_DEVICES}) #####"
for e in "${TIMING_EXPS[@]}"; do run_exp "${e}"; done

echo ""
echo "##### PHASE 3 -- HARDWARE COUNTERS (admin-gated; run SEPARATELY) ######"
cat <<EOF
[run_all] NOTE: Nsight Compute counter collection (Experiment 1) is NOT run by
          run_all.sh because it requires admin to unblock counters
          (NVreg_RestrictProfilingToAdminUsers=0 + reboot). When enabled:

              bash ${HERE}/ncu_counters.sh

          It pre-checks /proc/driver/nvidia/params and exits 2 with the
          unblock instructions if counters are still restricted.
EOF

echo ""
echo "######################################################################"
echo "[run_all] SUMMARY ($(date -u +'%Y-%m-%dT%H:%M:%SZ')):"
for line in "${RC_SUMMARY[@]}"; do echo "    ${line}"; done
echo "[run_all] full log: ${LOG}"
echo "[run_all] CSVs:     ${RESULTS_DIR}/*.csv"
echo "######################################################################"

# Exit non-zero if any present experiment FAILED (SKIP/OK do not fail the run).
for line in "${RC_SUMMARY[@]}"; do
  [[ "${line}" == FAIL* ]] && exit 1
done
exit 0
