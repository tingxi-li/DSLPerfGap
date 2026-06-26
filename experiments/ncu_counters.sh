#!/usr/bin/env bash
# =============================================================================
# ncu_counters.sh  --  turnkey Nsight Compute counter collection
# ASE-2026 #4134 rebuttal, Experiment 1 (the hardware-counter-grounded taxonomy)
#
# Substantiates the counter families R1-Q4 asked for, via the EXACT metric sets
# from REBUTTAL_PROTOCOLS_CRITICAL.md Experiment 1:
#   (A) vectorized / global-load efficiency   (conv vs GEMM vs LayerNorm)
#   (B) registers + spill proxy + occupancy    (RC3 / W13)
#   (C) warp-stall breakdown                    (--section WarpStateStats; RC0)
#   (D) L2 / DRAM                               (RC2b; feeds Exp 5)
#
# Each kernel is profiled with the single-launch harness run_one_kernel.py under
# an NVTX "TARGET" range, so ncu counts EXACTLY ONE invocation (never --set full,
# never a serialized multi-launch run).
#
# ---------------------------------------------------------------------------
# ADMIN GATE: on this machine counter collection is currently BLOCKED.
#   /proc/driver/nvidia/params -> RmProfilingAdminOnly: 1
# Every ncu collection returns ERR_NVGPUCTRPERM until an admin runs the Step-0
# unblock below + reboots (sudo is disallowed for us by project rules).
# This script PRE-CHECKS that flag and exits 2 with guidance if still blocked,
# rather than dumping ERR_NVGPUCTRPERM for every metric.
#
# PORTABILITY: every metric ID below is arch-portable -- the SAME script runs on
# A100/H100. Results auto-namespace under results/<gpu_slug>/ncu/ via the GPU
# slug, so Ada / A100 / H100 never collide.  See experiments/A100_H100_RUNBOOK.md.
# =============================================================================
set -u  # (no -e: we want to keep going across kernels and report per-target rc)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"
RUN_ONE="${HERE}/run_one_kernel.py"
PYTHON="${PYTHON:-python}"
PARAMS_FILE="/proc/driver/nvidia/params"

# Pin one GPU so collection is deterministic & uncontended (overridable).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---------------------------------------------------------------------------
# sudo TileLang recovery.  ncu needs root (RmProfilingAdminOnly=1), but under
# sudo HOME=/root, which hides the invoking user's ~/.local site-packages -->
# `import tilelang` (installed in ~user/.local on aarch64/GH200) fails rc=1,
# while system-installed Triton/cuBLAS still load.  That silently dropped the
# layer_norm/argmax/max_reduction TileLang RC0 targets.  Recover the invoking
# user's HOME + user-site so those targets profile instead of aborting.
if [[ "$(id -u)" == "0" && -n "${SUDO_USER:-}" ]]; then
  _U_HOME="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
  if [[ -n "${_U_HOME}" && -d "${_U_HOME}/.local" ]]; then
    export HOME="${_U_HOME}"
    _U_PYVER="$("${PYTHON}" -c 'import sys;print("python%d.%d"%sys.version_info[:2])' 2>/dev/null)"
    export PYTHONPATH="${_U_HOME}/.local/lib/${_U_PYVER}/site-packages:${PYTHONPATH:-}"
    echo "[ncu_counters] sudo detected -> recovered HOME=${HOME} + user-site for ${SUDO_USER} (TileLang import)"
  fi
fi

# ---------------------------------------------------------------------------
# Step 0 banner: how an admin unblocks counters (printed every run, on top).
# ---------------------------------------------------------------------------
print_unblock() {
  cat <<'EOF'
-----------------------------------------------------------------------------
 STEP 0 -- UNBLOCK NCU COUNTERS (admin-only, ~15 min, one reboot)
-----------------------------------------------------------------------------
 Nsight Compute hardware counters are gated by the NVIDIA driver. Until an
 admin lifts the restriction, EVERY collection returns ERR_NVGPUCTRPERM
 (including launch__registers_per_thread and the default sections).

 Hand the sysadmin (RHEL9 here; sudo is disallowed for the artifact user):

   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
       | sudo tee /etc/modprobe.d/nvidia-profiling.conf
   sudo dracut -f          # RHEL9   (Debian/Ubuntu: sudo update-initramfs -u)
   sudo reboot             # or rmmod/modprobe the nvidia* modules if headless

 Verify afterwards (NO sudo needed):
   grep RmProfilingAdminOnly /proc/driver/nvidia/params     # must read 0

 Fallbacks if admin declines:
   * a one-off  sudo ncu ...   collection, OR
   * the admin-free register/spill path (Triton kernel.n_regs / n_spills,
     ptxas -v, cudaFuncAttributes.numRegs) -- Experiment 3 Path A.

 On a cloud A100/H100 root instance you can run the three lines yourself.
-----------------------------------------------------------------------------
EOF
}

# ---------------------------------------------------------------------------
# Permission pre-check: read RmProfilingAdminOnly and fail fast if blocked.
# ---------------------------------------------------------------------------
check_permissions() {
  echo "[ncu_counters] checking counter permission via: grep RmProfilingAdminOnly ${PARAMS_FILE}"
  # Root bypasses the RmProfilingAdminOnly restriction: even with the flag at 1,
  # `sudo ncu ...` reads counters fine. So when running as root, do NOT fail-fast
  # on the flag -- that is the "fast path, no reboot" option.
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    echo "[ncu_counters] running as root (EUID=0) -- root bypasses RmProfilingAdminOnly;"
    echo "               proceeding with collection regardless of the flag value."
    return 0
  fi
  if [[ ! -r "${PARAMS_FILE}" ]]; then
    echo "[ncu_counters] WARNING: ${PARAMS_FILE} not readable; cannot confirm"
    echo "               counter permission. Proceeding -- ncu may still fail"
    echo "               with ERR_NVGPUCTRPERM if counters are restricted."
    return 0
  fi
  local line val
  line="$(grep RmProfilingAdminOnly "${PARAMS_FILE}" 2>/dev/null)"
  echo "[ncu_counters] ${PARAMS_FILE}: ${line:-<not found>}"
  val="$(echo "${line}" | grep -oE '[0-9]+' | head -n1)"
  if [[ "${val}" == "1" ]]; then
    echo ""
    echo "[ncu_counters] BLOCKED: RmProfilingAdminOnly == 1  ->  ncu would return"
    echo "               ERR_NVGPUCTRPERM for every metric. Failing fast (exit 2)."
    echo "               Apply the Step-0 admin fix above, reboot, then re-run."
    return 2
  fi
  echo "[ncu_counters] OK: counters appear UNBLOCKED (RmProfilingAdminOnly == ${val:-0})."
  return 0
}

# ---------------------------------------------------------------------------
# ncu availability check.
# ---------------------------------------------------------------------------
check_ncu() {
  if [[ ! -x "${NCU}" ]]; then
    if command -v ncu >/dev/null 2>&1; then
      NCU="$(command -v ncu)"
    else
      echo "[ncu_counters] ERROR: ncu not found at '${NCU}' and not on PATH." >&2
      echo "               Set NCU=/path/to/ncu and re-run." >&2
      return 1
    fi
  fi
  echo "[ncu_counters] using ncu: ${NCU}"
  "${NCU}" --version 2>/dev/null | head -n3
  return 0
}

# ---------------------------------------------------------------------------
# Output dir = results/<gpu_slug>/ncu/   (namespaced per arch; queried, not
# hardcoded, so A100/H100 land in their own dirs and never collide with Ada).
# ---------------------------------------------------------------------------
resolve_outdir() {
  local slug
  slug="$(cd "${HERE}" && "${PYTHON}" -c \
    "from _harness import device_slug; print(device_slug())" 2>/dev/null)"
  if [[ -z "${slug}" ]]; then
    slug="unknown_gpu"
    echo "[ncu_counters] WARNING: could not query device slug; using '${slug}'." >&2
  fi
  OUTDIR="${HERE}/results/${slug}/ncu"
  mkdir -p "${OUTDIR}"
  echo "[ncu_counters] GPU slug: ${slug}"
  echo "[ncu_counters] output dir: ${OUTDIR}"
}

# =============================================================================
# The four metric sets -- EXACT strings from REBUTTAL_PROTOCOLS_CRITICAL.md,
# copy-paste-correct for Nsight Compute on sm_89 (ncu 2024.3.2). No spaces.
# =============================================================================

# (A) Vectorized / global-load efficiency  (R1-Q4 #1; RC0b scalar-load test).
#   smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
#     = vectorized-load fraction: 100% => full 32 B/sector (LDG.128-class);
#       lower => scalar / uncoalesced 16-bit loads.
METRICS_A="\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
sm__sass_inst_executed_op_global_ld.sum"

# (B) Register usage + spill proxy + occupancy  (R1-Q4 #2; RC3 / W13).
#   non-zero local_op_ld/st bytes => register spill to local memory;
#   sm__warps_active...pct_of_peak_sustained_active => achieved occupancy.
METRICS_B="\
launch__registers_per_thread,\
launch__occupancy_limit_registers,\
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active"

# (C) Warp-stall breakdown  (R1-Q4 #3; RC0 sync-vs-latency).
#   collected together with --section WarpStateStats.
#   RC0 predicts stalled_barrier dominates the T.serial TileLang kernels;
#   if long_scoreboard dominates instead => memory-latency-bound, not sync-bound.
METRICS_C="\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"

# (D) L2 / DRAM  (RC2b; feeds Exp 5 L2-residency at 16384^2).
METRICS_D="\
lts__t_sector_hit_rate.pct,\
dram__bytes_read.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum"

# ---------------------------------------------------------------------------
# Target (kernel, impl) pairs to profile. Minimal high-value set:
#   conv2d triton        -- RC0b scalar-load + RC3 register pressure (conv side)
#   matmul triton        -- RC3 spill, RC2b L2, GEMM gap (DSL side)
#   matmul pytorch       -- cuBLAS reference (the GEMM column the paper compares to)
#   layer_norm tilelang  -- RC0b scalar 16-bit loads (the central LayerNorm anomaly)
#   argmax tilelang      -- RC0a stalled_barrier (24 __syncthreads/fragment story)
#   max_reduction tilelang -- RC0a sibling reduction kernel
#   logsumexp tilelang_opt -- RC3 register spill: the *optimized* reduction kernel
#                             that recovers on the A100 but spills on sm_90 (Hopper),
#                             loaded from experiments/opt_kernels/logsumexp_opt.py.
# All at "large" (the paper shapes). Override with TARGETS="kernel:impl:size ...".
# ---------------------------------------------------------------------------
DEFAULT_TARGETS=(
  "conv2d:triton:large"
  "matmul:triton:large"
  "matmul:pytorch:large"
  "layer_norm:tilelang:large"
  "argmax:tilelang:large"
  "max_reduction:tilelang:large"
  "logsumexp:tilelang_opt:large"
)

# Allow override:  TARGETS="matmul:triton:large conv2d:triton:large" bash ncu_counters.sh
if [[ -n "${TARGETS:-}" ]]; then
  read -r -a TARGET_LIST <<< "${TARGETS}"
else
  TARGET_LIST=("${DEFAULT_TARGETS[@]}")
fi

# Map metric-set letter -> (env var name, ncu extra args, filename suffix).
set_metrics() {  # $1 = A|B|C|D
  case "$1" in
    A) SET_METRICS="${METRICS_A}"; SET_EXTRA="";                          SET_TAG="loads" ;;
    B) SET_METRICS="${METRICS_B}"; SET_EXTRA="";                          SET_TAG="regs"  ;;
    C) SET_METRICS="${METRICS_C}"; SET_EXTRA="--section WarpStateStats";  SET_TAG="stalls" ;;
    D) SET_METRICS="${METRICS_D}"; SET_EXTRA="";                          SET_TAG="l2dram" ;;
    *) echo "[ncu_counters] BUG: unknown metric set '$1'" >&2; return 1 ;;
  esac
}

# Which kernel name to profile inside the TARGET range.
#   TileLang lowers its compute kernel to "func_kernel", but the unified-API
#   wrapper also launches PyTorch preamble/postamble copies
#   (at::*_elementwise_kernel) INSIDE the range. With a bare --launch-count 1 ncu
#   would grab the first launch = a trivial copy, not the reduction. So for
#   TileLang we pin --kernel-name regex:func_kernel. Triton/cuBLAS expose their
#   compute kernel as the first (only) launch, so they need no filter.
#   Portable: same names on A100/H100 (TileLang/Triton recompile per-arch).
kernel_filter_for() {  # $1 = impl  ->  echoes an ncu --kernel-name regex, or empty
  case "$1" in
    tilelang|tilelang_opt) echo "func_kernel" ;;
    *)                     echo "" ;;
  esac
}

# Run one (kernel, impl, size) x (metric set) collection.
profile_one() {  # $1 kernel  $2 impl  $3 size  $4 set-letter
  local kernel="$1" impl="$2" size="$3" letter="$4"
  set_metrics "${letter}" || return 1
  local out="${OUTDIR}/${kernel}_${impl}_${size}_${SET_TAG}.csv"
  local kfilter; kfilter="$(kernel_filter_for "${impl}")"
  local FILTER_ARGS=""
  [[ -n "${kfilter}" ]] && FILTER_ARGS="--kernel-name regex:${kfilter}"
  echo ""
  echo "----------------------------------------------------------------------"
  echo "[ncu_counters] set ${letter} (${SET_TAG}): ${kernel} / ${impl} / ${size}"
  echo "               -> ${out}"
  echo "----------------------------------------------------------------------"
  # --target-processes all   : Python spawns the CUDA process.
  # --nvtx --nvtx-include "TARGET/" --launch-count 1 : isolate the ONE launch.
  # --csv --log-file <out>    : machine-readable, namespaced output.
  # shellcheck disable=SC2086  (SET_EXTRA intentionally word-splits)
  # shellcheck disable=SC2086  (SET_EXTRA / FILTER_ARGS intentionally word-split)
  "${NCU}" \
    --target-processes all \
    --nvtx --nvtx-include "TARGET/" \
    ${FILTER_ARGS} \
    --launch-count 1 \
    ${SET_EXTRA} \
    --metrics "${SET_METRICS}" \
    --csv --log-file "${out}" \
    "${PYTHON}" "${RUN_ONE}" "${kernel}" "${impl}" "${size}"
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[ncu_counters] WARNING: set ${letter} for ${kernel}/${impl}/${size} exited rc=${rc}"
    echo "               (if rc indicates ERR_NVGPUCTRPERM, counters are still blocked.)"
  fi
  return ${rc}
}

# =============================================================================
# Main
# =============================================================================
main() {
  echo "======================================================================"
  echo " ncu_counters.sh -- ASE-2026 #4134 Exp 1 hardware-counter collection"
  echo "======================================================================"
  print_unblock

  check_ncu || exit 1

  check_permissions
  local perm_rc=$?
  if [[ ${perm_rc} -eq 2 ]]; then
    exit 2   # fail fast with guidance instead of dumping ERR_NVGPUCTRPERM
  fi

  resolve_outdir

  if [[ ! -f "${RUN_ONE}" ]]; then
    echo "[ncu_counters] ERROR: single-launch harness not found: ${RUN_ONE}" >&2
    exit 1
  fi

  echo ""
  echo "[ncu_counters] targets (${#TARGET_LIST[@]}):"
  for t in "${TARGET_LIST[@]}"; do echo "   - ${t}"; done
  echo "[ncu_counters] metric sets per target: A(loads) B(regs) C(stalls) D(l2dram)"

  local total=0 fail=0
  for t in "${TARGET_LIST[@]}"; do
    IFS=':' read -r kernel impl size <<< "${t}"
    size="${size:-large}"
    for letter in A B C D; do
      total=$((total + 1))
      profile_one "${kernel}" "${impl}" "${size}" "${letter}" || fail=$((fail + 1))
    done
  done

  echo ""
  echo "======================================================================"
  echo "[ncu_counters] done: ${total} collections attempted, ${fail} failed."
  echo "[ncu_counters] CSVs in: ${OUTDIR}"
  echo "[ncu_counters] consolidate into the Exp-1 table (RC0/RC2b/RC3/ref) per"
  echo "               REBUTTAL_PROTOCOLS_CRITICAL.md, then -> ncu_counters.csv."
  echo "======================================================================"
  [[ ${fail} -eq 0 ]] && return 0 || return 1
}

main "$@"
